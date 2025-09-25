# -*- coding: utf-8 -*-
"""
Steerable Large Language Model Generation with Activation Engineering.

This script provides a comprehensive toolkit for controlling the stylistic attributes of
Large Language Models (LLMs) using activation steering. It allows for the extraction
of "style vectors" from models fine-tuned on contrasting datasets (e.g., formal vs. informal)
and then uses these vectors to guide the generation of a base model at inference time.

The toolkit supports three main modes of operation:
1.  analyze: Extracts activation differences from base and LoRA-finetuned models,
    then computes a robust style vector using methods like PCA denoising.
2.  generate: Starts an interactive session to generate text with real-time
    control over style by applying one or two style vectors.
3.  generate_batch: Processes a JSONL file of prompts to generate styled
    responses in batch, useful for evaluation.
"""

import argparse
import gc
import functools
import os
import json
import re
from collections import defaultdict
from typing import List, Dict, Any, Callable, Tuple, Optional, Union

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer, PreTrainedModel
from peft import PeftModel
from datasets import load_dataset, Dataset
from sklearn.decomposition import PCA
from tqdm import tqdm
import numpy as np

# ======================================================================================
# Core Utility Classes and Functions
# ======================================================================================

class ActivationCollector:
    """
    A class to capture activations from specified layers during a forward pass using PyTorch hooks.
    """
    def __init__(self, capture_last_token_only: bool = False):
        self.activations: Dict[str, List[torch.Tensor]] = defaultdict(list)
        self.capture_last_token_only = capture_last_token_only

    def hook_fn(self, layer_name: str, module: torch.nn.Module, inp: Any, outp: torch.Tensor):
        """The hook function to be registered."""
        activation_to_capture = outp
        if self.capture_last_token_only:
            if activation_to_capture.dim() == 3:  # [batch, seq_len, hidden_dim]
                activation_to_capture = activation_to_capture[:, -1, :]
            elif activation_to_capture.dim() == 2:  # [seq_len, hidden_dim]
                activation_to_capture = activation_to_capture[-1, :]
        
        self.activations[layer_name].append(activation_to_capture.detach().cpu())

    def clear(self):
        """Clears all stored activations."""
        self.activations.clear()

class ActivationSteering:
    """
    A class to modify (steer) activations at inference time using PyTorch hooks.
    """
    def __init__(self, style_vector: torch.Tensor, alpha: float, device: str):
        self.style_vector = style_vector.to(device)
        self.alpha = alpha

    def hook_fn(self, module: torch.nn.Module, inp: Any, outp: torch.Tensor) -> torch.Tensor:
        """The hook function that applies the steering."""
        if outp.dim() >= 2:
            if outp.dim() == 3: # [batch, seq_len, hidden_dim]
                outp[:, -1, :] += self.alpha * self.style_vector
            else: # [seq_len, hidden_dim]
                outp[-1, :] += self.alpha * self.style_vector
        else:
            print(f"Warning: ActivationSteering hook received a tensor with unexpected dimensions: {outp.shape}")
        return outp

def parse_layer_range(range_str: str) -> List[int]:
    """Parses a layer range string (e.g., '15' or '10-20,30-40') into a list of layer indices."""
    layers = []
    if not range_str:
        return layers
    
    parts = range_str.split(',')
    for part in parts:
        part = part.strip()
        if '-' in part:
            try:
                start, end = map(int, part.split('-'))
                layers.extend(list(range(start, end + 1)))
            except ValueError:
                raise ValueError(f"Invalid layer range format: {range_str}")
        else:
            layers.append(int(part))
            
    return sorted(list(set(layers)))

def setup_environment(
    args: argparse.Namespace, 
    load_data: bool = False, 
    skip_layers: bool = False
) -> Union[Tuple[str, PreTrainedModel, PreTrainedTokenizer, List[str]], Tuple[str, PreTrainedModel, PreTrainedTokenizer, List[str], Dataset, Dataset]]:
    """
    Unified setup for environment, model, tokenizer, and target layers for all modes.
    Optionally loads datasets.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    print(f"Loading base model from '{args.base_model_path}'...")
    model = AutoModelForCausalLM.from_pretrained(args.base_model_path, torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    target_layer_names = []
    if not skip_layers and args.target_layers:
        target_layer_indices = parse_layer_range(args.target_layers)
        target_layer_names = [f"model.layers.{i}.mlp" for i in target_layer_indices]
        print(f"Target layers: {', '.join(target_layer_names)}")
    
    if load_data:
        print(f"Loading dataset from '{args.dataset_name}'...")
        dataset = load_dataset("json", data_files=args.dataset_name, split='train')
        # Assuming dataset has 'prompt', 'formal_response', and 'informal_response' columns
        formal_dataset = dataset.map(lambda x: {'response': x['formal_response'], 'prompt': x['prompt']})
        informal_dataset = dataset.map(lambda x: {'response': x['informal_response'], 'prompt': x['prompt']})
        return device, model, tokenizer, target_layer_names, formal_dataset, informal_dataset

    torch.set_float32_matmul_precision('high')
    return device, model, tokenizer, target_layer_names

def get_activations_batched(
    model: PreTrainedModel, 
    tokenizer: PreTrainedTokenizer, 
    dataset: Dataset, 
    target_layer_names: List[str], 
    lora_path: Optional[str] = None, 
    batch_size: int = 8,
    instruct: bool = True
) -> Dict[str, List[torch.Tensor]]:
    """
    Fetches model activations for a given dataset in batches.
    Can be used for both the base model and a LoRA-adapted model.
    """
    activation_collector = ActivationCollector(capture_last_token_only=False) # Capture full sequence
    final_activations: Dict[str, List[torch.Tensor]] = defaultdict(list)
    
    all_modules = dict(model.named_modules())
    target_modules = {name: all_modules[name] for name in target_layer_names}

    if lora_path:
        active_model = PeftModel.from_pretrained(model, lora_path)
        desc = f"Processing LoRA ({os.path.basename(lora_path)})"
    else:
        active_model = model
        desc = "Processing Base Model"
    active_model.eval()
    
    handles = [
        module.register_forward_hook(functools.partial(activation_collector.hook_fn, name))
        for name, module in target_modules.items()
    ]
        
    def collate_fn(batch: List[Dict]) -> Tuple[Dict[str, torch.Tensor], List[int]]:
        prompt_lengths = []
        texts_to_tokenize = []
        for item in batch:
            if instruct:
                prompt_messages = [{"role": "user", "content": item['prompt']}]
                prompt_text = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
                prompt_token_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
                prompt_lengths.append(len(prompt_token_ids))
                
                full_messages = prompt_messages + [{"role": "assistant", "content": item['response']}]
                full_text = tokenizer.apply_chat_template(full_messages, tokenize=False)
                texts_to_tokenize.append(full_text)
            else:
                prompt_lengths.append(0) 
                full_text = item.get('prompt', '') + item.get('response', '')
                texts_to_tokenize.append(full_text)
            
        inputs = tokenizer(texts_to_tokenize, return_tensors="pt", padding=True, truncation=True, max_length=1024)
        return inputs, prompt_lengths

    data_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)

    for batch_inputs, prompt_lengths in tqdm(data_loader, desc=desc):
        batch_inputs = {k: v.to(model.device) for k, v in batch_inputs.items()}
        activation_collector.clear()
        with torch.no_grad():
            active_model(**batch_inputs)
        
        for layer_name, full_activations_batch in activation_collector.activations.items():
            activations_tensor = full_activations_batch[0] # The hook gets the output for the whole batch
            for i in range(activations_tensor.size(0)):
                seq_len = batch_inputs['attention_mask'][i].sum().item()
                
                if instruct:
                    prompt_len = prompt_lengths[i]
                    # We are interested in the activations of the response tokens
                    response_activations = activations_tensor[i, prompt_len:seq_len, :]
                    if response_activations.size(0) > 0:
                        # Average the activations across the response tokens
                        final_activations[layer_name].append(response_activations.mean(dim=0).cpu())
                else: # Non-instruct mode: average over the whole sequence
                    full_activations = activations_tensor[i, :seq_len, :]
                    if full_activations.size(0) > 0:
                        final_activations[layer_name].append(full_activations.mean(dim=0).cpu())
    
    for handle in handles:
        handle.remove()
        
    if lora_path:
        del active_model
        gc.collect()
        torch.cuda.empty_cache()
        
    return final_activations

def geometric_median(points: np.ndarray, eps: float = 1e-5, max_iter: int = 1000) -> np.ndarray:
    """Calculates the geometric median of a set of points using the Weiszfeld algorithm."""
    if points.ndim == 1: 
        return np.median(points)
        
    y = np.mean(points, axis=0)
    for _ in range(max_iter):
        diffs = points - y
        dists = np.linalg.norm(diffs, axis=1)
        non_zero_dists = dists > eps
        if not np.any(non_zero_dists): break # All points are the same as the median
        
        weights = 1.0 / dists[non_zero_dists]
        y_new = np.sum(points[non_zero_dists] * weights[:, np.newaxis], axis=0) / np.sum(weights)
        
        if np.linalg.norm(y - y_new) < eps:
            return y_new
        y = y_new
    return y

def find_optimal_k(X, max_k=100, variance_threshold=0.90):
    """Automatically selects the number of PCA components based on cumulative explained variance."""
    # Ensure n_components is valid
    valid_max_k = min(max_k, X.shape[1], X.shape[0] - 1)
    if valid_max_k <= 0: return 1

    pca = PCA(n_components=valid_max_k)
    pca.fit(X)
    cumulative_var = np.cumsum(pca.explained_variance_ratio_)
    k = np.argmax(cumulative_var >= variance_threshold) + 1
    return k

def align_vector_direction(vector_to_align: np.ndarray, reference_vector: np.ndarray) -> np.ndarray:
    """Aligns the direction of a vector to match a reference vector by checking the dot product."""
    dot_product = np.dot(vector_to_align, reference_vector)
    if dot_product < 0:
        return -vector_to_align
    return vector_to_align

def compute_robust_style_vector(diff_vectors: np.ndarray) -> np.ndarray:
    """
    Computes a robust style vector from a set of difference vectors using PCA for denoising
    and mean aggregation in the latent space.
    """
    # 1. Use the mean direction as a stable reference for final alignment
    reference_direction = np.mean(diff_vectors, axis=0)

    # 2. Automatically select the optimal number of PCA components
    k = find_optimal_k(diff_vectors)
    print(f"  - Using {k} principal components for denoising.")
    
    # 3. Center the data and apply PCA
    mu = np.mean(diff_vectors, axis=0)
    X_centered = diff_vectors - mu
    pca = PCA(n_components=k)
    # Project data into the low-dimensional principal component space
    P_projected = pca.fit_transform(X_centered)
    
    # 4. Aggregate in the low-dimensional space (using mean for simplicity)
    m_star_center = P_projected.mean(axis=0)

    # 5. Project the low-dimensional center point back into the original high-dimensional space
    v_unaligned_direction = pca.inverse_transform(m_star_center)
    
    # 6. Align the direction of the resulting vector with the original mean direction
    # This ensures the semantic direction is consistent, regardless of PCA's arbitrary component signs.
    v_aligned_direction = align_vector_direction(v_unaligned_direction, reference_direction - mu)
    
    # 7. Add the mean back to get the final vector
    v_robust = mu + v_aligned_direction

    return v_robust

# ======================================================================================
# Core Function Modes
# ======================================================================================

def analyze_activations(args: argparse.Namespace):
    """
    Extracts all four sets of activations, computes the style vector using the specified method,
    and saves it to a file.
    """
    print(f"--- Starting activation analysis (Batch size: {args.batch_size}, Method: {args.method}) ---")
    device, base_model, tokenizer, target_layer_names, formal_dataset, informal_dataset = setup_environment(args, load_data=True)
    get_activations_fn = functools.partial(get_activations_batched, model=base_model, tokenizer=tokenizer, target_layer_names=target_layer_names, batch_size=args.batch_size, instruct=args.instruct)
    
    print("\nStep 1/4: Getting activations from Formal LoRA...")
    formal_lora_activations = get_activations_fn(dataset=formal_dataset, lora_path=args.formal_lora_path)
    
    print("\nStep 2/4: Getting activations from Informal LoRA...")
    informal_lora_activations = get_activations_fn(dataset=informal_dataset, lora_path=args.informal_lora_path)
    
    print("\nStep 3/4: Getting activations from Base Model on Formal Data...")
    base_on_formal_activations = get_activations_fn(dataset=formal_dataset)
    
    print("\nStep 4/4: Getting activations from Base Model on Informal Data...")
    base_on_informal_activations = get_activations_fn(dataset=informal_dataset)

    print("\n--- All activations extracted, computing style vectors ---")
    final_vectors = {}
    if args.method == 'pca_denoise':
        for layer_name in tqdm(target_layer_names, desc="Computing robust vectors with PCA Denoising"):
            formal_lora_tensor = torch.stack(formal_lora_activations[layer_name])
            informal_lora_tensor = torch.stack(informal_lora_activations[layer_name])
            base_formal_tensor = torch.stack(base_on_formal_activations[layer_name])
            base_informal_tensor = torch.stack(base_on_informal_activations[layer_name])
            
            # The core of the method: calculate the difference of differences
            diff_of_diffs = (formal_lora_tensor - base_formal_tensor) - (informal_lora_tensor - base_informal_tensor)

            v_robust = compute_robust_style_vector(diff_of_diffs.cpu().numpy())
            final_vectors[layer_name] = torch.tensor(v_robust, dtype=torch.float32)
    elif args.method == 'mean':
        for layer_name in tqdm(target_layer_names, desc="Computing vectors with simple mean"):
            formal_lora_tensor = torch.stack(formal_lora_activations[layer_name])
            informal_lora_tensor = torch.stack(informal_lora_activations[layer_name])
            base_formal_tensor = torch.stack(base_on_formal_activations[layer_name])
            base_informal_tensor = torch.stack(base_on_informal_activations[layer_name])
            
            diff_of_diffs = (formal_lora_tensor - base_formal_tensor) - (informal_lora_tensor - base_informal_tensor)
            final_vectors[layer_name] = diff_of_diffs.mean(axis=0)
    else:
        raise ValueError(f"Unknown method: {args.method}")

    print(f"\nComputation complete! Saving style vectors to: {args.output_path}")
    torch.save(final_vectors, args.output_path)
    print("Vectors saved successfully!")


def interactive_loop(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    device: str,
    all_analyzed_layers: List[str],
    vector_fn1: Callable[[str], torch.Tensor],
    vector_fn2: Optional[Callable[[str], torch.Tensor]] = None,
    instruct: bool = True
):
    """
    Main loop for interactive generation.
    Supports single-vector or dual-vector steering.
    """
    all_modules = dict(model.named_modules())
    is_dual_vector_mode = vector_fn2 is not None

    if is_dual_vector_mode:
        print("\nDual-vector steering mode activated. You will be prompted for two alpha values.")
    else:
        print("\nSingle-vector steering mode activated.")

    while True:
        prompt = input("\nEnter your prompt ('quit' to exit): ")
        if prompt.lower() == 'quit':
            break

        alpha1, alpha2 = 0.0, 0.0
        try:
            alpha_str1 = input(f"Enter {'first' if is_dual_vector_mode else ''} style strength Alpha 1 (e.g., 1.5): ")
            alpha1 = float(alpha_str1)
            if is_dual_vector_mode:
                alpha_str2 = input("Enter second style strength Alpha 2 (e.g., -2.0): ")
                alpha2 = float(alpha_str2)
        except ValueError:
            print("❌ Invalid alpha value. Please enter a number.")
            continue

        layers_to_steer_str = input("Enter layers to intervene on (e.g., '15' or '10-20,30'): ")
        try:
            layer_indices_to_steer = parse_layer_range(layers_to_steer_str)
            target_layer_names = [f"model.layers.{i}.mlp" for i in layer_indices_to_steer]
            
            unavailable_layers = [name for name in target_layer_names if name not in all_analyzed_layers]
            if unavailable_layers:
                print(f"❌ Error: Requested layers {', '.join(unavailable_layers)} not found in the loaded activation file. Please try again.")
                continue
            
            if not target_layer_names:
                print("⚠️ No layers selected. Generating with the base model.")
        except ValueError as e:
            print(f"❌ Error: {e}")
            continue

        handles = []
        try:
            combined_vector = {}
            # Pre-calculate combined steering vectors for each layer
            for layer_name in target_layer_names:
                total_steering_vector = torch.zeros(model.config.hidden_size, device=device)
                
                if alpha1 != 0:
                    style_vector1 = vector_fn1(layer_name)
                    norm1 = torch.linalg.norm(style_vector1)
                    if norm1 > 0:
                        total_steering_vector += (alpha1 * style_vector1) / norm1
                    else:
                         print(f"Warning: Vector 1 for layer {layer_name} has zero norm. Skipping.")

                if is_dual_vector_mode and alpha2 != 0:
                    style_vector2 = vector_fn2(layer_name)
                    norm2 = torch.linalg.norm(style_vector2)
                    if norm2 > 0:
                        total_steering_vector += (alpha2 * style_vector2) / norm2
                    else:
                        print(f"Warning: Vector 2 for layer {layer_name} has zero norm. Skipping.")

                # Register one hook per layer with the final combined vector
                if torch.linalg.norm(total_steering_vector) > 0:
                    steering_hook = ActivationSteering(total_steering_vector, alpha=1.0, device=device) # Alpha is baked in
                    handles.append(all_modules[layer_name].register_forward_hook(steering_hook.hook_fn))
                
            if instruct:
                messages = [{"role": "user", "content": prompt}]
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                text = prompt
            inputs = tokenizer(text, return_tensors="pt").to(model.device)

            generation_kwargs = {
                "max_new_tokens": 512, "do_sample": False,
                "pad_token_id": tokenizer.eos_token_id,
            }
            with torch.no_grad():
                outputs = model.generate(**inputs, **generation_kwargs)
        finally:
            for handle in handles:
                handle.remove()

        response_text = tokenizer.decode(outputs[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)

        print("\n" + "="*10 + " Model Output " + "="*10)
        print(response_text)
        print("="*32)

def generate_interactive(args: argparse.Namespace):
    """
    Main function for interactive generation. Loads one or two activation files.
    """
    num_vectors = len(args.activations_paths)
    print(f"--- Starting Interactive Generation (loading {num_vectors} activation file(s)) ---")
    device, model, tokenizer, _ = setup_environment(args, skip_layers=True)

    def create_vector_calculator(activations_data: Dict) -> Callable[[str], torch.Tensor]:
        """Creates a function that returns a style vector for a given layer."""
        def get_vector_for_layer(layer_name: str) -> torch.Tensor:
            vector = activations_data.get(layer_name, torch.zeros(model.config.hidden_size, device=device))
            if not isinstance(vector, torch.Tensor):
                vector = torch.tensor(vector, device=device, dtype=torch.float32)
            return vector
        return get_vector_for_layer

    print(f"Loading first activation data from '{args.activations_paths[0]}'...")
    activations1 = torch.load(args.activations_paths[0], map_location=device)
    all_analyzed_layers = list(activations1.keys())
    vector_fn1 = create_vector_calculator(activations1)
    
    vector_fn2 = None
    if num_vectors == 2:
        print(f"Loading second activation data from '{args.activations_paths[1]}'...")
        activations2 = torch.load(args.activations_paths[1], map_location=device)
        if set(activations2.keys()) != set(all_analyzed_layers):
            print("Warning: The set of layers in the two activation files is different. This may cause unexpected behavior.")
        vector_fn2 = create_vector_calculator(activations2)

    interactive_loop(
        model, tokenizer, device, all_analyzed_layers,
        vector_fn1=vector_fn1,
        vector_fn2=vector_fn2,
        instruct=args.instruct
    )

def generate_batch(args: argparse.Namespace):
    """Main function for batch generation from a JSONL file."""
    print("--- Starting Batch Generation ---")
    device, model, tokenizer, target_layer_names = setup_environment(args)
    
    print(f"Loading activation data from '{args.activations_path}'...")
    style_vectors_data = torch.load(args.activations_path, map_location=device)
    
    print(f"Loading test data from '{args.input_jsonl}'...")
    test_data = []
    with open(args.input_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                test_data.append(json.loads(line.strip()))
    print(f"Loaded {len(test_data)} test prompts.")
    
    # Check for existing results to avoid re-computation
    existing_indices = set()
    if os.path.exists(args.output_jsonl):
        with open(args.output_jsonl, 'r', encoding='utf-8') as f_out:
            for line in f_out:
                if line.strip():
                    try:
                        existing_indices.add(json.loads(line)['index'])
                    except (json.JSONDecodeError, KeyError):
                        continue
        if existing_indices:
            print(f"Found {len(existing_indices)} existing results. Skipping them.")
    
    remaining_data = [(i, item) for i, item in enumerate(test_data) if i not in existing_indices]
    if not remaining_data:
        print("All data has already been processed. Exiting.")
        return
    
    print(f"Processing {len(remaining_data)} new prompts.")
    
    # Prepare steering vectors once
    steering_vectors = {}
    for layer_name in target_layer_names:
        style_vector = style_vectors_data.get(layer_name)
        if style_vector is not None:
            if not isinstance(style_vector, torch.Tensor):
                style_vector = torch.tensor(style_vector, device=device, dtype=torch.float32)
            norm = torch.linalg.norm(style_vector)
            if norm > 0:
                steering_vectors[layer_name] = (style_vector / norm) * args.alpha
            else:
                print(f"Warning: Vector for layer {layer_name} has zero norm. It will not be used.")
        else:
            print(f"Warning: No vector found for target layer {layer_name}.")
    
    all_modules = dict(model.named_modules())
    
    with open(args.output_jsonl, 'a', encoding='utf-8') as f_out:
        for i in tqdm(range(0, len(remaining_data), args.generation_batch_size), desc="Batch Progress"):
            batch_data = remaining_data[i:i+args.generation_batch_size]
            indices, items = zip(*batch_data)
            
            batch_prompts = []
            for item in items:
                prompt = item['prompt']
                if args.instruct:
                    messages = [{"role": "user", "content": prompt}]
                    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                else:
                    text = prompt
                batch_prompts.append(text)

            inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(model.device)
            
            handles = []
            if args.alpha != 0:
                for layer_name in steering_vectors:
                    steering_hook = ActivationSteering(steering_vectors[layer_name], alpha=1.0, device=device) # Alpha is pre-multiplied
                    handles.append(all_modules[layer_name].register_forward_hook(steering_hook.hook_fn))
            
            try:
                generation_kwargs = {
                    "max_new_tokens": args.max_new_tokens, 
                    "do_sample": args.do_sample, 
                    "top_p": args.top_p, 
                    "temperature": args.temperature, 
                    "pad_token_id": tokenizer.eos_token_id, 
                    "repetition_penalty": args.repetition_penalty
                }
                
                with torch.no_grad():
                    outputs = model.generate(**inputs, **generation_kwargs)
                
                for j, (idx, item) in enumerate(batch_data):
                    response_ids = outputs[j, inputs.input_ids.shape[1]:]
                    response_text = tokenizer.decode(response_ids, skip_special_tokens=True)
                    
                    result = {
                        'index': idx,
                        'prompt': item['prompt'],
                        'generated_text': response_text,
                        'alpha': args.alpha,
                    }
                    f_out.write(json.dumps(result, ensure_ascii=False) + '\n')
            finally:
                for handle in handles:
                    handle.remove()
    
    print("✅ Batch generation complete.")

# ======================================================================================
# Main Function & Argument Parsing
# ======================================================================================

def main():
    """Main function to parse arguments and dispatch to the correct mode."""
    parser = argparse.ArgumentParser(description="A toolkit for steerable LLM generation using activation engineering.")
    subparsers = parser.add_subparsers(dest='mode', required=True, help="The mode to run: 'analyze', 'generate', or 'generate_batch'.")

    # Common arguments for all modes
    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument('--base_model_path', type=str, required=True, help='Path or Hugging Face name of the base LLM.')
    common_parser.add_argument('--target_layers', type=str, help="Layers to intervene on, e.g., '15' or '10-20,25'.")
    common_parser.add_argument('--instruct', action='store_true', default=True, help='Use instruction/chat format (default).')
    common_parser.add_argument('--no-instruct', dest='instruct', action='store_false', help='Do not use instruction/chat format.')

    # Arguments specific to activation extraction
    extraction_parser = argparse.ArgumentParser(add_help=False)
    extraction_parser.add_argument('--formal_lora_path', type=str, required=True, help="Path to the formal-style LoRA adapter.")
    extraction_parser.add_argument('--informal_lora_path', type=str, required=True, help="Path to the informal-style LoRA adapter.")
    extraction_parser.add_argument('--dataset_name', type=str, required=True, help="Path to the JSON dataset file with prompts and responses.")
    extraction_parser.add_argument('--batch_size', type=int, default=8, help="Batch size for processing the dataset.")

    # 'analyze' mode parser
    parser_analyze = subparsers.add_parser('analyze', help='Extract activations and compute style vectors.', parents=[common_parser, extraction_parser])
    parser_analyze.add_argument('--method', type=str, choices=['pca_denoise', 'mean'], default='pca_denoise', help="Method for computing the style vector. 'pca_denoise' is recommended.")
    parser_analyze.add_argument('--output_path', type=str, default='./style_vectors.pt', help="Path to save the computed style vectors.")
    parser_analyze.set_defaults(func=analyze_activations)

    # 'generate' (interactive) mode parser
    parser_generate = subparsers.add_parser('generate', help='Run interactive generation with style steering.', parents=[common_parser])
    parser_generate.add_argument('--activations_paths', type=str, nargs='+', required=True, help="Path(s) to one or two pre-computed style vector files (.pt).")
    parser_generate.set_defaults(func=generate_interactive)

    # 'generate_batch' mode parser
    parser_generate_batch = subparsers.add_parser('generate_batch', help='Run batch generation from a JSONL file.', parents=[common_parser])
    parser_generate_batch.add_argument('--activations_path', type=str, required=True, help="Path to the pre-computed style vector file (.pt).")
    parser_generate_batch.add_argument('--input_jsonl', type=str, required=True, help="Path to the input JSONL file (one prompt per line).")
    parser_generate_batch.add_argument('--output_jsonl', type=str, required=True, help="Path to the output JSONL file for saving results.")
    parser_generate_batch.add_argument('--alpha', type=float, required=True, help="The fixed style strength (alpha) to use for generation.")
    parser_generate_batch.add_argument('--generation_batch_size', type=int, default=8, help="Batch size for the generation process.")
    # Generation parameters
    parser_generate_batch.add_argument('--max_new_tokens', type=int, default=512, help="Maximum new tokens to generate.")
    parser_generate_batch.add_argument('--do_sample', action='store_true', default=False, help="Enable sampling.")
    parser_generate_batch.add_argument('--top_p', type=float, default=0.9, help="Top-p for nucleus sampling.")
    parser_generate_batch.add_argument('--temperature', type=float, default=0.7, help="Temperature for sampling.")
    parser_generate_batch.add_argument('--repetition_penalty', type=float, default=1.1, help="Repetition penalty.")
    parser_generate_batch.set_defaults(func=generate_batch)

    args = parser.parse_args()

    if args.mode == 'generate' and len(args.activations_paths) > 2:
        parser.error("Argument --activations_paths can accept at most two file paths.")

    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()