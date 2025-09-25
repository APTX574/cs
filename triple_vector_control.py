# -*- coding: utf-8 -*-
"""
Triple Vector Control Script - For batch generation using three distinct style vectors.

This script is an extension of a single/dual vector control architecture, modified to
support the weighted combination of three independent style vectors.

The output filename is automatically generated based on the weights (alphas) of the
three vectors.

Usage:
python triple_vector_control.py generate_batch \\
    --base_model_path /path/to/model \\
    --vector1_path /path/to/vector1.pt \\
    --vector2_path /path/to/vector2.pt \\
    --vector3_path /path/to/vector3.pt \\
    --alpha1 1.0 --alpha2 0.5 --alpha3 -0.3 \\
    --input_jsonl input.jsonl \\
    --target_layers "15-20"
"""

import argparse
import os
import json
from typing import List, Dict, Any, Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer, PreTrainedModel
from tqdm import tqdm
import numpy as np

# ======================================================================================
# Triple Vector Activation Steering Class
# ======================================================================================

class TripleActivationSteering:
    """
    Modifies (steers) activations at inference time using PyTorch hooks.
    This simplified version is designed for batch processing, where each instance
    is responsible for a single layer and applies a fixed, pre-computed steering vector.
    """
    def __init__(self, combined_vector: torch.Tensor, device: str):
        self.combined_vector = combined_vector.to(device)

    def hook_fn(self, module: torch.nn.Module, inp: Any, outp: torch.Tensor) -> torch.Tensor:
        """The hook function that applies the steering vector."""
        if outp.dim() >= 2:
            if outp.dim() == 3:  # [batch, seq_len, hidden_dim]
                # Apply steering to the last token's activation
                outp[:, -1, :] += self.combined_vector
            else:  # [seq_len, hidden_dim]
                outp[-1, :] += self.combined_vector
        return outp

# ======================================================================================
# Helper Functions
# ======================================================================================

def parse_layer_range(range_str: str) -> List[int]:
    """Parses a layer range string (e.g., '15' or '10-20,25') into a list of layer indices."""
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

def setup_environment(args: argparse.Namespace) -> Tuple[str, PreTrainedModel, PreTrainedTokenizer, List[str]]:
    """Sets up the environment, model, tokenizer, and target layers."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    print(f"Loading base model from '{args.base_model_path}'...")
    model = AutoModelForCausalLM.from_pretrained(args.base_model_path, torch_dtype=torch.float16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    target_layer_indices = parse_layer_range(args.target_layers)
    target_layer_names = [f"model.layers.{i}.mlp" for i in target_layer_indices]
    print(f"Target layers: {', '.join(target_layer_names)}")
    
    torch.set_float32_matmul_precision('high')
    return device, model, tokenizer, target_layer_names

def load_style_vector(vector_path: str, device: str, target_layer_names: List[str]) -> Dict[str, torch.Tensor]:
    """
    Loads and normalizes a style vector file.
    The file can be a single tensor or a dictionary mapping layer names to tensors.
    """
    print(f"Loading style vector: {os.path.basename(vector_path)}")
    vector_data = torch.load(vector_path, map_location=device, weights_only=False)
    
    style_vectors = {}

    if isinstance(vector_data, dict):
        print("  Detected dictionary format. Loading vectors for each target layer.")
        for layer_name in target_layer_names:
            if layer_name in vector_data:
                style_vector = vector_data[layer_name]
                if not isinstance(style_vector, torch.Tensor):
                    style_vector = torch.tensor(style_vector, device=device, dtype=torch.float16)
                else:
                    style_vector = style_vector.to(device).to(torch.float16)
                
                norm = torch.linalg.norm(style_vector)
                if norm > 0:
                    style_vectors[layer_name] = style_vector / norm
                else:
                    style_vectors[layer_name] = style_vector
                    print(f"  - Warning for layer {layer_name}: Vector norm is zero.")
            else:
                print(f"  - Warning: Vector for layer '{layer_name}' not found in file. Skipping.")

    elif isinstance(vector_data, torch.Tensor):
        print("  Detected single tensor format. Applying to all target layers.")
        style_vector = vector_data.to(device).to(torch.float16)
        norm = torch.linalg.norm(style_vector)
        if norm > 0:
            style_vector = style_vector / norm
        else:
            print(f"  Warning: Vector norm is zero.")
        
        for layer_name in target_layer_names:
            style_vectors[layer_name] = style_vector
    else:
        raise ValueError(f"Unsupported vector data format: {type(vector_data)}")
    
    return style_vectors

def generate_output_filename(input_jsonl: str, alpha1: float, alpha2: float, alpha3: float) -> str:
    """Generates an output filename based on the input filename and vector weights."""
    base_name = os.path.splitext(os.path.basename(input_jsonl))[0]
    
    # Create a filesystem-safe weight string
    weight_str = f"a1_{alpha1}_a2_{alpha2}_a3_{alpha3}"
    weight_str = weight_str.replace(".", "p")  # Replace periods with 'p'
    weight_str = weight_str.replace("-", "neg") # Replace minus sign with 'neg'
    
    return f"{base_name}_triple_{weight_str}.jsonl"

def process_outputs(batch_data, outputs, inputs, tokenizer, args):
    """Processes the model's generation outputs and formats them for saving."""
    batch_results = []
    for i, (idx, item) in enumerate(batch_data):
        if i < len(outputs):
            # For left-padded batches, the input length is constant
            input_ids_len = inputs.input_ids[i].shape[0]
            response_ids = outputs[i][input_ids_len:]
            response_text = tokenizer.decode(response_ids, skip_special_tokens=True)
            
            result = {
                'index': idx,
                'prompt': item['prompt'],
                'generated_text': response_text,
                'alpha1': args.alpha1,
                'alpha2': args.alpha2,
                'alpha3': args.alpha3,
            }
            batch_results.append(result)
    return batch_results

# ======================================================================================
# Triple Vector Batch Generation Function
# ======================================================================================

def generate_batch_triple(args: argparse.Namespace):
    """Main function to perform batch generation using three independent style vectors."""
    output_filename = generate_output_filename(args.input_jsonl, args.alpha1, args.alpha2, args.alpha3)
    args.output_jsonl = output_filename
    print(f"\nOutput will be saved to: {output_filename}")

    if os.path.exists(args.output_jsonl):
        print("Output file already exists. Skipping generation.")
        return

    print(f"\n--- Starting Triple-Vector Batch Generation ---")
    device, model, tokenizer, target_layer_names = setup_environment(args)
    
    print("\nLoading three style vectors...")
    style_vectors1 = load_style_vector(args.vector1_path, device, target_layer_names)
    style_vectors2 = load_style_vector(args.vector2_path, device, target_layer_names)
    style_vectors3 = load_style_vector(args.vector3_path, device, target_layer_names)
    
    print(f"\nWeight Configuration:")
    print(f"  Vector 1 Alpha: {args.alpha1}")
    print(f"  Vector 2 Alpha: {args.alpha2}")
    print(f"  Vector 3 Alpha: {args.alpha3}")

    print(f"\nLoading test data from '{args.input_jsonl}'...")
    test_data = []
    with open(args.input_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                test_data.append(json.loads(line.strip()))
    print(f"Loaded {len(test_data)} test items.")
    
    # Enumerate data for indexing
    remaining_data = list(enumerate(test_data))
    
    # Pre-compute combined vectors and register hooks for each layer
    all_modules = dict(model.named_modules())
    handles = []
    if args.alpha1 != 0 or args.alpha2 != 0 or args.alpha3 != 0:
        print("\nPre-computing combined vectors and registering activation hooks...")
        for layer_name in target_layer_names:
            combined_vector = torch.zeros(model.config.hidden_size, device=device, dtype=torch.float16)

            if args.alpha1 != 0 and layer_name in style_vectors1:
                combined_vector += args.alpha1 * style_vectors1[layer_name]
            if args.alpha2 != 0 and layer_name in style_vectors2:
                combined_vector += args.alpha2 * style_vectors2[layer_name]
            if args.alpha3 != 0 and layer_name in style_vectors3:
                combined_vector += args.alpha3 * style_vectors3[layer_name]
            
            if torch.linalg.norm(combined_vector) > 0:
                steering_hook = TripleActivationSteering(combined_vector, device)
                handles.append(all_modules[layer_name].register_forward_hook(steering_hook.hook_fn))
    
    print(f"Registered {len(handles)} hooks.")

    try:
        with open(args.output_jsonl, 'w', encoding='utf-8') as f_out:
            for i in tqdm(range(0, len(remaining_data), args.generation_batch_size), desc="Batch Generation Progress"):
                batch_data = remaining_data[i:i+args.generation_batch_size]
                
                indices, items = zip(*batch_data)
                batch_prompts = []
                for item in items:
                    prompt = item['prompt']
                    if args.instruct:
                        messages = [{"role": "user", "content": f"{prompt}"}]
                        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    else:
                        text = prompt
                    batch_prompts.append(text)
                
                inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=512, padding_side="left").to(model.device)

                generation_kwargs = {
                    "max_new_tokens": args.max_new_tokens, 
                    "do_sample": args.do_sample, 
                    "top_p": args.top_p, 
                    "temperature": args.temperature, 
                    "pad_token_id": tokenizer.eos_token_id, 
                    "repetition_penalty": args.repetition_penalty,
                }
                
                with torch.no_grad():
                    outputs = model.generate(**inputs, **generation_kwargs)
                
                batch_results = process_outputs(batch_data, outputs, inputs, tokenizer, args)
                for result in batch_results:
                    f_out.write(json.dumps(result, ensure_ascii=False) + '\n')
    finally:
        print("\nCleaning up all activation hooks...")
        for handle in handles:
            handle.remove()
        print("Cleanup complete.")

    print(f"✅ Batch generation finished! Results saved to {args.output_jsonl}")

# ======================================================================================
# Main Function & Argument Parsing
# ======================================================================================

def main():
    """Main function to parse arguments and execute the triple-vector batch generation."""
    parser = argparse.ArgumentParser(description="Controllable generation for LLMs using three style vectors.")
    subparsers = parser.add_subparsers(dest='mode', required=True, help="Execution mode")

    # Parser for the batch generation mode
    parser_generate_batch = subparsers.add_parser('generate_batch', help='Perform batch generation using three style vectors.')
    
    # Model and layer arguments
    parser_generate_batch.add_argument('--base_model_path', type=str, required=True, help='Path or Hugging Face name of the base LLM.')
    parser_generate_batch.add_argument('--target_layers', type=str, required=True, help="Layers to intervene on, e.g., '15' or '10-20'.")
    
    # Vector path arguments
    parser_generate_batch.add_argument('--vector1_path', type=str, required=True, help="Path to the first style vector file (.pt).")
    parser_generate_batch.add_argument('--vector2_path', type=str, required=True, help="Path to the second style vector file (.pt).")
    parser_generate_batch.add_argument('--vector3_path', type=str, required=True, help="Path to the third style vector file (.pt).")
    
    # Weight (alpha) arguments
    parser_generate_batch.add_argument('--alpha1', type=float, required=True, help="Weight for the first vector.")
    parser_generate_batch.add_argument('--alpha2', type=float, required=True, help="Weight for the second vector.")
    parser_generate_batch.add_argument('--alpha3', type=float, required=True, help="Weight for the third vector.")
    
    # Input file argument
    parser_generate_batch.add_argument('--input_jsonl', type=str, required=True, help="Path to the input JSONL file (one prompt per line).")
    
    # Generation parameters
    parser_generate_batch.add_argument('--instruct', action='store_true', default=True, help='Use instruction/chat format (default is True).')
    parser_generate_batch.add_argument('--no-instruct', dest='instruct', action='store_false', help='Do not use instruction/chat format.')
    parser_generate_batch.add_argument('--max_new_tokens', type=int, default=512, help="Maximum number of new tokens to generate.")
    parser_generate_batch.add_argument('--do_sample', action='store_true', default=False, help="Enable nucleus sampling.")
    parser_generate_batch.add_argument('--top_p', type=float, default=0.9, help="Top-p for nucleus sampling.")
    parser_generate_batch.add_argument('--temperature', type=float, default=0.7, help="Sampling temperature.")
    parser_generate_batch.add_argument('--repetition_penalty', type=float, default=1.1, help="Repetition penalty.")
    parser_generate_batch.add_argument('--generation_batch_size', type=int, default=8, help="Batch size for the generation process.")
    
    parser_generate_batch.set_defaults(func=generate_batch_triple)
    
    args = parser.parse_args()

    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()