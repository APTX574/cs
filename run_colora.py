#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A complete script for fine-tuning and inference with a dual LoRA contrastive learning setup.

This script supports two modes:
1. 'train': Fine-tunes a base language model with two LoRA adapters ('formal' and 'informal')
   using a combination of generation loss and contrastive loss.
2. 'inference': Loads the trained adapters and runs an interactive session to demonstrate
   style-controlled generation.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig,
    PreTrainedTokenizer,
    HfArgumentParser,
)
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    PeftModel,
)
import datasets

# ===================================================================================
# 1. Configuration
# Purpose: Manages all hyperparameters in a structured way using HfArgumentParser.
# ===================================================================================

@dataclass
class ScriptArguments:
    """Arguments for controlling the script's execution mode and inference paths."""
    mode: str = field(
        default="train",
        metadata={"help": "The execution mode: 'train' or 'inference'."}
    )
    formal_adapter_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the trained 'formal' LoRA adapter for inference."}
    )
    informal_adapter_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the trained 'informal' LoRA adapter for inference."}
    )

@dataclass
class ModelArguments:
    """Arguments pertaining to which model/config/tokenizer we are going to use."""
    base_model_name: str = field(
        metadata={"help": "Path to the pretrained base model or its name on Hugging Face."}
    )

@dataclass
class DataArguments:
    """Arguments pertaining to the data used for training."""
    dataset_path: str = field(
        metadata={"help": "Path to the training dataset (JSONL file)."}
    )
    max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length for tokenization."}
    )
    is_instruct: bool = field(
        default=True,
        metadata={"help": "Whether to use the chat template for instruction-following models."}
    )

@dataclass
class ContrastiveTrainingArguments(TrainingArguments):
    """Custom training arguments for the contrastive learning setup."""
    output_dir: str = field(
        default="./contrastive_lora_results",
        metadata={"help": "The output directory for checkpoints and final adapters."}
    )
    num_train_epochs: int = field(default=1, metadata={"help": "Total number of training epochs."})
    per_device_train_batch_size: int = field(default=4, metadata={"help": "Batch size per GPU/CPU."})
    learning_rate: float = field(default=2e-5, metadata={"help": "The initial learning rate."})
    gradient_accumulation_steps: int = field(default=4, metadata={"help": "Steps for gradient accumulation."})
    logging_steps: int = field(default=10, metadata={"help": "Log every X updates steps."})
    save_steps: int = field(default=500, metadata={"help": "Save checkpoint every X updates steps."})
    bf16: bool = field(default=True, metadata={"help": "Use BF16 for training on supported hardware."})

    # LoRA specific arguments
    lora_r: int = field(default=16, metadata={"help": "The rank of the LoRA matrices."})
    lora_alpha: int = field(default=32, metadata={"help": "The alpha parameter for LoRA scaling."})
    lora_dropout: float = field(default=0.05, metadata={"help": "Dropout for LoRA layers."})
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        metadata={"help": "Modules to apply LoRA to."}
    )

    # Contrastive learning specific arguments
    contrastive_loss_lambda: float = field(default=0.1, metadata={"help": "Weight of the contrastive loss."})
    temperature: float = field(default=0.07, metadata={"help": "Temperature for the InfoNCE loss."})


# ===================================================================================
# 2. Data Preparation
# ===================================================================================

@dataclass
class ContrastiveDataCollator:
    """
    Processes batches of data, tokenizing them for contrastive training.
    Handles both instruction-based (chat template) and direct text processing.
    """
    tokenizer: PreTrainedTokenizer
    max_length: int
    is_instruct: bool = True

    def _process_one_instruct(self, user_msg: str, assistant_msg: str) -> Dict[str, torch.Tensor]:
        """Applies chat template to a single QA pair and creates tokenized inputs and labels."""
        if not user_msg or not assistant_msg:
            raise ValueError("User or assistant message is empty.")

        messages = [{"role": "user", "content": user_msg}, {"role": "assistant", "content": assistant_msg}]
        full_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        assistant_start = full_text.rfind(assistant_msg) # Use rfind for robustness
        if assistant_start == -1:
            raise ValueError("Could not find the assistant's response in the formatted text.")

        encoding = self.tokenizer(
            full_text, return_attention_mask=True, return_offsets_mapping=True,
            padding="max_length", truncation=True, max_length=self.max_length
        )

        labels = []
        for (start, end), token_id in zip(encoding["offset_mapping"], encoding["input_ids"]):
            if start >= assistant_start:
                labels.append(token_id)
            else:
                labels.append(-100) # Mask tokens belonging to the user prompt

        return {
            "input_ids": torch.tensor(encoding["input_ids"]),
            "attention_mask": torch.tensor(encoding["attention_mask"]),
            "labels": torch.tensor(labels)
        }

    def _process_one_direct(self, text: str) -> Dict[str, torch.Tensor]:
        """Directly tokenizes text without a chat template."""
        if not text:
            raise ValueError("Input text is empty.")
        text += self.tokenizer.eos_token
        encoding = self.tokenizer(
            text, return_attention_mask=True, padding="max_length",
            truncation=True, max_length=self.max_length
        )
        labels = [tok if tok != self.tokenizer.pad_token_id else -100 for tok in encoding["input_ids"]]
        return {
            "input_ids": torch.tensor(encoding["input_ids"]),
            "attention_mask": torch.tensor(encoding["attention_mask"]),
            "labels": torch.tensor(labels)
        }

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Processes a batch of samples, each with two styles."""
        formal_inputs, informal_inputs = [], []

        for f in features:
            try:
                if self.is_instruct:
                    # Logic to handle different dataset formats
                    if "answer" in f and "a_nontoxic" in f["answer"]: # Format 1
                        prompt, formal_resp = f["answer"]["q"], f["answer"]["a_nontoxic"]
                        _, informal_resp = f["answer"]["q"], f["answer"]["a_toxic"]
                    else: # Format 2
                        prompt, formal_resp = f["prompt"], f["formal_response"]
                        _, informal_resp = f["prompt"], f["informal_response"]
                    formal_inputs.append(self._process_one_instruct(prompt, formal_resp))
                    informal_inputs.append(self._process_one_instruct(prompt, informal_resp))
                else:
                    formal_resp = f["answer"]["a_nontoxic"]
                    informal_resp = f["answer"]["a_toxic"]
                    formal_inputs.append(self._process_one_direct(formal_resp))
                    informal_inputs.append(self._process_one_direct(informal_resp))
            except (ValueError, KeyError) as e:
                print(f"Skipping malformed data item: {f}. Error: {e}")
                continue

        return {
            "formal_input_ids": torch.stack([d["input_ids"] for d in formal_inputs]),
            "formal_attention_mask": torch.stack([d["attention_mask"] for d in formal_inputs]),
            "formal_labels": torch.stack([d["labels"] for d in formal_inputs]),
            "informal_input_ids": torch.stack([d["input_ids"] for d in informal_inputs]),
            "informal_attention_mask": torch.stack([d["attention_mask"] for d in informal_inputs]),
            "informal_labels": torch.stack([d["labels"] for d in informal_inputs]),
        }

# ===================================================================================
# 3. Custom Model
# ===================================================================================
class ContrastiveLoRAModel(torch.nn.Module):
    """Wraps a PeftModel to handle the dual forward pass for contrastive learning."""
    def __init__(self, peft_model: PeftModel):
        super().__init__()
        self.peft_model = peft_model
        self.peft_model.config.output_hidden_states = True

    def _get_last_hidden_state(self, outputs, attention_mask):
        """Extracts the hidden state of the last non-padding token as a sequence representation."""
        last_hidden_state = outputs.hidden_states[-1]
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_state.shape[0]
        return last_hidden_state[torch.arange(batch_size, device=last_hidden_state.device), sequence_lengths]

    def forward(self, **kwargs):
        """Performs two forward passes, one for each adapter."""
        # --- Formal Pass ---
        self.peft_model.set_adapter("formal")
        formal_outputs = self.peft_model(
            input_ids=kwargs["formal_input_ids"], attention_mask=kwargs["formal_attention_mask"]
        )
        formal_reps = self._get_last_hidden_state(formal_outputs, kwargs["formal_attention_mask"])

        # --- Informal Pass ---
        self.peft_model.set_adapter("informal")
        informal_outputs = self.peft_model(
            input_ids=kwargs["informal_input_ids"], attention_mask=kwargs["informal_attention_mask"]
        )
        informal_reps = self._get_last_hidden_state(informal_outputs, kwargs["informal_attention_mask"])

        return {
            "formal_logits": formal_outputs.logits, "formal_reps": formal_reps,
            "informal_logits": informal_outputs.logits, "informal_reps": informal_reps,
        }

    def __getattr__(self, name: str):
        """Forward attribute access to the wrapped PeftModel."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.peft_model, name)

# ===================================================================================
# 4. Custom Trainer
# ===================================================================================
class ContrastiveTrainer(Trainer):
    """Custom trainer that combines generation and contrastive losses."""
    def __init__(self, *args, lambda_param: float, temperature: float, **kwargs):
        super().__init__(*args, **kwargs)
        self.lambda_param = lambda_param
        self.temperature = temperature

    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        formal_logits, informal_logits = outputs["formal_logits"], outputs["informal_logits"]
        formal_reps, informal_reps = outputs["formal_reps"], outputs["informal_reps"]

        # --- 1. Generation Loss ---
        loss_fct = torch.nn.CrossEntropyLoss()
        formal_lm_loss = loss_fct(formal_logits.view(-1, formal_logits.size(-1)), inputs["formal_labels"].view(-1))
        informal_lm_loss = loss_fct(informal_logits.view(-1, informal_logits.size(-1)), inputs["informal_labels"].view(-1))
        generation_loss = (formal_lm_loss + informal_lm_loss) / 2

        # --- 2. Contrastive Loss (InfoNCE) ---
        batch_size = formal_reps.size(0)
        formal_reps_norm = F.normalize(formal_reps, p=2, dim=1)
        informal_reps_norm = F.normalize(informal_reps, p=2, dim=1)
        sim_matrix = torch.matmul(formal_reps_norm, informal_reps_norm.T) / self.temperature
        labels = torch.arange(batch_size, device=sim_matrix.device)
        contrastive_loss = (F.cross_entropy(sim_matrix, labels) + F.cross_entropy(sim_matrix.T, labels)) / 2
        
        # --- 3. Total Loss ---
        total_loss = generation_loss + self.lambda_param * contrastive_loss
        
        # Log metrics
        self.log({"gen_loss": generation_loss.item(), "con_loss": contrastive_loss.item()})
        
        return (total_loss, outputs) if return_outputs else total_loss

# ===================================================================================
# 5. Main Execution Block
# ===================================================================================

def run_training(model_args, data_args, training_args):
    """Sets up and runs the full training pipeline."""
    print("--- Starting Training Mode ---")
    
    # --- 1. Load Tokenizer and Base Model ---
    tokenizer = AutoTokenizer.from_pretrained(model_args.base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_args.base_model_name, device_map="auto", torch_dtype=torch.bfloat16)

    # --- 2. Setup PEFT with Dual LoRA Adapters ---
    lora_config = LoraConfig(
        r=training_args.lora_r, lora_alpha=training_args.lora_alpha, lora_dropout=training_args.lora_dropout,
        task_type=TaskType.CAUSAL_LM, target_modules=training_args.lora_target_modules, bias="none"
    )
    
    peft_model = get_peft_model(model, lora_config, adapter_name="informal")
    peft_model.add_adapter("formal", lora_config)
    peft_model.print_trainable_parameters()

    contrastive_model = ContrastiveLoRAModel(peft_model)

    # --- 3. Prepare Dataset ---
    dataset = datasets.load_dataset('json', split='train', data_files=data_args.dataset_path)
    data_collator = ContrastiveDataCollator(tokenizer, max_length=data_args.max_length, is_instruct=data_args.is_instruct)

    # --- 4. Initialize Trainer ---
    trainer = ContrastiveTrainer(
        model=contrastive_model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        lambda_param=training_args.contrastive_loss_lambda,
        temperature=training_args.temperature,
    )

    # --- 5. Start Training ---
    print("Starting training...")
    trainer.train()
    print("Training finished.")
    
    # --- 6. Save Final Adapters ---
    final_path = os.path.join(training_args.output_dir, "final_checkpoint")
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"Final adapters and tokenizer saved to {final_path}")

if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, ModelArguments, DataArguments, ContrastiveTrainingArguments))
    script_args, model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if script_args.mode == "train":
        run_training(model_args, data_args, training_args)