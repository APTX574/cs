#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A script for fine-tuning a Causal Language Model using contrastive learning with dual LoRA adapters.

This script sets up a training pipeline where a base model is equipped with two separate LoRA
adapters, one for a "formal" style and one for an "informal" style. The training objective
combines standard language modeling (generation) loss with a contrastive loss. The contrastive
loss encourages the model to produce distinct representations for the two styles, pushing formal
and informal outputs further apart in the embedding space.

This approach is useful for steerable generation, where one can control the output style by
selectively activating or merging the trained LoRA adapters at inference time.
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
    HfArgumentParser,
)
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    PeftModel,
)

# For reproducibility and debugging, you can uncomment this line
# os.environ["WANDB_MODE"] = "offline"

# ===================================================================================
# 1. Configuration using HfArgumentParser
# Purpose: Manages all hyperparameters for the model, data, and training pipeline
#          in a structured and command-line friendly way.
# ===================================================================================

@dataclass
class ModelArguments:
    """Arguments pertaining to which model/config/tokenizer we are going to fine-tune from."""
    base_model_name: str = field(
        metadata={"help": "Path to the pretrained base model or its name on Hugging Face."}
    )

@dataclass
class DataArguments:
    """Arguments pertaining to what data we are going to use."""
    # This would typically be a path to a JSONL file.
    # For this example, we use a placeholder.
    dataset_name: str = field(
        default="<your_dataset_name_or_path>",
        metadata={"help": "The name or path of the dataset to use."}
    )
    max_length: int = field(
        default=128,
        metadata={"help": "Maximum sequence length for tokenization."}
    )

@dataclass
class ContrastiveTrainingArguments(TrainingArguments):
    """Custom training arguments for contrastive learning."""
    output_dir: str = field(
        default="./contrastive_lora_results",
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."}
    )
    num_train_epochs: int = field(default=3, metadata={"help": "Total number of training epochs to perform."})
    per_device_train_batch_size: int = field(default=4, metadata={"help": "Batch size per GPU/CPU for training."})
    learning_rate: float = field(default=1e-4, metadata={"help": "The initial learning rate for AdamW."})
    gradient_accumulation_steps: int = field(default=2, metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."})

    # LoRA specific arguments
    lora_r: int = field(default=8, metadata={"help": "The rank of the LoRA matrices."})
    lora_alpha: int = field(default=16, metadata={"help": "The alpha parameter for LoRA scaling."})
    lora_dropout: float = field(default=0.05, metadata={"help": "The dropout probability for LoRA layers."})
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        metadata={"help": "The names of the modules to apply LoRA to."}
    )

    # Contrastive learning specific arguments
    contrastive_loss_lambda: float = field(
        default=0.1,
        metadata={"help": "The weight of the contrastive loss component."}
    )
    temperature: float = field(
        default=0.07,
        metadata={"help": "The temperature parameter for the InfoNCE loss."}
    )
    use_contrastive_loss: bool = field(
        default=True,
        metadata={"help": "Whether to include the contrastive loss in the total loss calculation."}
    )


# ===================================================================================
# 2. Data Preparation
# Purpose: Defines the dataset structure and collation logic to provide correctly
#          formatted data to the model.
# ===================================================================================

class ContrastiveDataset(Dataset):
    """
    A dummy dataset for contrastive learning.
    In a real-world scenario, this would load data from a file (e.g., JSONL).
    Each item should contain a prompt, a formal response, and an informal response.
    """
    def __init__(self, num_samples=1000):
        self.num_samples = num_samples
        self.data = []
        for i in range(num_samples):
            self.data.append({
                "prompt": f"This is prompt number {i}.",
                "formal_response": f"This is a formal response to prompt {i}.",
                "informal_response": f"here's an informal answer for ya, prompt {i}!",
            })

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx]

@dataclass
class ContrastiveDataCollator:
    """
    Data collator that tokenizes prompts and responses and prepares batches
    for the custom contrastive model.
    """
    tokenizer: transformers.PreTrainedTokenizer
    max_length: int

    def __call__(self, instances: List[Dict]) -> Dict[str, torch.Tensor]:
        prompts = [ins["prompt"] for ins in instances]
        formal_responses = [ins["formal_response"] for ins in instances]
        informal_responses = [ins["informal_response"] for ins in instances]

        # Apply chat template if available (recommended for instruction-tuned models)
        if hasattr(self.tokenizer, "apply_chat_template"):
            formal_texts = [self.tokenizer.apply_chat_template([{"role": "user", "content": p}, {"role": "assistant", "content": r}], tokenize=False) for p, r in zip(prompts, formal_responses)]
            informal_texts = [self.tokenizer.apply_chat_template([{"role": "user", "content": p}, {"role": "assistant", "content": r}], tokenize=False) for p, r in zip(prompts, informal_responses)]
        else:
            formal_texts = [p + " " + r for p, r in zip(prompts, formal_responses)]
            informal_texts = [p + " " + r for p, r in zip(prompts, informal_responses)]

        # Tokenize both sets of texts
        formal_inputs = self.tokenizer(formal_texts, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")
        informal_inputs = self.tokenizer(informal_texts, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")

        # Prepare labels for language modeling loss (masking padding tokens)
        formal_labels = formal_inputs.input_ids.clone()
        formal_labels[formal_labels == self.tokenizer.pad_token_id] = -100
        informal_labels = informal_inputs.input_ids.clone()
        informal_labels[informal_labels == self.tokenizer.pad_token_id] = -100

        return {
            "formal_input_ids": formal_inputs.input_ids,
            "formal_attention_mask": formal_inputs.attention_mask,
            "formal_labels": formal_labels,
            "informal_input_ids": informal_inputs.input_ids,
            "informal_attention_mask": informal_inputs.attention_mask,
            "informal_labels": informal_labels,
        }

# ===================================================================================
# 3. Custom Model Definition
# Purpose: Wraps the PeftModel to handle the dual forward pass required for
#          contrastive learning.
# ===================================================================================

class ContrastiveLoRAModel(torch.nn.Module):
    """
    A custom model that wraps a PeftModel.
    Its forward method is overridden to support contrastive learning by performing
    two forward passes: one with the 'formal' adapter and one with the 'informal' adapter.
    """
    def __init__(self, peft_model: PeftModel):
        super().__init__()
        self.peft_model = peft_model
        # Ensure the model outputs hidden states, which are needed for representations
        self.peft_model.config.output_hidden_states = True

    def _get_last_hidden_state(self, outputs, attention_mask):
        """
        Extracts the hidden state of the last non-padding token as a sentence representation.
        This is a common strategy for getting a summary vector for a sequence.
        """
        # The last hidden state has shape [batch_size, seq_length, hidden_size]
        last_hidden_state = outputs.hidden_states[-1]

        # Find the index of the last non-padding token for each item in the batch
        sequence_lengths = attention_mask.sum(dim=1) - 1

        batch_size = last_hidden_state.shape[0]
        # Gather the hidden states at the calculated indices
        return last_hidden_state[torch.arange(batch_size, device=last_hidden_state.device), sequence_lengths]

    def forward(self, **kwargs):
        """
        Performs two forward passes, one for each adapter, and returns their
        logits and sentence representations.
        """
        # --- Formal Pass ---
        self.peft_model.set_adapter("formal")
        formal_outputs = self.peft_model(
            input_ids=kwargs["formal_input_ids"], attention_mask=kwargs["formal_attention_mask"]
        )
        formal_logits = formal_outputs.logits
        formal_reps = self._get_last_hidden_state(formal_outputs, kwargs["formal_attention_mask"])

        # --- Informal Pass ---
        self.peft_model.set_adapter("informal")
        informal_outputs = self.peft_model(
            input_ids=kwargs["informal_input_ids"], attention_mask=kwargs["informal_attention_mask"]
        )
        informal_logits = informal_outputs.logits
        informal_reps = self._get_last_hidden_state(informal_outputs, kwargs["informal_attention_mask"])

        return {
            "formal_logits": formal_logits,
            "informal_logits": informal_logits,
            "formal_reps": formal_reps,
            "informal_reps": informal_reps,
        }

    def __getattr__(self, name: str):
        """
        Forward attribute access to the wrapped PeftModel.
        This is necessary for the Trainer to access methods like `save_pretrained`.
        """
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.peft_model, name)

# ===================================================================================
# 4. Custom Trainer
# Purpose: Overrides the default loss calculation to combine the generation loss
#          and the contrastive loss.
# ===================================================================================

class ContrastiveTrainer(Trainer):
    def __init__(self, *args, lambda_param: float, temperature: float, use_contrastive_loss: bool, **kwargs):
        super().__init__(*args, **kwargs)
        self.lambda_param = lambda_param
        self.temperature = temperature
        self.use_contrastive_loss = use_contrastive_loss
        self._last_losses = {} # A dictionary to store loss components for logging

    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        formal_logits, informal_logits = outputs["formal_logits"], outputs["informal_logits"]
        formal_reps, informal_reps = outputs["formal_reps"], outputs["informal_reps"]
        formal_labels, informal_labels = inputs["formal_labels"], inputs["informal_labels"]

        # --- 1. Calculate Generation Loss ---
        # Standard cross-entropy loss for language modeling.
        # We calculate it for both formal and informal passes and average them.
        loss_fct = torch.nn.CrossEntropyLoss()
        formal_loss = loss_fct(formal_logits.view(-1, formal_logits.size(-1)), formal_labels.view(-1))
        informal_loss = loss_fct(informal_logits.view(-1, informal_logits.size(-1)), informal_labels.view(-1))
        generation_loss = (formal_loss + informal_loss) / 2

        # --- 2. Calculate Contrastive Loss (InfoNCE) ---
        # This loss pushes representations of corresponding formal/informal pairs together
        # while pushing non-corresponding pairs apart.
        batch_size = formal_reps.size(0)
        # Normalize representations to compute cosine similarity
        formal_reps = F.normalize(formal_reps, p=2, dim=1)
        informal_reps = F.normalize(informal_reps, p=2, dim=1)

        # Calculate similarity matrix between all formal and informal reps in the batch
        sim_matrix = torch.matmul(formal_reps, informal_reps.T) / self.temperature

        # The labels are the diagonal elements, as sim_matrix[i, i] is the similarity
        # between the i-th formal and i-th informal representation.
        labels = torch.arange(batch_size, device=sim_matrix.device)

        # Symmetric InfoNCE loss
        loss_formal_side = F.cross_entropy(sim_matrix, labels)
        loss_informal_side = F.cross_entropy(sim_matrix.T, labels)
        contrastive_loss = (loss_formal_side + loss_informal_side) / 2

        # --- 3. Calculate Total Loss ---
        if self.use_contrastive_loss:
            total_loss = generation_loss + self.lambda_param * contrastive_loss
        else:
            total_loss = generation_loss

        # --- Store individual losses for logging ---
        self._last_losses = {
            "gen_loss": generation_loss.item(),
            "con_loss": contrastive_loss.item(),
            "total_loss": total_loss.item(),
        }

        return (total_loss, outputs) if return_outputs else total_loss

    def log(self, logs: Dict[str, float]) -> None:
        """Overrides the log method to add our custom loss components."""
        if self._last_losses:
            logs.update(self._last_losses)
        super().log(logs)
