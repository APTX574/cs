# Causal-Steer: Disentangled Continuous Style Control without Parallel Corpora

A toolkit for controllable text generation in Large Language Models using activation engineering and LoRA-based contrastive learning.

## 🌟 Key Features

- **Multiple Control Modes**: Support for single-vector, dual-vector, and triple-vector style control
- **Interactive Generation**: Real-time interactive text generation interface
- **Batch Processing**: Efficient processing of large-scale datasets
- **Style Vector Extraction**: Extract robust style vectors from contrastive datasets
- **Comprehensive Evaluation**: Support for GPT-4 evaluation and statistical analysis
- **Flexible Model Support**: Compatible with Hugging Face models and LoRA adapters
- **Contrastive Learning**: Train dual LoRA adapters with contrastive learning

## 📁 Project Structure

```
linear_gen/
├── vector.py                  # Main utility script for activation steering
├── triple_vector_control.py   # Triple-vector control module
├── run_colora.py              # Contrastive LoRA training script
├── evaluate_with_gpt_v2.py    # GPT-4 evaluation script
├── analyze_scores_v2.py       # Statistical analysis script
├── datasets/                  # Dataset directory
│   ├── formal.jsonl          # Formal language data
│   ├── detex.jsonl           # Detoxification data
│   ├── knowledge.jsonl       # Knowledge-oriented data
│   └── toxicity_*.jsonl      # Toxicity-related data
├── pt/                       # Pre-extracted activation vectors
│   ├── activations_formal_new_pca_denoise1_o.pt    # Formal style (Qwen2.5-7B-Instruct)
│   ├── activations_detex_new_pca_denoise1_o.pt     # Detox style (Qwen2.5-7B-Instruct)
│   └── activations_know_new_pca_denoise1_o.pt      # Knowledge style (Qwen2.5-7B-Instruct)
├── script/                   # Convenience scripts
│   ├── run_analyze.sh        # Activation vector extraction
│   ├── run_generate.sh       # Interactive generation
│   ├── run_generate_batch.sh # Batch generation
│   └── pipline.sh           # Complete pipeline
└── requirements.txt          # Dependencies
```

## 🚀 Quick Start

### Environment Setup

```bash
# Clone the repository
git clone <repository-url>
cd linear_gen

# Install dependencies
pip install -r requirements.txt

# Or use conda environment
conda env create -f environment.yml
conda activate linear_gen
```

## 📋 Workflow Options

You can use this toolkit in two ways:

### Option 1: Train Your Own LoRA Adapters

#### 1. Train Contrastive LoRA Adapters

```python
python run_colora.py train \
    --base_model_path /path/to/base/model \
    --dataset_path datasets/formal.jsonl \
    --output_dir ./lora_output \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --learning_rate 1e-4
```

#### 2. Extract Activation Vectors

```bash
bash script/run_analyze.sh \
    --model-path /path/to/base/model \
    --formal-lora ./lora_output/formal \
    --informal-lora ./lora_output/informal \
    --dataset datasets/formal.jsonl \
    --output pt/custom_style.pt \
    --target-layers "15-25"
```

### Option 2: Use Pre-extracted Vectors (Qwen2.5-7B-Instruct)

You can directly use the pre-extracted activation vectors in the `pt/` directory, which were extracted from Qwen2.5-7B-Instruct:

- `activations_formal_new_pca_denoise1_o.pt`: Formal style control
- `activations_detex_new_pca_denoise1_o.pt`: Detoxification control  
- `activations_know_new_pca_denoise1_o.pt`: Knowledge-oriented control

#### Interactive Generation

```bash
bash script/run_generate.sh \
    /path/to/qwen2.5-7b-instruct \
    pt/activations_formal_new_pca_denoise1_o.pt \
    "18-23"
```

#### Batch Generation

```bash
bash script/run_generate_batch.sh \
    --model-path /path/to/qwen2.5-7b-instruct \
    --vector-path pt/activations_formal_new_pca_denoise1_o.pt \
    --input-file input_prompts.jsonl \
    --output-file generated_results.jsonl \
    --layers "18-23" \
    --alpha -2.5
```

## 📖 Detailed Feature Description

### Core Modules

#### vector.py - Main Utility Tool

Supports three execution modes:

1. **analyze**: Extract style vectors from contrastive data
2. **generate**: Interactive style-controlled generation
3. **generate_batch**: Batch style-controlled generation

```python
# Analysis mode - Extract style vectors
python vector.py analyze \
    --base_model_path ./models/llama-3-8b \
    --formal_lora_path ./lora/formal \
    --informal_lora_path ./lora/informal \
    --dataset_path ./datasets/formal.jsonl \
    --target_layers "15-25" \
    --method pca_denoise \
    --batch_size 16

# Generation mode - Interactive use
python vector.py generate \
    --base_model_path ./models/llama-3-8b \
    --activations_paths ./pt/formal_style.pt \
    --target_layers "18-23"

# Batch generation mode
python vector.py generate_batch \
    --base_model_path ./models/llama-3-8b \
    --activations_path ./pt/formal_style.pt \
    --input_jsonl ./input.jsonl \
    --output_jsonl ./output.jsonl \
    --target_layers "18-23" \
    --alpha -2.5
```

#### run_colora.py - Contrastive LoRA Training

Trains dual LoRA adapters with contrastive learning for style control:

```python
# Training mode
python run_colora.py train \
    --base_model_path /path/to/model \
    --dataset_path datasets/formal.jsonl \
    --output_dir ./lora_output \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --learning_rate 1e-4 \
    --lora_rank 16 \
    --lora_alpha 32

# Inference mode
python run_colora.py inference \
    --base_model_path /path/to/model \
    --formal_adapter_path ./lora_output/formal \
    --informal_adapter_path ./lora_output/informal
```

#### triple_vector_control.py - Multi-Vector Control

Supports simultaneous use of three different style vectors for fine-grained control:

```python
python triple_vector_control.py generate_batch \
    --base_model_path /path/to/model \
    --vector1_path pt/activations_formal_new_pca_denoise1_o.pt \
    --vector2_path pt/activations_know_new_pca_denoise1_o.pt \
    --vector3_path pt/activations_detex_new_pca_denoise1_o.pt \
    --alpha1 1.0 --alpha2 0.5 --alpha3 -0.3 \
    --input_jsonl input.jsonl \
    --target_layers "15-20"
```

### Evaluation Tools

#### GPT-4 Evaluation

```python
python evaluate_with_gpt_v2.py \
    --input generated_results.jsonl \
    --output evaluated_results.jsonl \
    --criteria formality \
    --model gpt-4-turbo \
    --workers 4
```

#### Statistical Analysis

```python
python analyze_scores_v2.py \
    --input evaluated_results.jsonl \
    --output analysis_report.json
```


**Note**: Before use, ensure you have properly configured OpenAI API keys (if using GPT evaluation features):

```bash
export OPENAI_API_KEY="your_api_key_here"
export OPENAI_BASE_URL="your_base_url_here"  # Optional
```