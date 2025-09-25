# 线性生成控制 (Linear Generation Control)

一个基于激活工程的大语言模型风格控制工具包，支持通过激活向量引导来实现可控文本生成。

## 🌟 主要特性

- **多种控制模式**：支持单向量、双向量和三向量风格控制
- **交互式生成**：实时交互式文本生成界面
- **批量处理**：高效处理大规模数据集
- **风格向量提取**：从对比数据集中提取稳健的风格向量
- **多种评估方式**：支持 GPT-4 评估和统计分析
- **灵活的模型支持**：兼容 Hugging Face 模型和 LoRA 适配器

## 📁 项目结构

```
linear_gen/
├── vector.py                  # 主要工具脚本
├── triple_vector_control.py   # 三向量控制模块
├── evaluate_with_gpt_v2.py    # GPT-4 评估脚本
├── analyze_scores_v2.py       # 结果统计分析
├── run_colora.py              # ColORA 运行脚本
├── datasets/                  # 数据集目录
│   ├── formal.jsonl          # 正式语言数据
│   ├── detex.jsonl           # 去毒数据
│   ├── knowledge.jsonl       # 知识型数据
│   └── toxicity_*.jsonl      # 毒性相关数据
├── pt/                       # 预训练激活向量
│   ├── activations_formal_new_pca_denoise1_o.pt
│   ├── activations_detex_new_pca_denoise1_o.pt
│   └── activations_know_new_pca_denoise1_o.pt
├── script/                   # 便捷脚本
│   ├── run_analyze.sh        # 分析脚本
│   ├── run_generate.sh       # 单次生成脚本
│   ├── run_generate_batch.sh # 批量生成脚本
│   └── pipline.sh           # 完整流水线
└── requirements.txt          # 依赖配置
```

## 🚀 快速开始

### 环境配置

```bash
# 克隆项目
git clone <repository-url>
cd linear_gen

# 安装依赖
pip install -r requirements.txt

# 或使用 conda 环境
conda env create -f environment.yml
conda activate linear_gen
```

### 基本使用流程

#### 1. 提取风格向量

```bash
bash script/run_analyze.sh \
    --model-path /path/to/base/model \
    --formal-lora /path/to/formal/lora \
    --informal-lora /path/to/informal/lora \
    --dataset datasets/formal.jsonl \
    --output pt/formal_style.pt \
    --target-layers "15-25"
```

#### 2. 交互式生成

```bash
bash script/run_generate.sh \
    /path/to/base/model \
    pt/formal_style.pt \
    "18-23"
```

#### 3. 批量生成

```bash
bash script/run_generate_batch.sh \
    --model-path /path/to/base/model \
    --vector-path pt/formal_style.pt \
    --input-file input_prompts.jsonl \
    --output-file generated_results.jsonl \
    --layers "18-23" \
    --alpha -2.5
```

## 📖 详细功能说明

### 核心模块

#### vector.py - 主要工具

支持三种运行模式：

1. **analyze**: 从对比数据中提取风格向量
2. **generate**: 交互式风格控制生成
3. **generate_batch**: 批量风格控制生成

```python
# 分析模式 - 提取风格向量
python vector.py analyze \
    --base_model_path ./models/llama-3-8b \
    --formal_lora_path ./lora/formal \
    --informal_lora_path ./lora/informal \
    --dataset_path ./datasets/formal.jsonl \
    --target_layers "15-25" \
    --method pca_denoise \
    --batch_size 16

# 生成模式 - 交互式使用
python vector.py generate \
    --base_model_path ./models/llama-3-8b \
    --activations_paths ./pt/formal_style.pt \
    --target_layers "18-23"

# 批量生成模式
python vector.py generate_batch \
    --base_model_path ./models/llama-3-8b \
    --activations_path ./pt/formal_style.pt \
    --input_jsonl ./input.jsonl \
    --output_jsonl ./output.jsonl \
    --target_layers "18-23" \
    --alpha -2.5
```

#### triple_vector_control.py - 多向量控制

支持同时使用三个不同的风格向量进行更精细的控制：

```python
python triple_vector_control.py generate_batch \
    --base_model_path /path/to/model \
    --vector1_path pt/formal_style.pt \
    --vector2_path pt/knowledge_style.pt \
    --vector3_path pt/detox_style.pt \
    --alpha1 1.0 --alpha2 0.5 --alpha3 -0.3 \
    --input_jsonl input.jsonl \
    --target_layers "15-20"
```

### 评估工具

#### GPT-4 评估

```python
python evaluate_with_gpt_v2.py \
    --input generated_results.jsonl \
    --output evaluated_results.jsonl \
    --criteria formality \
    --model gpt-4-turbo \
    --workers 4
```

#### 统计分析

```python
python analyze_scores_v2.py \
    --input evaluated_results.jsonl \
    --output analysis_report.json
```

## 🔧 高级配置

### 风格向量提取方法

- `pca_denoise`: PCA 降噪方法（推荐）
- `geometric_median`: 几何中位数方法
- `mean`: 简单平均方法

### 生成参数调节

```bash
# 调节风格强度（alpha 值）
--alpha -2.5    # 负值增强相反风格
--alpha 1.5     # 正值增强目标风格

# 目标层选择
--target-layers "18-23"  # 指定层范围
--target-layers "15"     # 单层
--target-layers "10-15,20-25"  # 多个范围

# 生成参数
--max_new_tokens 512
--temperature 0.7
--top_p 0.9
--repetition_penalty 1.2
```

### 批处理参数

```bash
--generation_batch_size 8  # 生成批大小
--compute-perplexity       # 计算困惑度
--no-instruct             # 禁用指令格式
```

## 📊 数据格式

### 输入格式 (JSONL)

```json
{"prompt": "您的提示文本"}
{"prompt": "另一个提示"}
```

### 输出格式 (JSONL)

```json
{
  "index": 0,
  "prompt": "原始提示",
  "generated_text": "生成的文本",
  "alpha": -2.5,
  "target_layers": "18-23"
}
```

### 评估结果格式

```json
{
  "index": 0,
  "prompt": "原始提示",
  "generated_text": "生成的文本",
  "gpt_score": 8.5,
  "gpt_reasoning": "评估理由"
}
```

## 🛠️ 便捷脚本

所有脚本都提供了默认配置和详细的使用说明：

- `script/run_analyze.sh`: 风格向量提取
- `script/run_generate.sh`: 交互式生成
- `script/run_generate_batch.sh`: 批量生成
- `script/pipline.sh`: 完整实验流水线

每个脚本都支持 `--help` 参数查看详细用法。

## 🔬 技术原理

### 激活引导原理

1. **向量提取**：从对比数据集中提取激活差异向量
2. **PCA 降噪**：使用主成分分析去除噪声
3. **实时引导**：在推理时通过钩子函数修改激活

### 风格控制机制

- **单向量控制**：使用单个风格向量进行控制
- **双向量控制**：同时控制两种风格属性
- **三向量控制**：精细控制多种风格维度

### 稳健性保证

- **几何中位数**：对异常值的鲁棒性
- **PCA 降噪**：自动选择最优主成分数量
- **方向对齐**：确保向量方向一致性

## 📈 性能优化

- **批处理**：支持高效的批量推理
- **内存管理**：自动清理激活钩子
- **并行评估**：多进程 GPT 评估
- **增量处理**：跳过已处理的数据

## 🤝 贡献指南

欢迎提交 Issues 和 Pull Requests！

## 📄 许可证

[请添加适当的许可证信息]

## 🙏 致谢

基于激活工程和风格控制的相关研究工作。

---

**注意**：使用前请确保已正确配置 OpenAI API 密钥（如需使用 GPT 评估功能）：

```bash
export OPENAI_API_KEY="your_api_key_here"
export OPENAI_BASE_URL="your_base_url_here"  # 可选
```