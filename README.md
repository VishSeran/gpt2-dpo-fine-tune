# 🤖 DPO Fine-Tuning with Hugging Face TRL

Fine-tune language models using **Direct Preference Optimization (DPO)** — a simpler, more stable alternative to RLHF that directly optimizes a model on human preference data without a separate reward model.

This repository walks through a complete DPO pipeline using Hugging Face's `trl` library, `PEFT`/`LoRA` for parameter-efficient training, and the `UltraFeedback` binarized dataset.

---

## 📖 What is DPO?

DPO (Direct Preference Optimization) is an alignment technique that trains a language model to prefer "chosen" responses over "rejected" ones, given a prompt. Unlike PPO-based RLHF, DPO:

- Requires **no separate reward model**
- Is **simpler to implement and tune**
- Is **more training-stable**
- Works directly on preference pairs: `{prompt, chosen, rejected}`

---

## 📁 Repository Structure

```
.
├── DPO_Fine-Tuning-v1.ipynb   # Main notebook — full DPO pipeline
├── README.md
└── requirements.txt           # Pinned dependencies
```

---

## 🚀 Quickstart

### 1. Clone the repo

```bash
git clone https://github.com/your-username/dpo-fine-tuning.git
cd dpo-fine-tuning
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install trl==0.11.4 peft==0.14.0 transformers==4.45.2 \
            datasets==3.2.0 matplotlib==3.9.0 numpy==1.26.0 \
            torch pandas rich
```

### 3. Run the notebook

```bash
jupyter notebook DPO_Fine-Tuning-v1.ipynb
```

---

## 🧪 Pipeline Overview

### 1. Model & Tokenizer Setup
Loads **meta-llama/Llama-3.2-1B-Instruct** as the base model and configures the tokenizer with right-side padding — required for stable causal LM training.

### 2. Dataset Preparation
Uses the [`BarraHome/ultrafeedback_binarized`](https://huggingface.co/datasets/BarraHome/ultrafeedback_binarized) dataset. Each record is preprocessed into the required DPO format:

| Field | Description |
|---|---|
| `prompt` | The user instruction |
| `chosen` | The preferred response |
| `rejected` | The dispreferred response |

### 3. LoRA Configuration
Applies **LoRA** (Low-Rank Adaptation) via `peft` to keep memory footprint small:

```python
LoraConfig(
    r=4,
    target_modules=['c_proj', 'c_attn'],
    task_type="CAUSAL_LM",
    lora_alpha=8,
    lora_dropout=0.1,
)
```

### 4. DPO Training
Configures and runs `DPOTrainer` with:
- `beta=0.1` (KL penalty — controls deviation from the reference policy)
- 5 training epochs
- Learning rate: `1e-4`
- Per-device batch size: `1` (CPU-friendly)

### 5. Evaluation
Plots training vs. evaluation loss curves and compares generated outputs between the base meta-llama/Llama-3.2-1B-Instruct and the DPO-tuned model on the same prompt.

---

## ⚙️ Configuration Reference

| Parameter | Value | Notes |
|---|---|---|
| `beta` | `0.1` | Typical range: 0.1 – 0.5 |
| `learning_rate` | `1e-4` | Standard for LoRA fine-tuning |
| `num_train_epochs` | `5` | Increase for larger datasets |
| `lora_r` | `4` | Increase to 16–64 for production |
| `max_length` | `512` | Truncates combined prompt+response |
| `per_device_train_batch_size` | `1` | Increase with GPU memory |

---

## 💡 GPU / Quantization (Optional)

For faster training on GPU with reduced memory, uncomment the **quantized model block** in the notebook. This uses `bitsandbytes` for 4-bit NF4 quantization:

```bash
pip install bitsandbytes
```

Then restart your kernel and use the `BitsAndBytesConfig` block provided in the notebook.

---

## 📊 Results

After 5 epochs on 50 training samples, the DPO-tuned model produces more concise and on-topic responses compared to the base meta-llama/Llama-3.2-1B-Instruct:

| Model | Response to *"Is higher octane gasoline better for your car?"* |
|---|---|
| meta-llama/Llama-3.2-1B-Instruct (base) | Generic, off-topic continuation |
| meta-llama/Llama-3.2-1B-Instruct (DPO-tuned) | More direct, preference-aligned response |

> **Note:** Results are illustrative due to the small dataset size used in this notebook. For meaningful alignment, scale to 10K+ preference pairs.

---

## ⚠️ Known Limitations

- **50 training samples** is insufficient for production alignment — used here for CPU/resource compatibility only.
- No reward margin or win-rate metrics are tracked; add these for rigorous evaluation.

---

