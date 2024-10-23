# Whisper Fine-Tuning (WFT)

WFT is a Python library for fine-tuning OpenAI's Whisper models on custom datasets. It simplifies the process of preparing datasets, fine-tuning models, and saving the results.

## Features

- Easy dataset preparation and preprocessing
- Fine-tuning Whisper models using LoRA (Low-Rank Adaptation)
- Support for custom datasets and Hugging Face datasets
- Flexible configuration options
- Metric calculation (CER or WER)

## Installation

```bash
pip install wft
```

## Quick Start

Here's a simple example to fine-tune a Whisper model on a custom dataset:

```python
from wft import WhisperFineTuner

outdir = "output"
ft = (
    WhisperFineTuner(outdir)
    .prepare_dataset(
        "mozilla-foundation/common_voice_16_1",
        src_subset="zh-TW",
        src_audio_column="audio",
        src_transcription_column="sentence",
    )
    .set_baseline("openai/whisper-large-v3-turbo", language="zh", task="transcribe")
    .set_metric("cer")
    .set_lora_config()  # Use default LoRA config
    .train()  # Use default training arguments
    .merge_and_save(f"{outdir}/merged_model")
)
```

## Usage

### 1. Prepare Dataset

You can prepare a dataset from a local source or use a pre-existing Hugging Face dataset:

```python
ft = WhisperFineTuner(outdir)
ft.prepare_dataset(
    "mozilla-foundation/common_voice_16_1",
    src_subset="zh-TW",
    src_audio_column="audio",
    src_transcription_column="sentence",
)
```

### 2. Set Baseline Model

Choose a Whisper model as your baseline:

```python
ft.set_baseline("openai/whisper-large-v3-turbo", language="zh", task="transcribe")
```

### 3. Configure Fine-tuning

Set the evaluation metric and LoRA configuration:

```python
ft.set_metric("cer")
ft.set_lora_config()  # Use default LoRA config
```

### 4. Train the Model

Start the fine-tuning process:

```python
ft.train()  # Use default training arguments
```

### 5. Save the Fine-tuned Model

Merge the LoRA weights with the base model and save:

```python
ft.merge_and_save(f"{outdir}/merged_model")
```

## Advanced Usage

### Custom LoRA Configuration

You can customize the LoRA configuration:

```python
from peft import LoraConfig

custom_lora_config = LoraConfig(
    r=32,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
)

ft.set_lora_config(custom_lora_config)
```

### Custom Training Arguments

Customize the training process:

```python
from transformers import Seq2SeqTrainingArguments

custom_training_args = Seq2SeqTrainingArguments(
    output_dir=outdir,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    learning_rate=1e-4,
    num_train_epochs=3,
    # Add more arguments as needed
)

ft.train(custom_training_args)
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.
