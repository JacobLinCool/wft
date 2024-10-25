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

Here's a simple example to fine-tune a Whisper model on a custom dataset and upload the results to Hugging Face:

```python
from wft import WhisperFineTuner

id = "whisper-large-v3-turbo-zh-TW-test-1"
org = "JacobLinCool" # if you want to push to Hugging Face
ft = (
    WhisperFineTuner(id, org)
    .set_baseline("openai/whisper-large-v3-turbo", language="zh", task="transcribe")
    .load_or_prepare_dataset(
        "mozilla-foundation/common_voice_16_1",
        src_subset="zh-TW",
        src_audio_column="audio",
        src_transcription_column="sentence",
    )
    .set_metric("cer")
    .train()  # Use default training arguments
)
```

## Usage

### 1. Set Baseline Model and Prepare Dataset

You can prepare a dataset from a local source or use a pre-existing Hugging Face dataset:

```python
ft = WhisperFineTuner(id)
    .set_baseline("openai/whisper-large-v3-turbo", language="zh", task="transcribe")
    .prepare_dataset(
        "mozilla-foundation/common_voice_16_1",
        src_subset="zh-TW",
        src_audio_column="audio",
        src_transcription_column="sentence",
    )
```

### 2. Configure Fine-tuning

Set the evaluation metric and LoRA configuration:

```python
# both CER and WER will be calculated, select one to use for selecting the best model
ft.set_metric("cer")

# The default LoRA configuration trains ~3% of the model's parameters
# You can pass a custom LoRA configuration to fine-tune more or fewer parameters
ft.set_lora_config(LoraConfig(...))

# You can use your own training arguments with the following method
ft.set_training_args(Seq2SeqTrainingArguments(...))
# or directly set the training arguments like
ft.training_args.num_train_epochs = 3
```

### 3. Train the Model

Start the fine-tuning process:

```python
ft.train()
```

### 4. Save the merged Fine-tuned Model

Merge the LoRA weights with the base model and save:

```python
ft.merge_and_save(f"{ft.dir}/merged_model")

# or push to Hugging Face
ft.merge_and_push("username/merged_model")
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
    output_dir=ft.dir,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    learning_rate=1e-4,
    num_train_epochs=3,
    # Add more arguments as needed
)

ft.set_training_args(custom_training_args)
# or just pass it to the train method
ft.train(custom_training_args)
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.
