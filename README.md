# 🗣️ Whisper Fine-Tuning (WFT)

**WFT** is a 🐍 Python library designed to streamline the fine-tuning process of 🤖 OpenAI's Whisper models on custom datasets. It simplifies 📦 dataset preparation, model 🛠️ fine-tuning, and result 💾 saving.

## ✨ Features

- **🤗 Hugging Face Integration**: Set your organization (or user) name, and everything syncs automatically to the 🤗 Hugging Face hub.
- **📄 Easy Dataset Preparation and Preprocessing**: Quickly prepare and preprocess datasets for 🛠️ fine-tuning.
- **🔧 Fine-Tuning Using LoRA (Low-Rank Adaptation)**: Fine-tune Whisper models with efficient LoRA techniques.
- **⚙️ Flexible Configuration Options**: Customize various aspects of the fine-tuning process.
- **📊 Evaluation Metrics**: Supports Character Error Rate (CER) or Word Error Rate (WER) for model evaluation.
- **📈 TensorBoard Logging**: Track your training progress in real-time with TensorBoard.
- **🤖 Automatic Model Merging and Saving**: Automatically merge fine-tuned weights and save the final model.
- **🔄 Resume Training**: Resume training seamlessly from interrupted runs.

## 🛠️ Installation

Install WFT using 🐍 pip:

```bash
pip install wft
```

## 🚀 Quick Start

Fine-tune a Whisper model on a custom dataset with just a few steps:

1. **🧩 Select a Baseline Model**: Choose a pre-trained Whisper model.
2. **🎵 Select a Dataset**: Use a dataset that includes 🎧 audio and ✍️ transcription columns.
3. **🏋️‍♂️ Start Training**: Use default training arguments to quickly fine-tune the model.

Here's an example:

```python
from wft import WhisperFineTuner

id = "whisper-large-v3-turbo-zh-TW-test-1"

ft = (
    WhisperFineTuner(id)
    .set_baseline("openai/whisper-large-v3-turbo", language="zh", task="transcribe")
    .prepare_dataset(
        "mozilla-foundation/common_voice_16_1",
        src_subset="zh-TW",
        src_audio_column="audio",
        src_transcription_column="sentence",
    )
    .train()  # Use default training arguments
)
```

That's it! 🎉 You can now fine-tune your Whisper model easily.

To enable 🤗 Hugging Face integration and push your training log and model to Hugging Face, set the `org` parameter when initializing `WhisperFineTuner`:

```python
id = "whisper-large-v3-turbo-zh-TW-test-1"
org = "JacobLinCool"  # Organization to push the model to Hugging Face

ft = (
    WhisperFineTuner(id, org)
    .set_baseline("openai/whisper-large-v3-turbo", language="zh", task="transcribe")
    .prepare_dataset(
        "mozilla-foundation/common_voice_16_1",
        src_subset="zh-TW",
        src_audio_column="audio",
        src_transcription_column="sentence",
    )
    .train()  # Use default training arguments
    .merge_and_push()  # Merge the model and push it to Hugging Face
)
```

This will automatically push your training logs 📜 and the fine-tuned model 🤖 to your Hugging Face account under the specified organization.

## 📚 Usage Guide

### 1️⃣ Set Baseline Model and Prepare Dataset

You can use a local dataset or a dataset from 🤗 Hugging Face:

```python
ft = (
    WhisperFineTuner(id)
    .set_baseline("openai/whisper-large-v3-turbo", language="zh", task="transcribe")
    .prepare_dataset(
        "mozilla-foundation/common_voice_16_1",
        src_subset="zh-TW",
        src_audio_column="audio",
        src_transcription_column="sentence",
    )
)
```

To upload the preprocessed dataset to Hugging Face:

```python
ft.push_dataset("username/dataset_name")
```

You can also prepare or load an already processed dataset:

```python
ft = (
    WhisperFineTuner(id)
    .set_baseline("openai/whisper-large-v3-turbo", language="zh", task="transcribe")
    .prepare_dataset(
        "username/preprocessed_dataset",
        "mozilla-foundation/common_voice_16_1",
        src_subset="zh-TW",
        src_audio_column="audio",
        src_transcription_column="sentence",
    )
)
```

### 2️⃣ Configure Fine-Tuning

Set the evaluation metric and 🔧 LoRA configuration:

```python
ft.set_metric("cer")  # Use CER (Character Error Rate) for evaluation

# Custom LoRA configuration to fine-tune different parts of the model
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

You can also set custom 🛠️ training arguments:

```python
from transformers import Seq2SeqTrainingArguments

custom_training_args = Seq2SeqTrainingArguments(
    output_dir=ft.dir,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    learning_rate=1e-4,
    num_train_epochs=3,
)

ft.set_training_args(custom_training_args)
```

### 3️⃣ Train the Model

To begin 🏋️‍♂️ fine-tuning:

```python
ft.train()
```

### 4️⃣ Save or Push the Fine-Tuned Model

Merge 🔧 LoRA weights with the baseline model and save it:

```python
ft.merge_and_save(f"{ft.dir}/merged_model")

# Or push to Hugging Face
ft.merge_and_push("username/merged_model")
```

## 🔬 Advanced Usage

### 🔧 Custom LoRA Configuration

Adjust the LoRA configuration to fine-tune different model parts:

```python
custom_lora_config = LoraConfig(
    r=32,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
)

ft.set_lora_config(custom_lora_config)
```

### ⚙️ Custom Training Arguments

Specify custom 🛠️ training settings:

```python
from transformers import Seq2SeqTrainingArguments

custom_training_args = Seq2SeqTrainingArguments(
    output_dir=ft.dir,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    learning_rate=1e-4,
    num_train_epochs=3,
)

ft.set_training_args(custom_training_args)
```

### 🔁 Run Custom Actions After Steps Using `.then()`

Add actions to be executed after each step:

```python
ft = (
    WhisperFineTuner(id)
    .set_baseline("openai/whisper-large-v3-turbo", language="zh", task="transcribe")
    .then(lambda ft: print(f"{ft.baseline_model=}"))
    .prepare_dataset(
        "mozilla-foundation/common_voice_16_1",
        src_subset="zh-TW",
        src_audio_column="audio",
        src_transcription_column="sentence",
    )
    .then(lambda ft: print(f"{ft.dataset=}"))
    .set_metric("cer")
    .then(lambda ft: setattr(ft.training_args, "num_train_epochs", 5))
    .train()
)
```

### 🔄 Resume Training From a Checkpoint

If training is interrupted, you can resume:

```python
ft = (
    WhisperFineTuner(id)
    .set_baseline("openai/whisper-large-v3-turbo", language="zh", task="transcribe")
    .prepare_dataset(
        "mozilla-foundation/common_voice_16_1",
        src_subset="zh-TW",
        src_audio_column="audio",
        src_transcription_column="sentence",
    )
    .set_metric("cer")
    .train(resume=True)
)
```

> **ℹ️ Note**: If no checkpoint is found, training will start from scratch without failure.

## 🤝 Contributing

We welcome contributions! 🎉 Feel free to submit a pull request.

## 📜 License

This project is licensed under the MIT License.

## Why there are so many emojis in this README?

Because ChatGPT told me to do so. 🤖📝
