import os
import torch
import signal
from time import time
from typing import Any, Literal, Callable
from datasets import DatasetDict
from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.integrations import TensorBoardCallback
from transformers.trainer_callback import ProgressCallback
from accelerate.utils.imports import is_bf16_available, is_cuda_available
from peft import LoraConfig, PeftMixedModel, get_peft_model
import evaluate
from evaluate import EvaluationModule
from huggingface_hub import HfApi
from .prepare_dataset import prepare_dataset
from .utils import DataCollatorSpeechSeq2SeqWithPadding
from .callbacks import (
    WFTTensorBoardCallback,
    WFTProgressCallback,
    ShuffleCallback,
    PushCallback,
)


class WhisperFineTuner:
    def __init__(self, id: str, org: str | None = None):
        """
        Initialize the WhisperFineTuner.

        Args:
            id (str): The unique identifier for the LoRA model, use alphanumeric characters, numbers, and dashes only. (e.g., "whisper-large-v3-turbo-zh-TW-test-1")
            org (str | None): The HuggingFace organization name, if present, the model will be uploaded to the Hub. (default: None)
        """
        self.id = id
        self.org = org
        self.dir = f"./exp/{id}"
        self.baseline: str | None = None
        self.feature_extractor: WhisperFeatureExtractor | None = None
        self.tokenizer: WhisperTokenizer | None = None
        self.processor: WhisperProcessor | None = None
        self.dataset: DatasetDict | None = None
        self.original_dataset: str | None = None
        self.baseline_model: WhisperForConditionalGeneration | None = None
        self.peft_model: PeftMixedModel | None = None
        self.metric_primary: EvaluationModule | None = None
        self.metric_secondary: EvaluationModule | None = None

        self.lora_config = LoraConfig(
            r=32,
            lora_alpha=8,
            use_rslora=True,
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
            lora_dropout=0.05,
            bias="none",
        )

        self.use_bf16 = is_bf16_available()
        self.use_fp16 = is_cuda_available() if not self.use_bf16 else False

        self.training_args = Seq2SeqTrainingArguments(
            output_dir=self.dir,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=8,
            auto_find_batch_size=True,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            generation_max_length=128,
            gradient_accumulation_steps=1,
            learning_rate=5e-6,
            warmup_steps=0,
            max_steps=1000,
            eval_strategy="steps",
            eval_steps=1000,
            eval_on_start=False,  # It may run multiple times if auto_find_batch_size is changing the batch size
            eval_accumulation_steps=32,
            save_strategy="steps",
            save_steps=1000,
            save_total_limit=3,
            load_best_model_at_end=True,
            bf16=self.use_bf16,
            fp16=self.use_fp16,
            remove_unused_columns=False,
            label_names=["labels"],
            report_to=["tensorboard"],
            logging_steps=1,
            push_to_hub=True if org is not None else False,
            hub_model_id=f"{org}/{id}" if org is not None else None,
            hub_strategy="checkpoint",
        )

    def set_baseline(
        self,
        baseline: str,
        language: str,
        task: Literal["transcribe", "translate"] = "transcribe",
    ):
        """
        Set the baseline model and initialize related components.

        Args:
            baseline (str): The name or path of the baseline Whisper model.
            language (str): The target language for the model.
            task (Literal["transcribe", "translate"]): The task to perform (default: "transcribe").

        Returns:
            self: The WhisperFineTuner instance.
        """
        self.baseline = baseline
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(baseline)
        self.tokenizer = WhisperTokenizer.from_pretrained(
            baseline, language=language, task=task
        )
        self.processor = WhisperProcessor.from_pretrained(
            baseline, language=language, task=task
        )
        dtype = (
            torch.bfloat16
            if self.use_bf16
            else torch.float16
            if self.use_fp16
            else torch.float32
        )
        self.baseline_model = WhisperForConditionalGeneration.from_pretrained(
            baseline, load_in_8bit=False, torch_dtype=dtype
        )
        self.baseline_model.config.forced_decoder_ids = None
        self.baseline_model.config.suppress_tokens = []
        return self

    def prepare_dataset(
        self,
        src_name: str,
        src_audio_column: str = "audio",
        src_transcription_column: str = "transcription",
        src_subset: str | None = None,
        src_train_split: str = "train+validation",
        src_test_split: str = "test",
        buffer_size: int = 256,
    ):
        """
        Prepare the dataset for fine-tuning.

        Args:
            src_name (str): The name or path of the source dataset.
            src_audio_column (str): The name of the audio column in the source dataset (default: "audio").
            src_transcription_column (str): The name of the transcription column in the source dataset (default: "transcription").
            src_subset (str | None): The subset of the dataset to use, if any (default: None).
            num_proc (int): The number of processes to use for data preparation (default: 4).

        Returns:
            self: The WhisperFineTuner instance.
        """
        if self.feature_extractor is None or self.tokenizer is None:
            raise ValueError("Please set the baseline first.")
        self.dataset = prepare_dataset(
            src_name=src_name,
            feature_extractor=self.feature_extractor,
            tokenizer=self.tokenizer,
            src_audio_column=src_audio_column,
            src_transcription_column=src_transcription_column,
            src_subset=src_subset,
            src_train_split=src_train_split,
            src_test_split=src_test_split,
            buffer_size=buffer_size,
        )
        self.original_dataset = src_name
        return self

    def set_lora_config(
        self,
        lora_config: LoraConfig,
    ):
        """
        Set the LoRA configuration for fine-tuning.

        Args:
            lora_config (LoraConfig): The LoRA configuration to use.

        Returns:
            self: The WhisperFineTuner instance.
        """
        self.lora_config = lora_config
        return self

    def set_metric(self, metric_type: Literal["cer", "wer"] = "wer"):
        """
        Set the evaluation metric for the model.

        Args:
            metric_type (Literal["cer", "wer"]): The type of metric to use (default: "wer").

        Returns:
            self: The WhisperFineTuner instance.
        """
        self.metric_primary = evaluate.load(metric_type)
        self.metric_secondary = evaluate.load("cer" if metric_type == "wer" else "wer")
        return self

    def set_training_args(self, training_args: Seq2SeqTrainingArguments):
        """
        Set the training arguments for the model.

        Args:
            training_args (Seq2SeqTrainingArguments): The training arguments to use.

        Returns:
            self: The WhisperFineTuner instance.
        """
        self.training_args = training_args
        return self

    def set_steps(self, max_steps: int, eval_steps: int, save_steps: int):
        """
        Set the number of steps for training, evaluation, and saving checkpoints.

        Args:
            max_steps (int): The maximum number of steps for training.
            eval_steps (int): The number of steps between evaluations.
            save_steps (int): The number of steps between saving checkpoints.

        Returns:
            self: The WhisperFineTuner instance.
        """
        self.training_args.max_steps = max_steps
        self.training_args.eval_steps = eval_steps
        self.training_args.save_steps = save_steps
        return self

    def train(
        self,
        training_args: Seq2SeqTrainingArguments | None = None,
        resume: bool = False,
    ):
        """
        Train the model using the prepared dataset and configurations.

        Args:
            training_args (Seq2SeqTrainingArguments | None): The training arguments to use. If None, uses default arguments.
            resume (bool): Whether to resume training from the last checkpoint (default: False).

        Returns:
            self: The WhisperFineTuner instance.
        """
        if self.dataset is None:
            raise ValueError("Please prepare or load the dataset first.")
        if self.feature_extractor is None or self.tokenizer is None:
            raise ValueError("Please set the baseline first.")
        if self.baseline_model is None:
            raise ValueError("Please set the baseline first.")

        if training_args is not None:
            self.training_args = training_args
        self.training_args.num_train_epochs = 0
        print(f"Training arguments: {self.training_args}")

        data_collator = DataCollatorSpeechSeq2SeqWithPadding(self.processor)
        tokenizer = self.tokenizer
        metric_primary = self.metric_primary
        metric_secondary = self.metric_secondary

        def compute_metrics(pred):
            pred_ids = pred.predictions
            label_ids = pred.label_ids

            # replace -100 with the pad_token_id
            label_ids[label_ids == -100] = tokenizer.pad_token_id

            decode_start = time()
            pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
            label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
            decode_runtime = time() - decode_start

            metric_primary_start = time()
            metric_primary_result = 100 * metric_primary.compute(
                predictions=pred_str, references=label_str
            )
            metric_primary_runtime = time() - metric_primary_start

            metric_secondary_start = time()
            metric_secondary_result = 100 * metric_secondary.compute(
                predictions=pred_str, references=label_str
            )
            metric_secondary_runtime = time() - metric_secondary_start

            mismatch_table = (
                "## Incorrect Predictions\n\n"
                "| i | Label | Prediction |\n| --- | --- | --- |\n"
            )
            match_table = (
                "## Correct Predictions\n\n" "| i | Prediction |\n| --- | --- |\n"
            )

            for i, (label, pred) in enumerate(zip(label_str, pred_str)):
                if label != pred:
                    mismatch_table += f"| {i} | {label} | {pred} |\n"
                else:
                    match_table += f"| {i} | {pred} |\n"

            # Close details tags
            mismatch_table += "\n"
            match_table += "\n"

            # Combine both tables
            markdown_table = mismatch_table + "\n" + match_table

            return {
                metric_primary.name: metric_primary_result,
                metric_secondary.name: metric_secondary_result,
                "pred": markdown_table,
                "decode_runtime": decode_runtime,
                f"{metric_primary.name}_runtime": metric_primary_runtime,
                f"{metric_secondary.name}_runtime": metric_secondary_runtime,
            }

        def preprocess_logits_for_metrics(logits, labels):
            # we got a tuple of logits and past_key_values here
            logits = logits[0].argmax(axis=-1)
            return logits

        self.training_args.metric_for_best_model = self.metric_primary.name
        self.training_args.greater_is_better = False

        self.lora_config.total_step = self.training_args.max_steps
        self.lora_config.tinit = self.training_args.warmup_steps

        if resume:
            try:
                if self.org is not None and not os.path.exists(self.dir):
                    hf = HfApi()
                    hf.snapshot_download(
                        self.training_args.hub_model_id, local_dir=self.dir
                    )
                if get_last_checkpoint(self.dir) is None:
                    raise Exception("No checkpoint found.")
            except Exception as e:
                print(f"Failed to resume training: {e}, starting from scratch.")
                resume = False
        self.peft_model = get_peft_model(self.baseline_model, self.lora_config)
        self.peft_model.print_trainable_parameters()

        self.trainer = trainer = Seq2SeqTrainer(
            model=self.peft_model,
            args=self.training_args,
            train_dataset=self.dataset["train"],
            eval_dataset=self.dataset["test"],
            tokenizer=self.feature_extractor,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )
        self.peft_model.config.use_cache = False

        trainer.remove_callback(TensorBoardCallback)
        trainer.remove_callback(ProgressCallback)
        trainer.add_callback(WFTProgressCallback())
        if (
            isinstance(self.training_args.report_to, list)
            and "tensorboard" in self.training_args.report_to
        ) or (
            isinstance(self.training_args.report_to, str)
            and self.training_args.report_to == "tensorboard"
        ):
            trainer.add_callback(WFTTensorBoardCallback())
        trainer.add_callback(ShuffleCallback())
        trainer.add_callback(PushCallback(self))

        def signal_handler(sig, frame):
            print("Training stopped by user.")
            trainer.save_model()
            print("Model saved locally.")
            if self.org is not None:
                self.push_to_hub()
                print("Model pushed to the Hub.")
            print("Bye!")
            exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        trainer.train(resume_from_checkpoint=resume)
        trainer.save_model(_internal_call=True)

        self.push_to_hub()

        return self

    def push_to_hub(self):
        if self.trainer is None:
            raise ValueError("Please train the model first.")

        if self.org is not None:
            # temporarily remove large preds and runtime metrics
            temp_storage = {
                "preds": [],
                "decode_runtime": [],
                f"{self.metric_primary.name}_runtime": [],
                f"{self.metric_secondary.name}_runtime": [],
            }
            for log in self.trainer.state.log_history:
                temp_storage["preds"].append(log.pop("eval_pred", None))
                temp_storage["decode_runtime"].append(log.pop("decode_runtime", None))
                temp_storage[f"{self.metric_primary.name}_runtime"].append(
                    log.pop(f"{self.metric_primary.name}_runtime", None)
                )
                temp_storage[f"{self.metric_secondary.name}_runtime"].append(
                    log.pop(f"{self.metric_secondary.name}_runtime", None)
                )

            self.trainer.push_to_hub(
                language=self.tokenizer.language,
                finetuned_from=self.baseline,
                tasks="automatic-speech-recognition",
                dataset_tags=self.original_dataset,
                tags=[
                    "wft",
                    "whisper",
                    "automatic-speech-recognition",
                    "audio",
                    "speech",
                ],
            )

            # restore preds and runtime metrics
            for log, pred, decode_runtime, primary_runtime, secondary_runtime in zip(
                self.trainer.state.log_history,
                temp_storage["preds"],
                temp_storage["decode_runtime"],
                temp_storage[f"{self.metric_primary.name}_runtime"],
                temp_storage[f"{self.metric_secondary.name}_runtime"],
            ):
                log["eval_pred"] = pred
                log["decode_runtime"] = decode_runtime
                log[f"{self.metric_primary.name}_runtime"] = primary_runtime
                log[f"{self.metric_secondary.name}_runtime"] = secondary_runtime

    def merge(
        self, dtype: torch.dtype | None = None
    ) -> WhisperForConditionalGeneration:
        """
        Merge the LoRA weights with the base model.

        Args:
            dtype (torch.dtype | None): The data type to use for the merged model (default: None).

        Returns:
            WhisperForConditionalGeneration: The merged model.
        """
        model = self.peft_model.merge_and_unload()
        if dtype is None:
            dtype = (
                torch.bfloat16
                if self.use_bf16
                else torch.float16
                if self.use_fp16
                else torch.float32
            )
        return model.to(dtype=dtype)

    def merge_and_save(self, outdir: str, dtype: torch.dtype | None = None):
        """
        Merge the LoRA weights with the base model and save it to a directory.

        Args:
            outdir (str): The output directory to save the merged model.
            dtype (torch.dtype | None): The data type to use for the merged model (default: None).

        Returns:
            self: The WhisperFineTuner instance.
        """
        model = self.merge(dtype)
        model.save_pretrained(outdir)
        self.processor.save_pretrained(outdir)
        self.tokenizer.save_pretrained(outdir)
        return self

    def merge_and_push(
        self, dest_name: str | None = None, dtype: torch.dtype | None = None
    ):
        """
        Merge the LoRA weights with the base model and upload it to the Hugging Face Hub.

        Args:
            dest_name (str | None): The destination name for the model on the Hub. If None, uses the original model ID + "-merged".
            dtype (torch.dtype | None): The data type to use for the merged model (default: None).

        Returns:
            self: The WhisperFineTuner instance.
        """
        if dest_name is None:
            if self.training_args.hub_model_id is None:
                raise ValueError("Please set the model name to push the model.")
            dest_name = f"{self.training_args.hub_model_id}-merged"

        model = self.merge(dtype)
        model.push_to_hub(dest_name)
        self.processor.push_to_hub(dest_name)
        self.tokenizer.push_to_hub(dest_name)

        with open(os.path.join(self.dir, "README.md"), "r") as f:
            content = f.read()
        content = content.replace("library_name: peft", "library_name: transformers")

        hf = HfApi()
        hf.upload_file(
            repo_id=dest_name,
            path_in_repo="README.md",
            path_or_fileobj=bytes(content, "utf-8"),
        )
        return self

    def then(self, func: Callable[["WhisperFineTuner"], Any]):
        """
        Execute a custom function on the WhisperFineTuner instance.

        Args:
            func (Callable[["WhisperFineTuner"], Any]): The function to execute.

        Returns:
            self: The WhisperFineTuner instance.
        """
        func(self)
        return self
