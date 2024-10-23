from .prepare_dataset import prepare_dataset
from .utils import DataCollatorSpeechSeq2SeqWithPadding
from typing import Any, Literal, Callable
from datasets import DatasetDict, load_dataset
from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from peft import LoraConfig, PeftMixedModel, get_peft_model
import evaluate
from evaluate import EvaluationModule


class WhisperFineTuner:
    def __init__(self, dir: str):
        """
        Initialize the WhisperFineTuner.

        Args:
            dir (str): The directory to save output files.
        """
        self.dir = dir
        self.baseline: str | None = None
        self.feature_extractor: WhisperFeatureExtractor | None = None
        self.tokenizer: WhisperTokenizer | None = None
        self.processor: WhisperProcessor | None = None
        self.dataset: DatasetDict
        self.baseline_model: WhisperForConditionalGeneration | None = None
        self.lora_config: LoraConfig | None = None
        self.peft_model: PeftMixedModel | None = None
        self.metric: EvaluationModule | None = None

        self.default_lora_config = LoraConfig(
            r=32,
            lora_alpha=8,
            use_rslora=True,
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
            lora_dropout=0.05,
            bias="none",
        )

        self.default_training_args = Seq2SeqTrainingArguments(
            output_dir=self.dir,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            auto_find_batch_size=True,
            generation_max_length=128,
            gradient_accumulation_steps=1,
            learning_rate=5e-4,
            num_train_epochs=5,
            warmup_steps=0,
            eval_strategy="epoch",
            eval_on_start=True,
            save_strategy="epoch",
            save_total_limit=3,
            load_best_model_at_end=True,
            fp16=True,
            remove_unused_columns=False,
            label_names=["labels"],
            report_to="tensorboard",
            logging_steps=1,
        )
        pass

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
        self.baseline_model = WhisperForConditionalGeneration.from_pretrained(
            baseline, load_in_8bit=False
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
        num_proc: int = 4,
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
            num_proc=num_proc,
        )
        return self

    def push_dataset(self, dest_name: str):
        """
        Push the prepared dataset to the Hugging Face Hub.

        Args:
            dest_name (str): The destination name for the dataset on the Hub.

        Returns:
            self: The WhisperFineTuner instance.
        """
        if self.dataset is None:
            raise ValueError("Please prepare or load the dataset first.")
        self.dataset.push_to_hub(dest_name)
        return self

    def load_dataset(self, dest_name: str):
        """
        Load a preprocessed dataset from the Hugging Face Hub.

        Args:
            dest_name (str): The name of the dataset on the Hub.

        Returns:
            self: The WhisperFineTuner instance.
        """
        ds = load_dataset(dest_name)

        if "train" not in ds or "test" not in ds:
            raise ValueError("Dataset does not contain both train and test splits.")

        if (
            "input_features" not in ds.column_names["train"]
            or "labels" not in ds.column_names["train"]
        ):
            raise ValueError(
                "Dataset (train) does not contain both input_features and labels columns."
            )

        if (
            "input_features" not in ds.column_names["test"]
            or "labels" not in ds.column_names["test"]
        ):
            raise ValueError(
                "Dataset (test) does not contain both input_features and labels columns."
            )

        self.dataset = ds.with_format("torch")
        return self

    def load_or_prepare_dataset(
        self,
        preprocessed_dataset_name: str,
        src_name: str,
        src_audio_column: str = "audio",
        src_transcription_column: str = "transcription",
        src_subset: str | None = None,
        src_train_split: str = "train+validation",
        src_test_split: str = "test",
        num_proc: int = 4,
    ):
        """
        Load a preprocessed dataset if available, or prepare and push a new one.

        Args:
            preprocessed_dataset_name (str): The name of the preprocessed dataset on the Hub.
            src_name (str): The name or path of the source dataset.
            src_audio_column (str): The name of the audio column in the source dataset (default: "audio").
            src_transcription_column (str): The name of the transcription column in the source dataset (default: "transcription").
            src_subset (str | None): The subset of the dataset to use, if any (default: None).
            num_proc (int): The number of processes to use for data preparation (default: 4).

        Returns:
            self: The WhisperFineTuner instance.
        """
        try:
            self.load_dataset(preprocessed_dataset_name)
        except:
            self.prepare_dataset(
                src_name=src_name,
                src_audio_column=src_audio_column,
                src_transcription_column=src_transcription_column,
                src_subset=src_subset,
                src_train_split=src_train_split,
                src_test_split=src_test_split,
                num_proc=num_proc,
            )
            self.push_dataset(preprocessed_dataset_name)
        return self

    def set_lora_config(
        self,
        lora_config: LoraConfig | None = None,
    ):
        """
        Set the LoRA configuration for fine-tuning.

        Args:
            lora_config (LoraConfig | None): The LoRA configuration to use. If None, uses the default configuration.

        Returns:
            self: The WhisperFineTuner instance.
        """
        if self.baseline_model is None:
            raise ValueError("Please set the baseline first.")
        if lora_config is None:
            lora_config = self.default_lora_config
        self.peft_model = get_peft_model(self.baseline_model, lora_config)
        return self

    def set_metric(self, metric_type: Literal["cer", "wer"] = "wer"):
        """
        Set the evaluation metric for the model.

        Args:
            metric_type (Literal["cer", "wer"]): The type of metric to use (default: "wer").

        Returns:
            self: The WhisperFineTuner instance.
        """
        self.metric = evaluate.load(metric_type)
        return self

    def train(self, training_args: Seq2SeqTrainingArguments | None = None):
        """
        Train the model using the prepared dataset and configurations.

        Args:
            training_args (Seq2SeqTrainingArguments | None): The training arguments to use. If None, uses default arguments.

        Returns:
            self: The WhisperFineTuner instance.
        """
        if self.dataset is None:
            raise ValueError("Please prepare or load the dataset first.")
        if self.feature_extractor is None or self.tokenizer is None:
            raise ValueError("Please set the baseline first.")
        if self.peft_model is None:
            raise ValueError("Please set the lora config first.")

        if training_args is None:
            training_args = self.default_training_args

        data_collator = DataCollatorSpeechSeq2SeqWithPadding(self.processor)
        tokenizer = self.tokenizer
        metric = self.metric

        def compute_metrics(pred):
            pred_ids = pred.predictions
            label_ids = pred.label_ids
            pred_ids = pred_ids[0].argmax(
                axis=-1
            )  # we got a tuple of logits and past_key_values here

            # replace -100 with the pad_token_id
            label_ids[label_ids == -100] = tokenizer.pad_token_id

            # we do not want to group tokens when computing the metrics
            pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
            label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

            wer_or_cer = 100 * metric.compute(
                predictions=pred_str, references=label_str
            )

            return {metric.name: wer_or_cer}

        training_args.metric_for_best_model = self.metric.name
        training_args.greater_is_better = False

        self.trainer = trainer = Seq2SeqTrainer(
            model=self.peft_model,
            args=training_args,
            train_dataset=self.dataset["train"],
            eval_dataset=self.dataset["test"],
            tokenizer=self.feature_extractor,
            data_collator=data_collator,
            compute_metrics=compute_metrics if self.metric is not None else None,
        )
        self.peft_model.config.use_cache = False

        trainer.train()
        return self

    def merge(self) -> WhisperForConditionalGeneration:
        """
        Merge the LoRA weights with the base model.

        Returns:
            WhisperForConditionalGeneration: The merged model.
        """
        return self.peft_model.merge_and_unload()

    def merge_and_save(self, outdir: str):
        """
        Merge the LoRA weights with the base model and save it to a directory.

        Args:
            outdir (str): The output directory to save the merged model.

        Returns:
            self: The WhisperFineTuner instance.
        """
        self.merge().save_pretrained(outdir)
        self.processor.save_pretrained(outdir)
        return self

    def merge_and_upload(self, dest_name: str):
        """
        Merge the LoRA weights with the base model and upload it to the Hugging Face Hub.

        Args:
            dest_name (str): The destination name for the model on the Hub.

        Returns:
            self: The WhisperFineTuner instance.
        """
        self.merge().push_to_hub(dest_name)
        self.processor.push_to_hub(dest_name)
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


if __name__ == "__main__":

    def filter_dataset(ft: WhisperFineTuner):
        # only use 3% to test if the code works
        ft.dataset["train"] = ft.dataset["train"].select(
            range(0, len(ft.dataset["train"]), 33)
        )
        ft.dataset["test"] = ft.dataset["test"].select(
            range(0, len(ft.dataset["test"]), 33)
        )
        print(ft.dataset)
        return ft

    outdir = "./test"
    ft = (
        WhisperFineTuner(outdir)
        .set_baseline("openai/whisper-large-v3-turbo", language="zh", task="transcribe")
        .then(lambda ft: print(ft.baseline_model))
        # .prepare_dataset(
        #     "mozilla-foundation/common_voice_16_1",
        #     src_subset="zh-TW",
        #     src_audio_column="audio",
        #     src_transcription_column="sentence",
        # )
        # .push_dataset("JacobLinCool/mozilla-foundation-common_voice_16_1-zh-TW-preprocessed")
        .load_dataset(
            "JacobLinCool/mozilla-foundation-common_voice_16_1-zh-TW-preprocessed"
        )
        .then(filter_dataset)
        .set_metric("cer")
        .set_lora_config()
        .train()
    )

    tuned_model = ft.merge()
    tuned_model.save_pretrained(f"{outdir}/merged_model")
    ft.processor.save_pretrained(f"{outdir}/merged_model")
