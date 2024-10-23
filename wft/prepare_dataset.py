from datasets import DatasetDict, load_dataset, Audio
from transformers import WhisperFeatureExtractor, WhisperTokenizer


def prepare_dataset(
    src_name: str,
    feature_extractor: WhisperFeatureExtractor,
    tokenizer: WhisperTokenizer,
    src_audio_column: str = "audio",
    src_transcription_column: str = "transcription",
    src_subset: str | None = None,
    src_train_split: str = "train+validation",
    src_test_split: str = "test",
    num_proc: int = 4,
) -> DatasetDict:
    ds = DatasetDict()

    ds["train"] = load_dataset(
        src_name,
        src_subset,
        split=src_train_split,
        trust_remote_code=True,
        num_proc=num_proc,
    )
    ds["test"] = load_dataset(
        src_name,
        src_subset,
        split=src_test_split,
        trust_remote_code=True,
        num_proc=num_proc,
    )
    print("loaded source dataset", ds)

    # remove all non-audio/transcription columns
    ds["train"] = ds["train"].remove_columns(
        [
            col
            for col in ds["train"].column_names
            if col not in [src_audio_column, src_transcription_column]
        ]
    )
    ds["test"] = ds["test"].remove_columns(
        [
            col
            for col in ds["test"].column_names
            if col not in [src_audio_column, src_transcription_column]
        ]
    )

    # resample the audio to 16kHz
    ds = ds.cast_column(src_audio_column, Audio(sampling_rate=16000, mono=True))

    def prepare_dataset(batch):
        audio = batch[src_audio_column]
        batch["input_features"] = feature_extractor(
            audio["array"], sampling_rate=audio["sampling_rate"]
        ).input_features[0]
        batch["labels"] = tokenizer(batch[src_transcription_column]).input_ids
        return batch

    ds = ds.map(
        prepare_dataset, remove_columns=ds.column_names["train"], num_proc=num_proc
    )

    ds = ds.with_format("torch")

    return ds
