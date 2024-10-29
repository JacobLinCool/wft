from datasets import IterableDatasetDict, Audio, load_dataset, interleave_datasets
from transformers import WhisperFeatureExtractor, WhisperTokenizer


def load_streaming_dataset(dataset_name, dataset_config_name, split, **kwargs):
    if "+" in split:
        # load multiple splits separated by the `+` symbol *with* streaming mode
        dataset_splits = [
            load_dataset(
                dataset_name,
                dataset_config_name,
                split=split_name,
                streaming=True,
                **kwargs,
            )
            for split_name in split.split("+")
        ]
        # interleave multiple splits to form one dataset
        interleaved_dataset = interleave_datasets(dataset_splits)
        return interleaved_dataset
    else:
        # load a single split *with* streaming mode
        dataset = load_dataset(
            dataset_name, dataset_config_name, split=split, streaming=True, **kwargs
        )
        return dataset


def prepare_dataset(
    src_name: str,
    feature_extractor: WhisperFeatureExtractor,
    tokenizer: WhisperTokenizer,
    src_audio_column: str = "audio",
    src_transcription_column: str = "transcription",
    src_subset: str | None = None,
    src_train_split: str = "train+validation",
    src_test_split: str = "test",
) -> IterableDatasetDict:
    ds = IterableDatasetDict()

    ds["train"] = load_streaming_dataset(
        src_name,
        src_subset,
        split=src_train_split,
        trust_remote_code=True,
    )
    ds["test"] = load_streaming_dataset(
        src_name,
        src_subset,
        split=src_test_split,
        trust_remote_code=True,
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
        prepare_dataset,
        remove_columns=list(next(iter(ds.values())).features),
    )

    ds = ds.with_format("torch")

    ds["train"] = ds["train"].shuffle(buffer_size=1024, seed=42)

    return ds
