from datasets import IterableDataset, Dataset, Audio, load_dataset, interleave_datasets
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
    buffer_size: int = 256,
):
    train = load_streaming_dataset(
        src_name,
        src_subset,
        split=src_train_split,
        trust_remote_code=True,
    )
    test = load_dataset(
        src_name,
        src_subset,
        split=src_test_split,
        trust_remote_code=True,
    )
    print("loaded source dataset", train, test)

    # remove all non-audio/transcription columns
    train = train.remove_columns(
        [
            col
            for col in train.column_names
            if col not in [src_audio_column, src_transcription_column]
        ]
    )
    test = test.remove_columns(
        [
            col
            for col in test.column_names
            if col not in [src_audio_column, src_transcription_column]
        ]
    )

    # resample the audio to 16kHz
    train = train.cast_column(src_audio_column, Audio(sampling_rate=16000, mono=True))
    test = test.cast_column(src_audio_column, Audio(sampling_rate=16000, mono=True))

    def prepare_dataset(batch):
        audio = batch[src_audio_column]
        batch["input_features"] = feature_extractor(
            audio["array"], sampling_rate=audio["sampling_rate"]
        ).input_features[0]
        batch["labels"] = tokenizer(batch[src_transcription_column]).input_ids
        return batch

    train = train.map(
        prepare_dataset,
        remove_columns=list(train.features),
    )
    test = test.map(prepare_dataset, remove_columns=test.column_names)

    train = train.with_format("torch")
    test = test.with_format("torch")

    train = train.shuffle(buffer_size=buffer_size, seed=42)

    return {"train": train, "test": test}
