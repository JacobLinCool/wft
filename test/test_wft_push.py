import os
import shutil
from wft import WhisperFineTuner


def filter4(ft: WhisperFineTuner):
    # Filter to 4 samples each
    ft.dataset["train"] = ft.dataset["train"].select(range(4))
    ft.dataset["test"] = ft.dataset["test"].select(range(4))
    print(f"Filtered dataset: {ft.dataset}")


def test_whisper_finetuner():
    # Set up test directory
    id = "wft-test-model"
    org = "JacobLinCool"
    merged_model_path = os.path.join(id, "merged_model")
    shutil.rmtree(id, ignore_errors=True)
    os.makedirs(id, exist_ok=True)

    try:
        ft = WhisperFineTuner(id, org)
        ft.training_args.num_train_epochs = 3
        # Initialize WhisperFineTuner
        ft = (
            ft.set_baseline("openai/whisper-tiny", language="en", task="transcribe")
            .prepare_dataset(
                "hf-internal-testing/librispeech_asr_dummy",
                src_transcription_column="text",
                src_train_split="validation[:10]",
                src_test_split="validation[10:20]",
            )
            .then(filter4)
            .set_metric("wer")
            .train()
            .merge_and_save(merged_model_path)
        )

        # Check if the merged model files exist
        assert os.path.exists(
            os.path.join(merged_model_path, "model.safetensors")
        ), "Merged model file not found"
        assert os.path.exists(
            os.path.join(merged_model_path, "config.json")
        ), "Model config file not found"

        print("WhisperFineTuner test completed successfully!")

    finally:
        # Clean up test directory
        # shutil.rmtree(id, ignore_errors=True)
        pass


if __name__ == "__main__":
    test_whisper_finetuner()
