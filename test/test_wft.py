import os
import shutil
from wft import WhisperFineTuner


def test_whisper_finetuner():
    # Set up test directory
    id = "test-model"

    try:
        ft = WhisperFineTuner(id)
        ft.training_args.num_train_epochs = 3

        shutil.rmtree(ft.dir, ignore_errors=True)
        merged_model_path = os.path.join(ft.dir, "merged_model")

        ft = (
            ft.set_baseline("openai/whisper-tiny", language="en", task="transcribe")
            .prepare_dataset(
                "hf-internal-testing/librispeech_asr_dummy",
                src_transcription_column="text",
                src_train_split="validation[:4]",
                src_test_split="validation[4:8]",
            )
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
        pass


if __name__ == "__main__":
    test_whisper_finetuner()
