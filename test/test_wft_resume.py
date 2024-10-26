import os
import shutil
from wft import WhisperFineTuner


def test_whisper_finetuner():
    # Set up test directory
    id = "test-resume-model"

    ft = WhisperFineTuner(id)
    shutil.rmtree(ft.dir, ignore_errors=True)
    merged_model_path = os.path.join(ft.dir, "merged_model")

    def train(n: int):
        ft = WhisperFineTuner(id)
        ft.training_args.num_train_epochs = n
        ft.set_baseline("openai/whisper-tiny", language="en", task="transcribe")
        ft.prepare_dataset(
            "hf-internal-testing/librispeech_asr_dummy",
            src_transcription_column="text",
            src_train_split="validation[:4]",
            src_test_split="validation[4:8]",
        )
        ft.set_metric("wer")
        ft.train(resume=True)
        ft.merge_and_save(merged_model_path)

    train(3)
    train(6)

    # Check if the merged model files exist
    assert os.path.exists(
        os.path.join(merged_model_path, "model.safetensors")
    ), "Merged model file not found"
    assert os.path.exists(
        os.path.join(merged_model_path, "config.json")
    ), "Model config file not found"

    print("WhisperFineTuner test completed successfully!")


if __name__ == "__main__":
    test_whisper_finetuner()
