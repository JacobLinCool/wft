import shutil
from wft import WhisperFineTuner


def test_whisper_finetuner():
    id = "wft-test-model"
    org = "JacobLinCool"

    try:
        ft = WhisperFineTuner(id, org)

        shutil.rmtree(ft.dir, ignore_errors=True)

        ft = (
            ft.set_baseline("openai/whisper-tiny", language="en", task="transcribe")
            .prepare_dataset(
                "hf-internal-testing/librispeech_asr_dummy",
                src_transcription_column="text",
                src_train_split="validation",
                src_test_split="validation[:10]",
            )
            .set_metric("wer")
            .set_steps(100, 10, 10)
            .train()
            .merge_and_push()
        )

        print("WhisperFineTuner test completed successfully!")

    finally:
        pass


if __name__ == "__main__":
    test_whisper_finetuner()
