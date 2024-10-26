import os
import shutil
from wft import WhisperFineTuner


def test_whisper_finetuner():
    id = "wft-test-model"
    org = "JacobLinCool"

    try:
        ft = WhisperFineTuner(id, org)
        ft.training_args.num_train_epochs = 5

        shutil.rmtree(ft.dir, ignore_errors=True)

        ft = (
            ft.set_baseline("openai/whisper-tiny", language="en", task="transcribe")
            .prepare_dataset(
                "hf-internal-testing/librispeech_asr_dummy",
                src_transcription_column="text",
                src_train_split="validation[:30]",
                src_test_split="validation[30:60]",
            )
            .set_metric("wer")
            .train()
            .merge_and_push()
        )

        print("WhisperFineTuner test completed successfully!")

    finally:
        pass


if __name__ == "__main__":
    test_whisper_finetuner()
