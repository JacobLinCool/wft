from .finetuner import WhisperFineTuner

class DevicePreset:
    def GH200(ft: WhisperFineTuner, batch_size: int = 1024):
        ft.training_args.per_device_train_batch_size = 16
        ft.training_args.gradient_accumulation_steps = batch_size // 16
        ft.training_args.per_device_eval_batch_size = 16
        ft.training_args.gradient_checkpointing = False
        ft.training_args.dataloader_num_workers = 4

    def A40(ft: WhisperFineTuner, batch_size: int = 1024):
        ft.training_args.per_device_train_batch_size = 8
        ft.training_args.gradient_accumulation_steps = batch_size // 8
        ft.training_args.per_device_eval_batch_size = 8
        ft.training_args.gradient_checkpointing = False
        ft.training_args.dataloader_num_workers = 4

# ref: https://github.com/vasistalodagala/whisper-finetune#hyperparameter-tuning
class ModelPreset:
    def Large(ft: WhisperFineTuner):
        ft.training_args.learning_rate = 5e-5
    def Medium(ft: WhisperFineTuner):
        ft.training_args.learning_rate = 6.25e-5
    def Small(ft: WhisperFineTuner):
        ft.training_args.learning_rate = 1.25e-4
    def Base(ft: WhisperFineTuner):
        ft.training_args.learning_rate = 2.5e-4
    def Tiny(ft: WhisperFineTuner):
        ft.training_args.learning_rate = 3.75e-4
