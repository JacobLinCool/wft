from transformers import TrainerCallback
from transformers.integrations import TensorBoardCallback, rewrite_logs
from transformers.integrations.integration_utils import logger
from transformers.trainer_callback import ProgressCallback, TrainerControl, TrainerState
from transformers.trainer_pt_utils import IterableDatasetShard
from transformers.training_args import TrainingArguments
from torch.utils.data import IterableDataset


class WFTTensorBoardCallback(TensorBoardCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero:
            return

        if self.tb_writer is None:
            self._init_summary_writer(args)

        if self.tb_writer is not None:
            logs = rewrite_logs(logs)
            for k, v in logs.items():
                if isinstance(v, (int, float)):
                    self.tb_writer.add_scalar(k, v, state.global_step)
                elif isinstance(v, str):
                    self.tb_writer.add_text(k, v, state.global_step)
                else:
                    logger.warning(
                        "Trainer is attempting to log a value of "
                        f'"{v}" of type {type(v)} for key "{k}" as a scalar. '
                        "This invocation of Tensorboard's writer.add_scalar() "
                        "is incorrect so we dropped this attribute."
                    )
            self.tb_writer.flush()


class WFTProgressCallback(ProgressCallback):
    def __init__(self, max_str_len=100):
        super().__init__()
        self.max_str_len = max_str_len

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_world_process_zero and self.training_bar is not None:
            # make a shallow copy of logs so we can mutate the fields copied
            # but avoid doing any value pickling.
            shallow_logs = {}
            for k, v in logs.items():
                if isinstance(v, str) and len(v) > self.max_str_len:
                    shallow_logs[k] = f"[String too long to display, length: {len(v)}]"
                else:
                    shallow_logs[k] = v
            _ = shallow_logs.pop("total_flos", None)
            # round numbers so that it looks better in console
            if "epoch" in shallow_logs:
                shallow_logs["epoch"] = round(shallow_logs["epoch"], 2)
            self.training_bar.write(str(shallow_logs))


class ShuffleCallback(TrainerCallback):
    def on_epoch_begin(self, args, state, control, train_dataloader, **kwargs):
        if isinstance(train_dataloader.dataset, IterableDatasetShard):
            pass  # set_epoch() is handled by the Trainer
        elif isinstance(train_dataloader.dataset, IterableDataset):
            train_dataloader.dataset.set_epoch(train_dataloader.dataset._epoch + 1)


class PushCallback(TrainerCallback):
    def __init__(self, ft):
        super().__init__()
        self.ft = ft

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if state.is_world_process_zero and args.push_to_hub:
            self.ft.push_to_hub()
