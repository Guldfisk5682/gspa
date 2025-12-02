import torch
import os
from transformers import Trainer
from transformers import TrainerCallback
from transformers.trainer_utils import EvalLoopOutput
from contextlib import contextmanager
from contextlib import nullcontext
from timm.utils import ModelEmaV2
import numpy as np
import random


def set_seed(seed=42):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class EMACallBack(TrainerCallback):
    def __init__(self, ema_decay=0.999):
        super().__init__()
        self.decay = ema_decay
        self.ema_model = None

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        """
        训练开始时：初始化 EMA 模型影子
        """
        print(f"\n[EMA] Initializing EMA model with decay={self.decay}...")

        self.ema_model = ModelEmaV2(model, decay=self.decay)

        # 确保 EMA 模型在正确的设备上 (虽然 timm 会尝试处理，但双保险更好)
        self.ema_model.to(args.device)

    def on_step_end(self, args, state, control, model=None, **kwargs):
        """
        每一步训练结束：更新 EMA 参数
        """
        if self.ema_model is not None:
            # update 方法执行公式: shadow = decay * shadow + (1-decay) * new
            self.ema_model.update(model)

    @contextmanager
    def use_ema_weights(self, model):
        """
        上下文管理器：暂时把 EMA 的权重“借”给主模型
        用于 Evaluation 或 Inference
        """
        if self.ema_model is None:
            yield
            return

        # 1. 保存原始模型的参数 (Backup)
        # 我们存储 state_dict 到 CPU 以节省显存，因为 eval 期间显存可能吃紧
        store = {k: v.clone().cpu() for k, v in model.state_dict().items()}

        # 2. 将 EMA 参数覆盖到模型中 (Swap in)
        # 处理 DDP 模型：如果 model 是 DDP (有 .module)，我们需要把 EMA 权重加载到 model.module
        # 找到真正的 base model (unwrapped)
        base_model = model
        while hasattr(base_model, "module"):
            base_model = base_model.module

        # self.ema_model.module 也是 base model 结构
        # 直接加载到 base_model 中，这样可以避开 DDP 的 module. 前缀问题
        base_model.load_state_dict(self.ema_model.module.state_dict())

        print(f"\n[EMA] Swapped EMA weights for evaluation...")

        try:
            yield  # 这里控制权交回给 evaluation_loop 执行评估
        finally:
            # 3. 恢复原始参数 (Restore)
            # 无论评估是否报错，都必须保证恢复，否则继续训练就乱了
            model.load_state_dict(store)
            print(f"[EMA] Restored original weights for training.\n")


class GSPATrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._stored_metrics = {}

        # 查找并保存 EMACallBack 实例，以便在 evaluation_loop 中使用
        self.ema_callback = None
        for callback in self.callback_handler.callbacks:
            if isinstance(callback, EMACallBack):
                self.ema_callback = callback
                break

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        outputs = model(**inputs)
        total_loss = outputs.loss

        if model.training:
            metrics = {
                "train/loss": total_loss.detach().mean().item(),
                "train/loss_task": (
                    outputs.loss_task.detach().mean().item()
                    if hasattr(outputs, "loss_task")
                    else 0.0
                ),
                "train/loss_align": (
                    outputs.loss_align.detach().mean().item()
                    if hasattr(outputs, "loss_align")
                    else 0.0
                ),
                "train/gate_mean": (
                    outputs.gate_mean.detach().mean().item()
                    if hasattr(outputs, "gate_mean")
                    else 0.0
                ),
                "train/gate_std": (
                    outputs.gate_std.detach().mean().item()
                    if hasattr(outputs, "gate_std")
                    else 0.0
                ),
            }
            self._stored_metrics.update(metrics)

        return (total_loss, outputs) if return_outputs else total_loss

    def log(self, logs: dict[str, float], *args, **kwargs) -> None:
        """重写日志记录，添加自定义指标"""
        if self._stored_metrics:
            logs.update(self._stored_metrics)
        super().log(logs, *args, **kwargs)

    def save_model(self, output_dir=None, _internal_call=False):
        """
        重写 save_model，在保存原始权重的同时，额外保存 EMA 权重
        """
        if output_dir is None:
            output_dir = self.args.output_dir

        # 1. 保存原始模型 (用于 Resume Training)
        super().save_model(output_dir, _internal_call)

        # 2. 保存 EMA 模型 (用于 Inference)
        if self.ema_callback:
            with self.ema_callback.use_ema_weights(self.model):
                # 获取 unwrapped model
                model_to_save = self.model
                while hasattr(model_to_save, "module"):
                    model_to_save = model_to_save.module
                
                ema_path = os.path.join(output_dir, "pytorch_model_ema.bin")
                torch.save(model_to_save.state_dict(), ema_path)
                print(f"[EMA] Saved EMA weights to {ema_path}")

    def evaluation_loop(
        self,
        dataloader,
        description,
        prediction_loss_only=None,
        ignore_keys=None,
        metric_key_prefix="eval",
    ):
        """
        MTDA 专用评估循环 (DDP Compatible)
        """
        ema_context = (
            self.ema_callback.use_ema_weights(self.model)
            if self.ema_callback
            else nullcontext()
        )

        with ema_context:
            model = self._wrap_model(self.model, training=False, dataloader=dataloader)
            model.eval()

            all_preds = []
            all_labels = []
            all_domain_ids = []

            domain_names = [
                "Art",
                "Clipart",
                "Product",
                "Real World",
            ]  # 硬编码，软编码重写太麻烦

            for i, inputs in enumerate(dataloader):
                inputs = self._prepare_inputs(inputs)

                # 修复：直接获取 domain_ids，它已经是 tensor 且在 device 上
                domain_ids = inputs.pop("domain_ids")

                with torch.no_grad():

                    outputs = model(**inputs)
                    logits = outputs.logits
                    preds = torch.argmax(logits, dim=-1)

                gathered_preds = self.accelerator.gather_for_metrics(
                    preds
                )  # all_gather concatenate
                gathered_labels = self.accelerator.gather_for_metrics(inputs["labels"])
                gathered_domain_ids = self.accelerator.gather_for_metrics(domain_ids)

                all_preds.append(gathered_preds.cpu())
                all_labels.append(gathered_labels.cpu())
                all_domain_ids.append(gathered_domain_ids.cpu())

            all_preds = torch.cat(all_preds, dim=0)
            all_labels = torch.cat(all_labels, dim=0)
            all_domain_ids = torch.cat(all_domain_ids, dim=0)

            metrics = {}

            for id, name in enumerate(domain_names):
                domain_mask = all_domain_ids == id
                if domain_mask.any():
                    domain_preds = all_preds[domain_mask]
                    domain_labels = all_labels[domain_mask]
                    domain_acc = (domain_preds == domain_labels).float().mean().item()
                    metrics[f"{metric_key_prefix}_{name}_acc"] = domain_acc

            metrics[f"{metric_key_prefix}_acc_mean"] = (
                (all_preds == all_labels).float().mean().item()
            )

            return EvalLoopOutput(
                predictions=all_preds,
                label_ids=all_labels,
                metrics=metrics,
                num_samples=len(all_labels),
            )
