import torch
from transformers import Trainer
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


class GSPATrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        outputs = model(**inputs)
        loss = outputs.loss

        if self.state.global_step % self.args.logging_steps == 0:
            def _reduce(val):
                if val is None:
                    return 0.0
                if isinstance(val, torch.Tensor):
                    # If gathered from DataParallel, may have shape [n_devices]
                    if val.numel() == 1:
                        return val.item()
                    else:
                        return val.mean().item()
                # Fallback for python floats
                try:
                    return float(val)
                except Exception:
                    return 0.0

            logs = {
                "loss_task": _reduce(getattr(outputs, "loss_task", None)),
                "loss_align": _reduce(getattr(outputs, "loss_align", None)),
                "gate_mean": _reduce(getattr(outputs, "gate_mean", None)),
                "gate_std": _reduce(getattr(outputs, "gate_std", None)),
            }
            self.log(logs)

        return (loss, outputs) if return_outputs else loss

    def evaluation_loop(
        self,
        dataloader,
        description,
        prediction_loss_only=None,
        ignore_keys=None,
        metric_key_prefix="eval",
    ):
        """重写评估循环，支持多域 MTDA 评估"""
        model = self._wrap_model(self.model, training=False, dataloader=dataloader)
        model.eval()

        all_preds = []
        all_labels = []
        all_domain_ids = []

        for inputs in dataloader:
            inputs = self._prepare_inputs(inputs)

            with torch.no_grad():
                # 评估模式：只传 pixel_values（使用 infer 方法）
                outputs = model(pixel_values=inputs["pixel_values"])
                logits = outputs.logits

            preds = logits.argmax(dim=-1)
            all_preds.append(preds.cpu())
            all_labels.append(inputs["labels"].cpu())
            all_domain_ids.append(inputs["domain_ids"].cpu())

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        all_domain_ids = torch.cat(all_domain_ids)

        # 按域计算 accuracy
        domain_names = ["Art", "Clipart", "Product", "Real World"]
        metrics = {}
        all_correct = 0
        all_total = 0

        for domain_id, domain_name in enumerate(domain_names):
            mask = all_domain_ids == domain_id
            if mask.sum() == 0:
                continue
            domain_preds = all_preds[mask]
            domain_labels = all_labels[mask]
            correct = (domain_preds == domain_labels).sum().item()
            total = len(domain_labels)
            acc = correct / total
            metrics[f"{metric_key_prefix}_acc_{domain_name}"] = acc
            all_correct += correct
            all_total += total

        # 计算平均 accuracy
        metrics[f"{metric_key_prefix}_acc_mean"] = (
            all_correct / all_total if all_total > 0 else 0.0
        )

        return type(
            "EvalLoopOutput",
            (),
            {
                "predictions": all_preds.numpy(),
                "label_ids": all_labels.numpy(),
                "metrics": metrics,
                "num_samples": len(all_labels),
            },
        )()
