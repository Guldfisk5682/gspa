import torch
from transformers import Trainer
import numpy as np
import random
import math


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
    def __init__(self, *args, lambda_max=0.5, align_warmup_epochs=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.lambda_max = lambda_max
        self.align_warmup_epochs = align_warmup_epochs  # 前几个 epoch λ=0

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        # 暖启动步骤计算
        if self.state.max_steps > 0 and self.args.num_train_epochs > 0:
            steps_per_epoch = self.state.max_steps / self.args.num_train_epochs
        else:
            steps_per_epoch = 0
        warmup_steps = int(self.align_warmup_epochs * steps_per_epoch)

        # 若处于暖启动阶段：λ=0
        if self.state.global_step < warmup_steps:
            lambda_p = 0.0
        else:
            # 重新规范化进度 p' 到 [0,1]
            if self.state.max_steps > warmup_steps and self.state.max_steps > 0:
                p_prime = (self.state.global_step - warmup_steps) / (
                    self.state.max_steps - warmup_steps
                )
                p_prime = max(0.0, min(1.0, p_prime))
            else:
                p_prime = 0.0
            # 渐进式 lambda：从 0 逐渐增加到 lambda_max
            # 公式：λ_p = (2 * λ_max) / (1 + exp(-10 * p')) - λ_max
            lambda_p = (2.0 * self.lambda_max) / (
                1.0 + math.exp(-10.0 * p_prime)
            ) - self.lambda_max

        # 将 lambda 传递给模型
        model._align_weight = lambda_p

        outputs = model(**inputs)
        loss = outputs.loss

        # 记录各项损失和当前 lambda
        if self.state.global_step % self.args.logging_steps == 0:
            logs = {
                "loss_task": (
                    outputs.loss_task.item() if outputs.loss_task is not None else 0.0
                ),
                "loss_align": (
                    outputs.loss_align.item() if outputs.loss_align is not None else 0.0
                ),
                "lambda_align": lambda_p,  # 记录当前对齐损失权重
                "warmup_steps": warmup_steps,
                "logits": (outputs.logits if outputs.logits is not None else 0.0),
                "gate_mean": (
                    outputs.gate_mean if outputs.gate_mean is not None else 0.0
                ),
                "gate_std": outputs.gate_std if outputs.gate_std is not None else 0.0,
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
