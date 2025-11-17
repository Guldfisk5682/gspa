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
    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        outputs = model(**inputs)
        loss = outputs.loss

        # 记录各项损失
        if self.state.global_step % self.args.logging_steps == 0:
            logs = {
                "loss_task": (
                    outputs.loss_task.item() if outputs.loss_task is not None else 0.0
                ),
                "loss_align": (
                    outputs.loss_align.item() if outputs.loss_align is not None else 0.0
                ),
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
        metric_key_prefix="eval",
    ):
        model = self._wrap_model(self.model, training=False, dataloader=dataloader)
        model.eval()

        all_preds = []
        all_labels = []

        for inputs in dataloader:
            inputs = self._prepare_inputs(inputs)

            with torch.no_grad():
                # 测试时只传图像，不传目标域
                outputs = model(inputs["pixel_values"], inputs["pixel_values"])
                logits = outputs.logits

            preds = logits.argmax(dim=-1)
            all_preds.append(preds.cpu())
            all_labels.append(inputs["labels"].cpu())

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)

        accuracy = (all_preds == all_labels).float().mean().item()

        metrics = {f"{metric_key_prefix}_accuracy": accuracy}

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
