import os
from transformers import TrainingArguments
from data.office_home import (
    OfficeHomeSource,
    OfficeHomeTarget,
    PairedDataset,
    MultiDomainEvalDataset,
    DataCollatorGSPA,
)
from model.gspa import ConditionalPromptLearner
from model.config import GSPAConfig
from utils.misc import set_seed, GSPATrainer
import torch
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

base_path = os.path.dirname(os.path.abspath(__file__))


def main():
    set_seed(42)

    # 配置
    model_name = "openai/clip-vit-base-patch16"
    source_domain = "Product"
    output_dir = os.path.join(base_path, "outputs", "product_to_others")
    log_dir = os.path.join(output_dir, "logs")

    # 训练数据集
    # 源域数据启用轻度增强
    train_source = OfficeHomeSource(source_domain, model_name, augment=True)
    train_target = OfficeHomeTarget(source_domain, model_name)
    train_dataset = PairedDataset(train_source, train_target)
    print("PairedDataset created successfully")

    # 评估数据集（多目标域）
    eval_dataset = MultiDomainEvalDataset(source_domain, model_name)
    print("MultiDomainEvalDataset created successfully")

    # 模型
    classnames = train_source.get_classes()
    print("class names ok")
    config = GSPAConfig(model_name_or_path=model_name, ctx=4, class_names=classnames)
    print("config ok")
    model = ConditionalPromptLearner(config)
    print("Model initialized successfully")

    # 训练参数
    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=log_dir,
        num_train_epochs=25,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        max_grad_norm=1.0,
        logging_steps=50,
        eval_strategy="epoch",  # 每个 epoch 评估一次
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,  # 训练结束时加载最佳模型
        metric_for_best_model="eval_acc_mean",  # 使用平均 accuracy 选择最佳模型
        greater_is_better=True,
        bf16=True,
        dataloader_num_workers=8,
        remove_unused_columns=False,
        dataloader_pin_memory=True,
        report_to="tensorboard",
        optim="adamw_torch",
        weight_decay=0.01,
    )

    # 设置分组学习率：gate 使用更小的学习率
    # 学习率分组：gate 最小，metanet 提升 10x，其他保持基准
    base_lr = 2e-5
    meta_lr_scale = 10.0  # 可调倍率

    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if "gate" in n and p.requires_grad
            ],
            "lr": 2e-6,  # gate 的学习率
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if "metanet" in n and p.requires_grad
            ],
            "lr": base_lr * meta_lr_scale,  # metanet 提升 10x
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if "gate" not in n and "metanet" not in n and p.requires_grad
            ],
            "lr": base_lr,  # 其他参数
        },
    ]

    # Trainer（已移除对齐动态 λ，保持简洁）
    trainer = GSPATrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorGSPA(),
        optimizers=(
            torch.optim.AdamW(optimizer_grouped_parameters, weight_decay=0.01),
            None,
        ),
    )

    # 训练
    print(f"Starting training: {source_domain} -> Others")
    print(f"Output directory: {output_dir}")
    print(f"Log directory: {log_dir}")
    trainer.train()
    # trainer.train(resume_from_checkpoint=True)

    trainer.save_model(os.path.join(output_dir, "best_model"))


if __name__ == "__main__":
    main()
