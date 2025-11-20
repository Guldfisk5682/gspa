from transformers import AutoImageProcessor
from datasets import load_dataset
from torch.utils.data import Dataset
import random
import torch


class OfficeHomeSource(Dataset):
    """
    源域数据（有标签）
    """

    def __init__(self, domain, model_name):
        self.domain = domain
        self.image_processor = AutoImageProcessor.from_pretrained(model_name)

        self.dataset = load_dataset("flwrlabs/office-home", split="train").filter(
            lambda example: example["domain"] == domain
        )

        self.classes = self.dataset.features["label"].names

    def __len__(self):
        return len(self.dataset)

    def get_classes(self):
        return self.classes

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"]
        label = item["label"]

        if image.mode != "RGB":
            image = image.convert("RGB")

        pixel_values = self.image_processor(
            image, return_tensors="pt"
        ).pixel_values.squeeze(
            0
        )  # (,3,224,224)

        return {
            "pixel_values": pixel_values,
            "label": torch.tensor(label, dtype=torch.long),
        }


class OfficeHomeTarget(Dataset):
    """目标域数据集（无标签，域均衡采样）"""

    def __init__(self, source_domain, model_name):
        self.image_processor = AutoImageProcessor.from_pretrained(model_name)

        full_dataset = load_dataset("flwrlabs/office-home", split="train")

        all_domains = full_dataset.unique("domain")
        self.target_domains = [d for d in all_domains if d != source_domain]

        print("Processing target domains:", self.target_domains)

        # 高效筛选：遍历一次获取所有需要的索引，避免多次全量 filter
        self.target_datasets = {
            domain: full_dataset.filter(lambda x: x["domain"] == domain)
            for domain in self.target_domains
        }

        self.length = max(len(ds) for ds in self.target_datasets.values()) * len(
            self.target_domains
        )

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # 域均衡策略：轮流从三个域采样
        num_domains = len(self.target_domains)
        domain_idx = idx % num_domains
        domain = self.target_domains[domain_idx]
        dataset = self.target_datasets[domain]

        sample_idx = (idx // num_domains) % len(dataset)
        item = dataset[sample_idx]

        image = item["image"]
        if image.mode != "RGB":
            image = image.convert("RGB")

        pixel_values = self.image_processor(
            image, return_tensors="pt"
        ).pixel_values.squeeze(0)

        return {"pixel_values": pixel_values}


class OfficeHomeEval(Dataset):
    """评估数据集（单个目标域）"""

    def __init__(self, target_domain, model_name):
        self.image_processor = AutoImageProcessor.from_pretrained(model_name)

        self.dataset = load_dataset("flwrlabs/office-home", split="train").filter(
            lambda example: example["domain"] == target_domain
        )

        self.domain = target_domain

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"]
        label = item["label"]

        if image.mode != "RGB":
            image = image.convert("RGB")

        pixel_values = self.image_processor(
            image, return_tensors="pt"
        ).pixel_values.squeeze(0)

        return {
            "pixel_values": pixel_values,
            "label": torch.tensor(label, dtype=torch.long),
            "domain": self.domain,
        }


class PairedDataset(Dataset):
    def __init__(self, source_dataset, target_dataset):
        self.source = source_dataset
        self.target = target_dataset
        self.length = len(source_dataset)
        self.target_length = len(target_dataset)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        source_sample = self.source[idx]
        target_idx = random.randint(0, self.target_length - 1)
        target_sample = self.target[target_idx]
        return {
            "pixel_values_S": source_sample["pixel_values"],
            "pixel_values_T": target_sample["pixel_values"],
            "labels": source_sample["label"],
        }


class DataCollatorTrain:
    """训练用 Collator（同时处理源域和目标域）"""

    def __call__(self, features):
        pixel_values_S = torch.stack([f["pixel_values_S"] for f in features])
        pixel_values_T = torch.stack([f["pixel_values_T"] for f in features])
        labels = torch.stack([f["labels"] for f in features])

        return {
            "pixel_values_S": pixel_values_S,
            "pixel_values_T": pixel_values_T,
            "labels": labels,
        }


class MultiDomainEvalDataset(Dataset):
    """多目标域评估数据集（拼接多个 OfficeHomeEval）"""

    def __init__(self, source_domain, model_name):
        """
        Args:
            source_domain: 源域名称，用于排除
            model_name: 模型名称
        """
        full_dataset = load_dataset("flwrlabs/office-home", split="train")
        all_domains = full_dataset.unique("domain")
        self.target_domains = [d for d in all_domains if d != source_domain]

        # 为每个目标域创建 OfficeHomeEval
        self.domain_datasets = [
            OfficeHomeEval(domain, model_name) for domain in self.target_domains
        ]

        # 计算累积长度，用于索引映射
        self.cumulative_sizes = [0]
        for ds in self.domain_datasets:
            self.cumulative_sizes.append(self.cumulative_sizes[-1] + len(ds))

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        # 找到对应的域和域内索引
        for i, (start, end) in enumerate(
            zip(self.cumulative_sizes[:-1], self.cumulative_sizes[1:])
        ):
            if start <= idx < end:
                domain_idx = i
                local_idx = idx - start
                return self.domain_datasets[domain_idx][local_idx]
        raise IndexError(f"Index {idx} out of range")


class DataCollatorEval:
    """评估用 Collator"""

    def __call__(self, features):
        pixel_values = torch.stack([f["pixel_values"] for f in features])
        labels = torch.stack([f["label"] for f in features])
        domains = [f["domain"] for f in features]

        # 将域名映射为整数（用于后续分组）
        domain_to_id = {"Art": 0, "Clipart": 1, "Product": 2, "Real World": 3}
        domain_ids = torch.tensor([domain_to_id[d] for d in domains], dtype=torch.long)

        return {
            "pixel_values": pixel_values,
            "labels": labels,
            "domain_ids": domain_ids,
        }
