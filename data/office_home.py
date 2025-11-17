from transformers import AutoProcessor
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
        self.processor = AutoProcessor.from_pretrained(model_name)

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

        pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze(
            0
        )  # (,3,224,224)

        return {
            "pixel_values": pixel_values,
            "label": torch.tensor(label, dtype=torch.long),
        }


class OfficeHomeTarget(Dataset):
    """目标域数据集（无标签，域均衡采样）"""

    def __init__(self, source_domain, model_name):
        self.processor = AutoProcessor.from_pretrained(model_name)

        full_dataset = load_dataset("flwrlabs/office-home", split="train")

        # 手动分组：一次遍历，按域分类
        self.target_datasets = {}
        for item in full_dataset:
            domain = item["domain"]
            if domain != source_domain:  # 排除源域
                if domain not in self.target_datasets:
                    self.target_datasets[domain] = []
                self.target_datasets[domain].append(item)

        self.target_domains = list(self.target_datasets.keys())  # 自动得到三个域名
        self.length = max(len(items) for items in self.target_datasets.values())

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # 域均衡策略：轮流从三个域采样
        domain = self.target_domains[idx % len(self.target_domains)]
        dataset = self.target_datasets[domain]

        # 如果 idx 超出该域长度，循环采样
        sample_idx = idx % len(dataset)
        item = dataset[sample_idx]

        image = item["image"]
        if image.mode != "RGB":
            image = image.convert("RGB")

        pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze(
            0
        )

        return {"pixel_values": pixel_values}


class OfficeHomeEval(Dataset):
    """评估数据集（单个目标域）"""

    def __init__(self, target_domain, model_name):
        self.processor = AutoProcessor.from_pretrained(model_name)

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

        pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze(
            0
        )

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


class DataCollatorEval:
    """评估用 Collator"""

    def __call__(self, features):
        pixel_values = torch.stack([f["pixel_values"] for f in features])
        labels = torch.stack([f["label"] for f in features])
        domains = [f["domain"] for f in features]

        return {"pixel_values": pixel_values, "labels": labels, "domains": domains}
