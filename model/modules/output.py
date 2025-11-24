from transformers.modeling_outputs import ModelOutput
from typing import Optional
from dataclasses import dataclass
import torch

@dataclass
class GSPAOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    loss_task: Optional[torch.FloatTensor] = None
    loss_align: Optional[torch.FloatTensor] = None
    gate_mean: Optional[torch.FloatTensor] = None
    gate_std: Optional[torch.FloatTensor] = None