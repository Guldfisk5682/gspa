from transformers.modeling_outputs import ModelOutput
from typing import Optional
import torch

class GSPAOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    loss_task: Optional[torch.FloatTensor] = None
    loss_align: Optional[torch.FloatTensor] = None
    gate_mean: Optional[float] = None
    gate_std: Optional[float] = None