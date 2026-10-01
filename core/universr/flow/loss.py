import torch
import torch.nn.functional as F


def flow_matching_loss(predicted_vf: torch.Tensor, target_vf: torch.Tensor) -> torch.Tensor:
    """
    Flow matching loss; L2 loss between estimated and target vector field.
    """
    return F.mse_loss(predicted_vf, target_vf)
