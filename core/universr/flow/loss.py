import torch


def band_weight_vector(bands, gen_start_bin, num_bins, bin_hz, default=1.0):
    """Per-bin weights over the generated region from [{lo_hz, hi_hz, weight}, ...]; later bands override earlier ones."""
    w = torch.full((num_bins,), float(default))
    centers = (torch.arange(num_bins, dtype=torch.float32) + gen_start_bin) * bin_hz
    for b in bands or []:
        lo, hi = float(b.get("lo_hz", 0.0)), float(b.get("hi_hz", float("inf")))
        w[(centers >= lo) & (centers < hi)] = float(b["weight"])
    return w.reshape(1, 1, num_bins, 1)


def flow_matching_loss(predicted_vf: torch.Tensor, target_vf: torch.Tensor, weight: torch.Tensor = None) -> torch.Tensor:
    """L2 between estimated and target vector field, optionally weighted per frequency bin."""
    err = (predicted_vf.float() - target_vf.float()).square()
    if weight is not None:
        err = err * weight.to(err.device)
    return err.mean()
