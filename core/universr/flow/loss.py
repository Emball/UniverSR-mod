import torch


def band_weight_vector(bands, gen_start_bin, num_bins, bin_hz, default=1.0):
    """Per-bin weights over the generated region. Entries are {lo_hz, hi_hz, weight} (later ones override) or
    {shape: gaussian, center_hz, sigma_hz, gain}, which adds gain*exp(-0.5*((f-center)/sigma)^2) on top of the current weight (Apollo semantics)."""
    w = torch.full((num_bins,), float(default))
    centers = (torch.arange(num_bins, dtype=torch.float32) + gen_start_bin) * bin_hz
    for b in bands or []:
        if b.get("shape") == "gaussian":
            z = (centers - float(b["center_hz"])) / float(b["sigma_hz"])
            w = w + float(b["gain"]) * torch.exp(-0.5 * z * z)
            continue
        lo, hi = float(b.get("lo_hz", 0.0)), float(b.get("hi_hz", float("inf")))
        w[(centers >= lo) & (centers < hi)] = float(b["weight"])
    return w.reshape(1, 1, num_bins, 1)


def flow_matching_loss(predicted_vf: torch.Tensor, target_vf: torch.Tensor, weight: torch.Tensor = None) -> torch.Tensor:
    """L2 between estimated and target vector field, optionally weighted per frequency bin."""
    err = (predicted_vf.float() - target_vf.float()).square()
    if weight is not None:
        err = err * weight.to(err.device)
    return err.mean()
