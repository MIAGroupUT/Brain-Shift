import torch
from torch.utils.checkpoint import checkpoint


# Helper function that calculates for one bin, the histogram
def histogram_helper(x_flat, last_bin, current_bin, next_bin, histogram, i):
    histogram_new = histogram.clone()
    add1 = (((current_bin > x_flat) & (x_flat >= last_bin)) * (x_flat - last_bin)).sum(dim=-1)
    add2 = (((next_bin > x_flat) & (x_flat >= current_bin)) * (next_bin - x_flat)).sum(dim=-1)
    histogram_new[:, :, i] = histogram_new[:, :, i] + add1 + add2
    return histogram_new


# A differentiable histogram implementation with torch checkpoints for memory reasons
def differentiable_histogram(x, n_bins=90):
    if len(x.shape) < 4:
        raise ValueError("The input must have a batch and channel dimension.")
    batch, channels = x.shape[:2]

    # Sometimes 1.0 != 1.0 based on dtype. This is a hack.
    assert x.max() <= 1.0001
    assert x.min() >= -0.0001

    histogram = torch.zeros(batch, channels, n_bins, device=x.device)
    delta = 1.0 / n_bins

    bins = torch.linspace(0, 1, n_bins, device=x.device)

    # Flatten x for easier masking
    x_flat = x.reshape(batch, channels, -1)

    # First and last bin excluded
    for i in (range(1, n_bins - 1)):
        current_bin = bins[i]
        last_bin = bins[i - 1]
        next_bin = bins[i + 1]

        # This result is also checkpointed since there are a lot of intermediate products within this function.
        histogram = checkpoint(histogram_helper, x_flat, last_bin, current_bin, next_bin, histogram, i).clone()

    histogram.div_(delta)

    # Normalize on size, so that we can treat this as a distribution
    return histogram / torch.sum(histogram)
