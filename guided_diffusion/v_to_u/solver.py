import torch
import matplotlib.pyplot as plt
import deepwave
from deepwave import scalar

def receiver(v, device):
    v = torch.squeeze(v)

    # ny = 2301
    # nx = 751

    ny = v.shape[1]
    nx = v.shape[0]
    dx = 10

    n_shots = 5

    n_sources_per_shot = 1
    d_source = 14  # 20 * 4m = 80m
    first_source = 0  # 10 * 4m = 40m
    source_depth = 2  # 2 * 4m = 8m

    n_receivers_per_shot = 64
    d_receiver = 1  # 6 * 4m = 24m
    first_receiver = 0  # 0 * 4m = 0m
    receiver_depth = 2  # 2 * 4m = 8m

    freq = 15
    # nt = 750
    nt = 1000
    dt = 0.001
    peak_time = 1.5 / freq

    # source_locations
    source_locations = torch.zeros(n_shots, n_sources_per_shot, 2,
                                dtype=torch.long, device=device)
    source_locations[..., 1] = source_depth
    source_locations[:, 0, 0] = (torch.arange(n_shots) * d_source +
                                first_source)

    # receiver_locations
    receiver_locations = torch.zeros(n_shots, n_receivers_per_shot, 2,
                                    dtype=torch.long, device=device)
    receiver_locations[..., 1] = receiver_depth
    receiver_locations[:, :, 0] = (
        (torch.arange(n_receivers_per_shot) * d_receiver +
        first_receiver)
        .repeat(n_shots, 1)
    )

    # source_amplitudes
    source_amplitudes = (
        deepwave.wavelets.ricker(freq, nt, dt, peak_time)
        .repeat(n_shots, n_sources_per_shot, 1)
        .to(device)
    )

    out = scalar(v, dx, dt, source_amplitudes=source_amplitudes,
                source_locations=source_locations,
                receiver_locations=receiver_locations,
                accuracy=8,
                pml_freq=freq)

    receiver_amplitudes = out[-1]
    return receiver_amplitudes