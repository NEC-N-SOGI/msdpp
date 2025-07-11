import numpy as np
import torch
from matplotlib import pyplot as plt


def hours_on_circle_hist(
    hours: torch.Tensor, n_width: int = 24, bottom: int = 8, max_height: int = 100
) -> None:
    theta = torch.tensor(list(range(23, -1, -1))) / 24 * 2 * torch.pi

    cnt_hours = [(hours == i).sum().item() for i in range(24)]
    norm_cnt_horus = torch.tensor(cnt_hours) / sum(cnt_hours)

    radii = max_height * norm_cnt_horus
    width = (2 * np.pi) / n_width / 2

    ax = plt.subplot(111, polar=True)
    bars = ax.bar(theta, radii, width=width, bottom=bottom)

    # Use custom colors and opacity
    for r, bar in zip(radii, bars, strict=False):
        bar.set_facecolor(plt.cm.jet(r / 10.0))  # type: ignore[attr-defined]
        bar.set_alpha(0.9)

    # change the angle labels to hours
    ax.set_xticks(theta)
    ax.set_xticklabels(list(range(24)))  # type: ignore[arg-type]

    # remove the ylabels
    # set 0 am to the top
    ax.set_theta_offset(np.pi / 2 + np.pi / 12)  # type: ignore[attr-defined]

    plt.show()


def datetime_polar_hist(
    hours: torch.Tensor, minutes: torch.Tensor, bins: int = 40
) -> None:
    time_embeds = hours * 60 + minutes
    max_embeds = 24 * 60

    bin_step = max_embeds / bins
    n_data_in_bins = [
        ((time_embeds >= i * bin_step) & (time_embeds < (i + 1) * bin_step)).sum()
        / len(time_embeds)
        * 100
        for i in range(bins)
    ]

    # flip n_data_in_bins
    n_data_in_bins.reverse()
    theta = torch.linspace(0, 2 * torch.pi, bins + 1)[:-1]

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, polar=True)
    ax.bar(theta, n_data_in_bins, width=torch.pi / bins, bottom=8)

    for r, bar in zip(n_data_in_bins, ax.patches, strict=False):
        bar.set_facecolor(plt.cm.jet(r / 10.0))  # type: ignore[attr-defined]
        bar.set_alpha(0.9)

    ax.set_xticks(torch.tensor(list(range(23, -1, -1))) / 24 * 2 * torch.pi)
    ax.set_xticklabels(list(range(24)))  # type: ignore[arg-type]
    ax.set_theta_offset(torch.pi / 2 + torch.pi / 12)  # type: ignore[attr-defined]
    plt.show()
