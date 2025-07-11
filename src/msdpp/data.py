import torch


def datetime_embeds(hours: torch.Tensor, minutes: torch.Tensor) -> torch.Tensor:
    time_embeds = hours * 60 + minutes

    max_embeds = 23 * 60 + 59

    return torch.stack(
        [
            torch.sin(time_embeds / max_embeds * 2 * torch.pi),
            torch.cos(time_embeds / max_embeds * 2 * torch.pi),
        ],
        1,
    )
