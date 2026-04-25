import torch
import torch.nn.functional as F
import torchio as tio


def random_crop(
    volume: torch.Tensor,
    min_scale: float = 0.5,
    max_scale: float = 1.0,
) -> torch.Tensor:
    """
    Randomly crop a sub-volume of `volume` and resize it
    back to the original shape using trilinear interpolation.
    Input:  4D torch.Tensor (C, D, H, W) on any device.
    Output: same shape & same device.
    """
    C, D, H, W = volume.shape
    device = volume.device

    # pick random scale
    scale = torch.empty(1, device=device).uniform_(min_scale, max_scale).item()
    new_D = max(1, int(D * scale))
    new_H = max(1, int(H * scale))
    new_W = max(1, int(W * scale))

    # pick random corner
    z0 = torch.randint(0, D - new_D + 1, (), device=device).item()
    y0 = torch.randint(0, H - new_H + 1, (), device=device).item()
    x0 = torch.randint(0, W - new_W + 1, (), device=device).item()

    # crop
    cropped = volume[
        :,
        z0 : z0 + new_D,
        y0 : y0 + new_H,
        x0 : x0 + new_W,
    ]

    # resize back to (D, H, W) with trilinear
    cropped = cropped.unsqueeze(0)  # (1, C, new_D, new_H, new_W)
    resized = F.interpolate(
        cropped,
        size=(D, H, W),
        mode='trilinear',
        align_corners=False,
    )
    return resized.squeeze(0)  # (C, D, H, W)


def build_augmentation(cfg: dict) -> tio.Compose:
    """
    Build a torchio Compose where each transform
    takes & returns a torch.Tensor (4D: C, D, H, W).
    """
    transforms = []
    
    for name, params in cfg.items():
        if name == 'random_crop':
            # Capture params in closure
            min_s = params.get('min_scale', 0.5)
            max_s = params.get('max_scale', 1.0)
            transforms.append(
                tio.Lambda(
                    lambda x, ms=min_s, mx=max_s: random_crop(x, ms, mx),
                    p=params.get('p', 0.5),
                )
            )
        elif name == 'random_flip':
            transforms.append(
                tio.RandomFlip(axes=('LR',), p=params.get('p', 0.5))
            )
        elif name == 'random_affine':
            transforms.append(
                tio.RandomAffine(
                    scales=params.get('scales', 0),
                    degrees=params.get('degrees', 10),
                    translation=params.get('translation', 10),
                    p=params.get('p', 0.5),
                )
            )
        elif name == 'random_noise':
            transforms.append(
                tio.RandomNoise(
                    std=params.get('std', 0.05),
                    p=params.get('p', 0.2)
                )
            )
        else:
            print(f"Warning: Unknown transform '{name}', skipping")
    
    return tio.Compose(transforms) if transforms else None
