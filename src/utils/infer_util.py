import os
import imageio
import rembg
import torch
import numpy as np
import PIL.Image
from PIL import Image
from typing import Any


def remove_background(image: PIL.Image.Image,
    rembg_session: Any = None,
    force: bool = False,
    **rembg_kwargs,
) -> PIL.Image.Image:
    do_remove = True
    if image.mode == "RGBA" and image.getextrema()[3][0] < 255:
        do_remove = False
    do_remove = do_remove or force
    if do_remove:
        image = rembg.remove(image, session=rembg_session, **rembg_kwargs,bgcolor=[255,255,255,0])
    return image


def resize_foreground(
    image: PIL.Image.Image,
    ratio: float,
) -> PIL.Image.Image:
    image = np.array(image)
    assert image.shape[-1] == 4
    alpha = np.where(image[..., 3] > 0)
    y1, y2, x1, x2 = (
        alpha[0].min(),
        alpha[0].max(),
        alpha[1].min(),
        alpha[1].max(),
    )
    # crop the foreground
    fg = image[y1:y2, x1:x2]
    # pad to square
    size = max(fg.shape[0], fg.shape[1])
    ph0, pw0 = (size - fg.shape[0]) // 2, (size - fg.shape[1]) // 2
    ph1, pw1 = size - fg.shape[0] - ph0, size - fg.shape[1] - pw0
    new_image = np.pad(
        fg,
        ((ph0, ph1), (pw0, pw1), (0, 0)),
        mode="constant",
        constant_values=((0, 0), (0, 0), (0, 0)),
    )

    # compute padding according to the ratio
    new_size = int(new_image.shape[0] / ratio)
    # pad to size, double side
    ph0, pw0 = (new_size - size) // 2, (new_size - size) // 2
    ph1, pw1 = new_size - size - ph0, new_size - size - pw0
    new_image = np.pad(
        new_image,
        ((ph0, ph1), (pw0, pw1), (0, 0)),
        mode="constant",
        constant_values=((0, 0), (0, 0), (0, 0)),
    )
    new_image = PIL.Image.fromarray(new_image)
    return new_image

def resize_foreground_torch(
    image: torch.Tensor,
    ratio: float,
) -> torch.Tensor:
    # Assert input shape and value range
    assert image.ndim == 3, "Input tensor must be 3-dimensional."
    assert image.shape[-1] ==4, "The last dimension (channel) must be 4."
    # assert torch.all(image >= 0) and torch.all(image <= 1), "Pixel values must be in the range [0, 1]."
    

    alpha = torch.where(image[..., 3] > 0)
    y1, y2, x1, x2 = (
        alpha[0].min(),
        alpha[0].max(),
        alpha[1].min(),
        alpha[1].max(),
    )
    
    # Crop the foreground
    fg = image[y1:y2, x1:x2]
    import kiui
    # kiui.vis.plot_image(fg)

    # pad to square
    size = max(fg.shape[0], fg.shape[1])
    ph0, pw0 = (size - fg.shape[0]) // 2, (size - fg.shape[1]) // 2
    ph1, pw1 = size - fg.shape[0] - ph0, size - fg.shape[1] - pw0
    new_image = torch.nn.functional.pad(
        fg.permute(2, 0, 1), # pad requires (C, H, W),
        ( pw0, pw1,ph0, ph1),
        mode="constant",
        value=0,
    )
    new_image = new_image.permute(1, 2, 0) # back to H,W,C
    # kiui.vis.plot_image(new_image)

    # Calculate new size
    new_size = int(new_image.shape[0] / ratio)

    # Pad
    ph0, pw0 = (new_size - size) // 2, (new_size - size) // 2
    ph1, pw1 = new_size - size - ph0, new_size - size - pw0

 
    new_image = torch.nn.functional.pad(
        new_image.permute(2, 0, 1), # pad requires (C, H, W)
        (ph0, ph1, pw0, pw1),
        mode='constant',
        value=0,
    )
    new_image = new_image.permute(1, 2, 0)
    # kiui.vis.plot_image(new_image)
    return new_image

def images_to_video(
    images: torch.Tensor, 
    output_path: str, 
    fps: int = 30,
) -> None:
    # images: (N, C, H, W)
    video_dir = os.path.dirname(output_path)
    video_name = os.path.basename(output_path)
    os.makedirs(video_dir, exist_ok=True)

    frames = []
    for i in range(len(images)):
        frame = (images[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        assert frame.shape[0] == images.shape[2] and frame.shape[1] == images.shape[3], \
            f"Frame shape mismatch: {frame.shape} vs {images.shape}"
        assert frame.min() >= 0 and frame.max() <= 255, \
            f"Frame value out of range: {frame.min()} ~ {frame.max()}"
        frames.append(frame)
    imageio.mimwrite(output_path, np.stack(frames), fps=fps, quality=10)


def save_video(
    frames: torch.Tensor,
    output_path: str,
    fps: int = 30,
) -> None:
    # images: (N, C, H, W)
    frames = [(frame.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8) for frame in frames]
    writer = imageio.get_writer(output_path, fps=fps)
    for frame in frames:
        writer.append_data(frame)
    writer.close()