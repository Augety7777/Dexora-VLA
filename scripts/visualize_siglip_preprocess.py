#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import SiglipImageProcessor


def expand2square(pil_img: Image.Image, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def save_tensor_as_image(t: torch.Tensor, mean, std, out_path: str):
    """
    t: Float tensor in shape [3, H, W], normalized by (t - mean)/std expected by SigLIP.
    This function denormalizes to [0,1], converts to uint8 and saves.
    """
    device = t.device
    mean = torch.as_tensor(mean, dtype=t.dtype, device=device).view(3, 1, 1)
    std = torch.as_tensor(std, dtype=t.dtype, device=device).view(3, 1, 1)
    img = (t * std + mean).clamp(0.0, 1.0)  # [3, H, W]
    img_uint8 = (img.permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8)  # [H, W, 3]
    Image.fromarray(img_uint8).save(out_path)


def main():
    parser = argparse.ArgumentParser(description="Visualize SigLIP (384) preprocessing on an image.")
    parser.add_argument(
        "--image_path",
        type=str,
        default="/baai-cwm-vepfs/cwm/zongzheng.zhang/Dex-RDT/data/ours/final/action190/episode_7/camera_head/frame_000031.jpg",
        help="Path to the input RGB image.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outs/siglip_preview",
        help="Directory to save visualization outputs.",
    )
    parser.add_argument(
        "--do_pad",
        action="store_true",
        help="Pad non-square image to square (matching training's image_aspect_ratio=pad).",
    )
    parser.add_argument(
        "--vision_tower",
        type=str,
        default="google/siglip-so400m-patch14-384",
        help="SigLIP vision tower name (determines preprocessing rules).",
    )
    args = parser.parse_args()

    image_path = Path(args.image_path)
    assert image_path.exists(), f"Image not found: {image_path}"
    os.makedirs(args.output_dir, exist_ok=True)

    # Load image and processor
    img = Image.open(str(image_path)).convert("RGB")
    processor = SiglipImageProcessor.from_pretrained(args.vision_tower)

    # 1) Optional: pad to square with background color = processor.image_mean (like training dataset)
    padded_img = img
    if args.do_pad:
        bg_color = tuple(int(x * 255) for x in processor.image_mean)
        padded_img = expand2square(img, bg_color)
        padded_out = os.path.join(args.output_dir, f"{image_path.stem}_padded.jpg")
        padded_img.save(padded_out)
        print(f"Saved padded image: {padded_out} (size={padded_img.size})")

    # 2) SigLIP preprocess to pixel_values (normalized tensor, typically 384x384)
    inputs = processor.preprocess(padded_img, return_tensors="pt")
    pixel_values = inputs["pixel_values"][0]  # [3, H, W], float
    H, W = pixel_values.shape[-2], pixel_values.shape[-1]
    print(f"SigLIP pixel_values shape: {tuple(pixel_values.shape)}  (HxW = {H}x{W})")
    print(
        f"SigLIP normalization: mean={processor.image_mean}, std={processor.image_std}; "
        f"value range (min,max)=({float(pixel_values.min()):.4f}, {float(pixel_values.max()):.4f})"
    )

    # 3) Save the denormalized visualization of the preprocessed tensor
    vis_out = os.path.join(args.output_dir, f"{image_path.stem}_preprocessed_{H}x{W}.jpg")
    save_tensor_as_image(pixel_values, processor.image_mean, processor.image_std, vis_out)
    print(f"Saved denormalized preprocessed image: {vis_out}")


if __name__ == "__main__":
    main()


