#python -u
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

base = Path(__file__).resolve().parents[0]

def is_green_pixel(rgb: np.ndarray, *, g_min: int, delta_rg: int, delta_bg: int) -> np.ndarray:
    """Return boolean mask of pixels considered 'green'.

    A pixel is green if:
      - G >= g_min
      - G - R >= delta_rg
      - G - B >= delta_bg

    rgb: uint8 array with shape (..., 3)
    """
    r = rgb[..., 0].astype(np.int16)
    g = rgb[..., 1].astype(np.int16)
    b = rgb[..., 2].astype(np.int16)
    return (g >= g_min) & ((g - r) >= delta_rg) & ((g - b) >= delta_bg)


def _dominant_color_from_ref_image(ref_path: Path) -> np.ndarray:
    """Extract a representative RGB color from a reference image.

    If the image has an alpha channel, transparent pixels are ignored.
    Returns uint8 RGB array of shape (3,).
    """
    img = Image.open(ref_path)
    arr = np.asarray(img)

    if arr.ndim != 3 or arr.shape[2] not in (3, 4):
        raise ValueError(f"Unsupported reference image mode/shape: {ref_path} shape={arr.shape}")

    if arr.shape[2] == 4:
        rgb = arr[..., :3]
        alpha = arr[..., 3]
        mask = alpha > 0
        if not np.any(mask):
            # Fallback: treat as opaque
            mask = np.ones(alpha.shape, dtype=bool)
        pixels = rgb[mask]
    else:
        pixels = arr.reshape(-1, 3)

    # Use median to be robust to highlights/edges.
    color = np.median(pixels.astype(np.float32), axis=0)
    return np.clip(np.rint(color), 0, 255).astype(np.uint8)


def _match_any_color(rgb: np.ndarray, colors: np.ndarray, *, max_dist: float) -> np.ndarray:
    """Return boolean mask of pixels whose RGB is close to any color in colors.

    rgb: uint8 array [H,W,3]
    colors: uint8 array [K,3]
    max_dist: Euclidean distance threshold in RGB space
    """
    if colors.size == 0:
        return np.zeros(rgb.shape[:2], dtype=bool)
    pix = rgb.astype(np.int16)
    cols = colors.astype(np.int16)
    # Compute min distance to any reference color.
    # dist^2 = sum((pix - col)^2)
    # Broadcasting: [H,W,1,3] - [1,1,K,3] => [H,W,K,3]
    diff = pix[..., None, :] - cols[None, None, :, :]
    dist2 = np.sum(diff * diff, axis=-1)
    min_dist2 = np.min(dist2, axis=-1)
    return min_dist2 <= (max_dist * max_dist)


def image_to_green_majority_matrix(
    image_path: Path,
    *,
    out_size: int = 128,
    green_ratio_threshold: float = 0.5,
    g_min: int = 80,
    delta_rg: int = 20,
    delta_bg: int = 20,
    zero_ref_paths: list[Path] | None = None,
    zero_ratio_threshold: float = 0.5,
    zero_max_dist: float = 35.0,
) -> np.ndarray:
    img = Image.open(image_path).convert("RGB")
    w, h = img.size

    # patch_size is defined as width/out_size; height is handled proportionally via boundaries.
    patch_w = w / out_size
    patch_h = h / out_size

    rgb = np.asarray(img, dtype=np.uint8)  # [H, W, 3]
    mat = np.zeros((out_size, out_size), dtype=np.uint8)

    zero_colors = np.zeros((0, 3), dtype=np.uint8)
    if zero_ref_paths:
        zero_colors = np.stack([
            _dominant_color_from_ref_image(p) for p in zero_ref_paths
        ], axis=0)

    # Use boundary arrays to avoid accumulated float error.
    x_edges = np.rint(np.linspace(0, w, out_size + 1)).astype(int)
    y_edges = np.rint(np.linspace(0, h, out_size + 1)).astype(int)

    # Ensure edges are within bounds and monotonic.
    x_edges[0], x_edges[-1] = 0, w
    y_edges[0], y_edges[-1] = 0, h

    for i in range(out_size):
        y0, y1 = y_edges[i], y_edges[i + 1]
        if y1 <= y0:
            y1 = min(h, y0 + 1)
        for j in range(out_size):
            x0, x1 = x_edges[j], x_edges[j + 1]
            if x1 <= x0:
                x1 = min(w, x0 + 1)

            patch = rgb[y0:y1, x0:x1, :]
            if patch.size == 0:
                mat[i, j] = 0
                continue

            if zero_colors.size > 0:
                zero_mask = _match_any_color(patch, zero_colors, max_dist=zero_max_dist)
                zero_ratio = float(np.mean(zero_mask))
                mat[i, j] = 0 if zero_ratio >= zero_ratio_threshold else 1
            else:
                green_mask = is_green_pixel(patch, g_min=g_min, delta_rg=delta_rg, delta_bg=delta_bg)
                green_ratio = float(np.mean(green_mask))
                mat[i, j] = 1 if green_ratio >= green_ratio_threshold else 0

    # Keep patch_size concept documented via computed values (not otherwise required)
    _ = (patch_w, patch_h)
    return mat


def save_matrix_txt(mat: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in mat:
            f.write(" ".join(str(int(v)) for v in row))
            f.write("\n")


def main(num) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "将图片转换为128x128的0/1矩阵 "
            "如果提供了zero_ref参数 标注为0 "
            "如果绿色多就是1"
        )
    )
    path_int = base / "地图" / "地图源文件" / f"地图{i}.png"
    path_out = base / "地图" / "tensorImg" / f"image_{i}.csv"
    parser.add_argument("--image", type=str,default=path_int, help="输入图片的路径")#--是可以选择的
    parser.add_argument("--out", type=str, default=path_out, help="输出路径为地图里面的tensoring")
    parser.add_argument("--size", type=int, default=128, help="输出矩阵的大小 128默认")
    parser.add_argument(
        "--green_ratio",
        type=float,
        default=0.5,
        help="如果绿色格子大于0.5占比就记为1",
    )
    parser.add_argument("--g_min", type=int, default=80, help="Minimum G channel to be considered green")
    parser.add_argument("--delta_rg", type=int, default=20, help="Minimum (G-R) to be considered green")
    parser.add_argument("--delta_bg", type=int, default=20, help="Minimum (G-B) to be considered green")
    #?
    parser.add_argument(
        "--zero_ref",
        type=str,
        action="append",
        default=[],
        help=(
            "Reference image path whose dominant (non-transparent) color should be treated as 0. "
            "Can be specified multiple times. If provided, green-based mode is disabled."
        ),
    )
    parser.add_argument(
        "--zero_ratio",
        type=float,
        default=0.5,
        help="If ratio of pixels matching any --zero_ref color in a patch >= this => patch is 0 (default: 0.5)",
    )
    parser.add_argument(
        "--zero_max_dist",
        type=float,
        default=35.0,
        help="Max Euclidean RGB distance to consider a pixel matching a --zero_ref color (default: 35)",
    )

    args = parser.parse_args()
    #改这里
    image_path = Path(args.image).expanduser().resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    out_path = Path(args.out).expanduser()
    if not out_path.is_absolute():
        # Default output relative to Desktop if user runs from elsewhere
        # (keeps behavior predictable when double-click running).
        out_path = Path.home() / "Desktop" / out_path
    #自动把路径里的 ~（波浪号）换成你电脑的「用户主目录」
    zero_ref_paths = [Path(p).expanduser().resolve() for p in (args.zero_ref or [])]
    for p in zero_ref_paths:
        if not p.exists():
            raise FileNotFoundError(f"--zero_ref image not found: {p}")

    mat = image_to_green_majority_matrix(
        image_path,
        out_size=args.size,
        green_ratio_threshold=args.green_ratio,
        g_min=args.g_min,
        delta_rg=args.delta_rg,
        delta_bg=args.delta_bg,
        zero_ref_paths=zero_ref_paths if zero_ref_paths else None,
        zero_ratio_threshold=args.zero_ratio,
        zero_max_dist=args.zero_max_dist,
    )
    save_matrix_txt(mat, out_path)

    # Print a small preview
    print(f"Input: {image_path}")
    print(f"Output: {out_path}")
    print(f"Matrix shape: {mat.shape}")
    print("Preview (top-left 10x10):")
    print(mat[:10, :10])


if __name__ == "__main__":
    for i in range(1,11):
        main(i)
