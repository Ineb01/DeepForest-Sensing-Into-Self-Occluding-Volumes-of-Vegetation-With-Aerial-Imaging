"""
Bring a campaign's RGB band up to alpha-aware alignment with NIR, and export
matching poses for it.

Runs AlphaChannelStep (SIFT + RANSAC onto NIR, alpha marks warp coverage),
crops the 50 px border the rest of the pipeline uses, and writes
poses_<campaign>_RGB.json alongside the existing per-band pose files. No
COLMAP rerun: every RGB view is aligned onto a specific NIR capture, so it
inherits that capture's pose outright, exactly like RED already does.

    python build_rgb_alpha.py --dataset March
    python build_rgb_alpha.py --dataset May
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from controllers import MultiSpectralProcessor
from modules import AlphaChannelStep

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = PROJECT_ROOT / "practical" / "data"
POSE_DIR = PROJECT_ROOT / "practical" / "LFR" / "poses"

CROP_MARGIN = 50


def crop_rgb_only(dataset_dir):
    """Crop just the RGB band's 'align' RGBA output, leaving NIR/RED cropped/
    untouched -- CropBordersStep would also re-touch the reference band."""
    src = dataset_dir / "RGB_irradiancee_RGB" / "align"
    dst = dataset_dir / "RGB_irradiancee_RGB" / "cropped_rgba"
    dst.mkdir(parents=True, exist_ok=True)
    names = sorted(os.listdir(src))
    for name in names:
        img = cv2.imread(str(src / name), cv2.IMREAD_UNCHANGED)
        h, w = img.shape[:2]
        img = img[CROP_MARGIN:h - CROP_MARGIN, CROP_MARGIN:w - CROP_MARGIN]
        cv2.imwrite(str(dst / name), img)
    print(f"  cropped {len(names)} RGBA image(s) -> {dst}")
    return names


def export_rgb_poses(campaign, cropped_names):
    """poses_<campaign>_RGB.json, keyed off poses_<campaign>_NIR.json.

    cropped_names are '..._RGB.png' stems produced by AlphaChannelStep, which
    already carries the exact NIR capture each RGB view was warped onto: it
    builds the RGB filename straight from the reference frame it aligned to.
    """
    nir_path = POSE_DIR / f"poses_{campaign.lower()}_NIR.json"
    nir_poses = json.load(open(nir_path))["images"]

    have = {Path(n).stem.lower() for n in cropped_names}
    images, missing = [], []
    for entry in nir_poses:
        stem = Path(entry["imagefile"]).stem
        rgb_stem = stem.replace("_NIR", "_RGB").replace("_nir", "_RGB")
        if rgb_stem.lower() not in have:
            missing.append(rgb_stem)
            continue
        images.append({
            "imagefile": rgb_stem + ".png",
            "M3x4": entry["M3x4"],
        })

    out = POSE_DIR / f"poses_{campaign.lower()}_RGB.json"
    with open(out, "w") as f:
        json.dump({"images": images}, f, indent=4)
    print(f"  {len(images)} pose(s) -> {out.name}"
          + (f", {len(missing)} NIR capture(s) had no aligned RGB: {missing[:3]}" if missing else ""))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=["October", "March", "May"])
    args = parser.parse_args()

    dataset_dir = DATA_ROOT / f"dataset_{args.dataset}"
    if not dataset_dir.is_dir():
        sys.exit(f"no such dataset: {dataset_dir}")

    print(f"{args.dataset}: aligning RGB onto NIR with alpha coverage")
    controller = MultiSpectralProcessor(str(dataset_dir), ["NIR", "RGB"], AlphaChannelStep)
    controller.process()

    print(f"{args.dataset}: cropping RGB only")
    cropped_names = crop_rgb_only(dataset_dir)

    print(f"{args.dataset}: exporting RGB poses")
    export_rgb_poses(args.dataset, cropped_names)


if __name__ == "__main__":
    main()
