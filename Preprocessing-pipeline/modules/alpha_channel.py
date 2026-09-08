import os
from pathlib import Path

import cv2
import numpy as np
from .base_module import BaseProcessingModule

TIME_TOLERANCE_S = 2


def _capture_time(name):
    """Seconds-of-day from an IMG_yymmdd_hhmmss_nnnn_BAND filename."""
    stamp = Path(name).stem.split("_")[2]
    return int(stamp[0:2]) * 3600 + int(stamp[2:4]) * 60 + int(stamp[4:6])


class AlphaChannelStep(BaseProcessingModule):
    """Aligns each secondary band onto the reference band, like ImageAlignmentStep,
    but keeps the SIFT/RANSAC warp's coverage mask as a 4th channel instead of
    discarding it.

    A secondary sensor's footprint can be offset from the reference one, so a
    warped frame is only partly filled; downstream consumers that read the
    alpha channel (the AOS view loader) can then exclude the unfilled region
    from an integral instead of averaging it in as though it were black
    scenery. ImageAlignmentStep writes 3-channel JPEGs and has no way to carry
    that distinction, which is why this is a separate step rather than a flag
    on it. Output goes to the same 'align' folder ImageAlignmentStep uses, as
    PNG instead of JPEG so the alpha channel survives.
    """

    def action(self):
        DIR = self.DIR
        ref_band = self.channel_names[0]
        ref_dir = os.path.join(DIR, f'{ref_band}_irradiancee_RGB', 'undistord_solving')
        ref_imgs = sorted(os.listdir(ref_dir))
        bands = [f'{band}_irradiancee_RGB' for band in self.channel_names[1:]]

        for img in ref_imgs:
            ref_img = cv2.imread(os.path.join(ref_dir, img), -1)
            for band in bands:
                band_name = band.replace('_irradiancee_RGB', '')
                target_img = img.replace(ref_band, band_name)
                band_dir = os.path.join(DIR, band, 'undistord_solving')
                target_path = os.path.join(band_dir, target_img)
                if not os.path.exists(target_path):
                    # The secondary sensor stamps its own trigger time, which
                    # can land a second off the reference band's. Fall back to
                    # the nearest capture within TIME_TOLERANCE_S instead of
                    # dropping the view outright.
                    ref_t = _capture_time(img)
                    candidates = os.listdir(band_dir) if os.path.isdir(band_dir) else []
                    near = sorted((abs(_capture_time(c) - ref_t), c) for c in candidates)
                    if not near or near[0][0] > TIME_TOLERANCE_S:
                        print(f"  no source for {target_img}, skipping")
                        continue
                    target_img = near[0][1]
                    target_path = os.path.join(band_dir, target_img)
                    print(f"  {img} -> nearest {band_name} capture is {target_img} "
                          f"({near[0][0]}s off)")
                tgt_img = cv2.imread(target_path, -1)

                sift = cv2.SIFT_create()
                kp1, des1 = sift.detectAndCompute(tgt_img, None)
                kp2, des2 = sift.detectAndCompute(ref_img, None)
                if des1 is None or des2 is None:
                    print(f"  no features for {target_img}, skipping")
                    continue

                matches = cv2.BFMatcher().knnMatch(des1, des2, k=2)
                good = [m for m, n in matches if m.distance < 0.75 * n.distance]
                if len(good) < 4:
                    print(f"  only {len(good)} good match(es) for {target_img}, skipping")
                    continue

                src = np.float32([kp1[m.queryIdx].pt for m in good])
                dst = np.float32([kp2[m.trainIdx].pt for m in good])
                H, _ = cv2.findHomography(src, dst, cv2.RANSAC)
                if H is None:
                    print(f"  homography failed for {target_img}, skipping")
                    continue

                size = (ref_img.shape[1], ref_img.shape[0])
                warped = cv2.warpPerspective(tgt_img, H, size)
                solid = np.full(tgt_img.shape[:2], 255, dtype=np.uint8)
                mask = cv2.warpPerspective(solid, H, size)
                # Warp interpolation softens the boundary; keep only fully covered pixels.
                mask = (mask == 255).astype(np.uint8) * 255

                if warped.ndim == 2:
                    warped = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)
                rgba = np.dstack([warped[:, :, :3], mask])

                align_dir = os.path.join(DIR, band, 'align')
                os.makedirs(align_dir, exist_ok=True)
                # Named after the reference (NIR) capture, not the matched
                # target -- the tolerance fallback above can pick a target
                # file stamped a second later, and every other consumer
                # (poses export included) expects one output per reference
                # capture, keyed by the reference's own timestamp.
                name = img.split('.')[0][:-3]
                out_name = name + band_name + '.png'
                cv2.imwrite(os.path.join(align_dir, out_name), rgba)

        print("AlphaChannelStep done")
