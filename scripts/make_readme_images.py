"""Generate PNG figures for README from a synthetic beam.

Run from the repository root:
    python scripts/make_readme_images.py

Output: docs/images/*.png
"""
import os

import numpy as np
from scipy.ndimage import rotate
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from laser_beam_analysis import analyze_beam
from laser_beam_analysis.visualization import plot_3d_beam_cross_section


OUT_DIR = "docs/images"


def _synthetic_beam():
    h, w = 400, 400
    y, x = np.mgrid[0:h, 0:w]
    xc, yc = w / 2.0, h / 2.0
    beam = np.exp(
        -((x - xc) ** 2) / (2 * 10.0 ** 2)
        - ((y - yc) ** 2) / (2 * 100.0 ** 2)
    )
    beam = rotate(beam, 6.0, reshape=False, order=3, mode="constant", cval=0.0)
    rng = np.random.default_rng(0)
    img = beam * 200.0 + 30.0 + rng.normal(0, 2.0, size=(h, w))
    return img


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    image = _synthetic_beam()
    result = analyze_beam(image)

    # 1. Raw -> aligned -> background-subtracted
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(result.raw_image, cmap="viridis", aspect="auto")
    axes[0].set_title("Raw input")
    axes[1].imshow(result.aligned_image, cmap="viridis", aspect="auto")
    axes[1].set_title(f"Aligned (angle = {result.alignment['angle_deg']:.2f} deg)")
    axes[2].imshow(result.processed_image, cmap="viridis", aspect="auto")
    axes[2].set_title("Background-subtracted")
    for ax in axes:
        ax.set_xlabel("x, px")
        ax.set_ylabel("z, px")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/01_preprocessing.png", dpi=120)
    plt.close()

    # 2. BASEX vs Hansen-Law
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(result.basex_image, cmap="viridis", aspect="auto")
    axes[0].set_title("Inverse Abel: BASEX")
    axes[0].set_xlabel("r, px")
    axes[0].set_ylabel("z, px")
    axes[1].imshow(result.hansenlaw_image, cmap="viridis", aspect="auto")
    axes[1].set_title("Inverse Abel: Hansen-Law")
    axes[1].set_xlabel("r, px")
    axes[1].set_ylabel("z, px")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/02_abel.png", dpi=120)
    plt.close()

    # 3. w(z) from BASEX and Hansen-Law
    plt.figure(figsize=(10, 5), dpi=120)
    for label, profiles in (
        ("BASEX", result.profiles_basex),
        ("Hansen-Law", result.profiles_hansenlaw),
    ):
        z = [p["z_index"] for p in profiles if p["success"]]
        w = [p["beam_radius_w"] for p in profiles if p["success"]]
        plt.plot(z, w, marker="o", markersize=4, label=label)
    plt.xlabel("z, px")
    plt.ylabel("Beam radius w, px")
    plt.title("Beam width w(z) from Gaussian fits")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/03_beam_parameters.png", dpi=120)
    plt.close()

    # 4. 3D cross-section at the central z
    mid_z = result.basex_image.shape[0] // 2
    plot_3d_beam_cross_section(
        result.basex_image,
        z_index=mid_z,
        radius_max=40.0,
        figsize=(10, 7),
        dpi=120,
    )
    plt.savefig(f"{OUT_DIR}/04_3d_cross_section.png", dpi=120)
    plt.close()

    print("Images written to", OUT_DIR)


if __name__ == "__main__":
    main()