"""Debug: isolate where the z_scan fit failure comes from."""
import traceback

import numpy as np

from laser_beam_analysis.profiles import (
    extract_radial_profile,
    fit_gaussian_radial_profile,
)


def main():
    rng = np.random.default_rng(0)
    h, w = 81, 81
    y, x = np.mgrid[0:h, 0:w]
    xc, yc = (w - 1) / 2.0, (h - 1) / 2.0
    img = np.exp(-((x - xc) ** 2 + (y - yc) ** 2) / (2 * 12.0 ** 2))
    img = img + rng.normal(0, 0.001, size=(h, w))

    print("img.shape:", img.shape)
    print("img[40, 40]:", img[40, 40])
    print("any NaN in img:", bool(np.any(~np.isfinite(img))))

    r, profile = extract_radial_profile(img, z_index=40)
    print("r.size:", r.size)
    print("r[:5]:", r[:5], "...", r[-5:])
    print("profile[:5]:", profile[:5])
    print("profile[-5:]:", profile[-5:])
    print("profile.argmax:", int(np.argmax(profile)))
    print("profile.max:", float(np.max(profile)))
    print("any NaN in profile:", bool(np.any(~np.isfinite(profile))))

    try:
        res = fit_gaussian_radial_profile(
            r, profile, fit_fraction=0.05, robust=True, r0_max=3.0
        )
        print("FIT OK -> sigma =", res["sigma"])
    except Exception:
        print("FIT FAILED, traceback:")
        traceback.print_exc()


if __name__ == "__main__":
    main()