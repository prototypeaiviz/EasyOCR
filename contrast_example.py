"""
Understanding contrast_grey() and adjust_contrast_grey()
=========================================================
These two functions live in easyocr/recognition.py and are used during the
two-pass inference strategy: if the model is not confident about a prediction,
it re-runs the image with contrast enhancement to make the text more visible.

Run:
    python contrast_example.py
"""

import numpy as np
from PIL import Image
import os

# ── LOAD IMAGE ────────────────────────────────────────────────────────────────

dataset_dir = "examples_aiviz_dataset"
image_name  = "A7kL2 p9Qx__03.png"
image_path  = os.path.join(dataset_dir, image_name)

# Load as grayscale (recognition model only sees L-mode images)
img_pil = Image.open(image_path).convert("L")
img     = np.array(img_pil)   # shape [H, W], dtype uint8, values 0–255

print(f"Image shape : {img.shape}")
print(f"Pixel range : min={img.min()}  max={img.max()}  mean={img.mean():.1f}")


# ═══════════════════════════════════════════════════════════════════════════════
# FUNCTION 1 — contrast_grey(img)
# ═══════════════════════════════════════════════════════════════════════════════
#
# Goal: measure how much contrast the image has (0 = flat grey, 1 = full B&W).
#
# Why percentiles and not min/max?
#   A single white dot or black speck would make min=0, max=255 even if the
#   rest of the image is uniformly grey.  Percentiles are robust — they ignore
#   the extreme outliers and describe the bulk of the pixel distribution.

def contrast_grey(img):
    high = np.percentile(img, 90)   # 90% of pixels are BELOW this value
    low  = np.percentile(img, 10)   # 10% of pixels are BELOW this value
    #
    # contrast = (high - low) / (high + low)
    #
    #   numerator   high - low : the spread of pixel values
    #                            wide spread  → lots of dark AND bright pixels → high contrast
    #                            narrow spread → all pixels similar colour    → low contrast
    #
    #   denominator high + low : a normalisation factor relative to the brightness level
    #                            np.maximum(10, ...) prevents division by zero when the
    #                            image is nearly black (high ≈ 0, low ≈ 0)
    #
    # Examples:
    #   Pure black-and-white:  high=255, low=0   → (255-0)/(255+0)   = 1.0  (maximum)
    #   Mid-grey everywhere:   high=128, low=128 → (0)/(256)         = 0.0  (minimum)
    #   Faint text on white:   high=240, low=200 → (40)/(440) ≈ 0.09 (very low)
    #
    return (high - low) / np.maximum(10, high + low), high, low


# ── RUN IT ────────────────────────────────────────────────────────────────────

contrast, high, low = contrast_grey(img)

print("\n--- contrast_grey() ---")
print(f"  p10 (low)  = {low:.1f}   ← 10% of pixels are darker than this")
print(f"  p90 (high) = {high:.1f}   ← 90% of pixels are darker than this")
print(f"  spread     = high - low = {high - low:.1f}")
print(f"  contrast   = {contrast:.4f}   (0=flat grey, 1=full black-white)")

if contrast >= 0.4:
    print("  → contrast is GOOD (≥ 0.4), no adjustment needed")
else:
    print("  → contrast is LOW (< 0.4), adjustment WILL be applied")


# ═══════════════════════════════════════════════════════════════════════════════
# FUNCTION 2 — adjust_contrast_grey(img, target=0.4)
# ═══════════════════════════════════════════════════════════════════════════════
#
# Goal: if contrast < target, linearly STRETCH the pixel values so the image
# fills the full 0–255 range.  This makes faint text darker and bright
# backgrounds whiter, giving the model a clearer signal.
#
# The math is a simple linear rescaling:
#
#   Step 1: ratio = 200 / (high - low)
#           This is the scale factor.  If high-low=40 (narrow range),
#           ratio = 200/40 = 5.0 → we will stretch 5× to fill more range.
#
#   Step 2: img_new = (img - low + 25) * ratio
#           - (img - low) : shifts so the darkest pixel becomes ~0
#           - + 25        : a small margin so blacks are not clipped to exactly 0
#           - × ratio     : stretches the range
#
#   Step 3: clip to [0, 255]
#           After stretching some values may go below 0 or above 255.
#           np.maximum / np.minimum clamp them back to valid uint8 range.

def adjust_contrast_grey(img, target=0.4):
    contrast, high, low = contrast_grey(img)

    if contrast >= target:
        # Already contrasty enough — leave the image completely unchanged.
        print(f"\nadjust_contrast_grey(): contrast={contrast:.3f} ≥ target={target} → NO CHANGE")
        return img

    # --- need to stretch ---
    img = img.astype(int)   # convert to int first so arithmetic doesn't wrap
    # the 200 is the target spread you want between p10 and p90 after stretching
    #  200 leaves room so the stretch is strong but not brutal. It is a hand-tuned constant — "make the contrast good, but not violently overexposed."
    ratio = 200. / np.maximum(10, high - low)
    img_stretched = (img - low + 25) * ratio

    # Clip to [0, 255] and convert back to uint8
    img_clipped = np.maximum(0, np.minimum(255, img_stretched)).astype(np.uint8)

    new_contrast, new_high, new_low = contrast_grey(img_clipped)

    print(f"\n--- adjust_contrast_grey() ---")
    print(f"  Input  contrast = {contrast:.4f}  (p10={low:.0f}, p90={high:.0f})")
    print(f"  ratio  = 200 / max(10, {high:.0f} - {low:.0f}) = {ratio:.3f}")
    print(f"  formula: (pixel - {low:.0f} + 25) × {ratio:.3f}, then clip to [0,255]")
    print(f"  Output contrast = {new_contrast:.4f}  (p10={new_low:.0f}, p90={new_high:.0f})")
    print(f"  Pixel range before: [{img.min()}, {img.max()}]")
    print(f"  Pixel range after : [{img_clipped.min()}, {img_clipped.max()}]")

    return img_clipped


# ── RUN IT ────────────────────────────────────────────────────────────────────

img_adjusted = adjust_contrast_grey(img, target=0.4)


# ── TRACE THROUGH A SINGLE PIXEL ─────────────────────────────────────────────
# Pick a mid-brightness pixel and show exactly what the formula does to it.

print("\n--- Tracing the formula on a single pixel ---")
sample_pixel_val = int(img.mean())   # use the average pixel as an example
_, high, low = contrast_grey(img)
ratio = 200. / max(10, high - low)

before = sample_pixel_val
after  = (before - low + 25) * ratio
after  = max(0, min(255, after))

print(f"  Example pixel value : {before}")
print(f"  Step 1 — shift      : {before} - {low:.0f} + 25 = {before - low + 25:.1f}")
print(f"  Step 2 — scale      : {before - low + 25:.1f} × {ratio:.3f} = {(before - low + 25) * ratio:.1f}")
print(f"  Step 3 — clip       : clamp({(before - low + 25) * ratio:.1f}, 0, 255) = {after:.1f}")
print(f"  Net effect : pixel {before} → {after:.0f}  (shifted {'brighter' if after > before else 'darker'})")


# ── SAVE BOTH IMAGES SO YOU CAN SEE THE DIFFERENCE ───────────────────────────

out_original = "contrast_original.png"
out_adjusted = "contrast_adjusted.png"

Image.fromarray(img).save(out_original)
Image.fromarray(img_adjusted).save(out_adjusted)

print(f"\nSaved '{out_original}' and '{out_adjusted}' — open both to see the effect.")
