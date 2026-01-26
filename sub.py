#!/usr/bin/env python3
"""
Subtract bias from SER file.

Usage:
    python sub.py input.ser bias.fits output.ser
"""

import sys
import numpy as np
from astropy.io import fits
from ser import Ser, SerWriter


def main():
    if len(sys.argv) != 4:
        print("Usage: python sub.py input.ser bias.fits output.ser")
        sys.exit(1)

    input_path = sys.argv[1]
    bias_path = sys.argv[2]
    output_path = sys.argv[3]

    # Load bias
    print(f"Loading bias from {bias_path}...")
    with fits.open(bias_path) as hdul:
        bias = hdul[0].data.astype(np.float32)
    print(f"  Bias shape: {bias.shape}, mean: {bias.mean():.1f}")

    # Open input SER
    print(f"Opening {input_path}...")
    ser = Ser(input_path)
    n_frames = ser.count
    height, width = ser.ysize, ser.xsize
    print(f"  {n_frames} frames, {width}x{height}")

    # Check dimensions match
    if bias.shape != (height, width):
        print(f"Error: Bias shape {bias.shape} doesn't match SER shape ({height}, {width})")
        sys.exit(1)

    # Create output SER
    print(f"Writing {output_path}...")
    out = SerWriter(output_path)
    out.set_sizes(width, height, 2)  # 16-bit output

    # Process frames
    for i in range(n_frames):
        frame = ser.load_img(i).astype(np.float32)
        subtracted = frame - bias
        # Clip to valid range and convert to uint16
        result = np.clip(subtracted, 0, 65535).astype(np.uint16)
        out.add_image(result)

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{n_frames} frames...")

    ser.close()
    out.close()

    print(f"Done! Output: {output_path}")


if __name__ == "__main__":
    main()
