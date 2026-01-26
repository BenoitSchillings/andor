#!/usr/bin/env python3
"""
Average all frames from a SER file and save as FITS.

Usage:
    python mean.py input.ser output.fits
"""

import sys
import numpy as np
from ser import Ser
from astropy.io import fits


def main():
    if len(sys.argv) != 3:
        print("Usage: python mean.py input.ser output.fits")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    # Open SER file
    ser = Ser(input_file)
    print(f"Loaded {input_file}: {ser.count} frames, {ser.xsize}x{ser.ysize}, {ser.depth*8}-bit")

    # Accumulate frames
    accumulator = np.zeros((ser.ysize, ser.xsize), dtype=np.float64)

    for i in range(ser.count):
        frame = ser.load_img(i)
        accumulator += frame
        if (i + 1) % 100 == 0 or i == ser.count - 1:
            print(f"\rProcessing: {i + 1}/{ser.count}", end="", flush=True)

    print()
    ser.close()

    # Compute mean
    mean_frame = (accumulator / ser.count).astype(np.float32)

    # Save as FITS
    hdu = fits.PrimaryHDU(mean_frame)
    hdu.header['NFRAMES'] = ser.count
    hdu.header['COMMENT'] = f'Mean of {ser.count} frames from {input_file}'
    hdu.writeto(output_file, overwrite=True)

    print(f"Saved {output_file} (mean of {ser.count} frames)")
    print(f"  Min: {mean_frame.min():.1f}, Max: {mean_frame.max():.1f}, Mean: {mean_frame.mean():.1f}")


if __name__ == "__main__":
    main()
