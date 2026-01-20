#!/usr/bin/env python3
"""
Photon Counting Compression Tool

Compresses EMCCD SER files using photon counting.
Default mode counts photons (0, 1, 2, 3...) preserving intensity information.

Usage:
    python photon_compress.py photons.ser bias.ser output.pcc
    python photon_compress.py photons.ser --bias-level 1490 output.pcc
"""

import sys
import argparse
import numpy as np
import zlib
import struct
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from ser import Ser
from emccd_stats import analyze_emccd

sys.stdout.reconfigure(line_buffering=True)

MAGIC = b'PCC2'


def compress_data(data):
    """Compress bytes using zlib level 9."""
    return zlib.compress(data, 9)


def pack_bits(binary_array):
    """Pack binary array into bytes (8 pixels per byte)."""
    flat = binary_array.ravel()
    pad_len = (8 - len(flat) % 8) % 8
    if pad_len:
        flat = np.concatenate([flat, np.zeros(pad_len, dtype=np.uint8)])
    return np.packbits(flat).tobytes()


def count_photons_hybrid(frame, em_gain, readout_noise):
    """
    Hybrid photon counting:
    - Binary (0 or 1) for faint pixels (< 2× gain) - most accurate for low signal
    - Proportional for bright pixels (stars) - preserves intensity/shape

    This avoids misclassifying 1-photon events as 2+ (exponential tail problem)
    while still preserving star intensities for alignment.
    """
    thresh_detect = 3 * readout_noise  # Detection threshold
    thresh_bright = 2.0 * em_gain       # Transition to proportional

    counts = np.zeros(frame.shape, dtype=np.uint8)

    # Faint pixels: binary (0 or 1)
    faint_photon = (frame > thresh_detect) & (frame <= thresh_bright)
    counts[faint_photon] = 1

    # Bright pixels: proportional (divide by gain, round)
    bright = frame > thresh_bright
    if np.any(bright):
        # Estimate photon count from signal
        photon_est = frame[bright] / em_gain
        # Round to nearest integer, minimum 2 (since we're above thresh_bright)
        counts[bright] = np.clip(np.round(photon_est), 2, 255).astype(np.uint8)

    return counts


def main():
    parser = argparse.ArgumentParser(description='Compress EMCCD SER to photon counts')
    parser.add_argument('input', help='Input SER file')
    parser.add_argument('bias', help='Bias SER file')
    parser.add_argument('output', help='Output PCC file')
    parser.add_argument('--bias-level', type=float, default=None,
                        help='Use constant bias level instead of bias file')
    parser.add_argument('--sigma', type=float, default=5.0,
                        help='Detection threshold in sigma (default: 5.0)')
    parser.add_argument('--binary', action='store_true',
                        help='Use binary mode (0/1 only, better compression)')
    parser.add_argument('--threads', type=int, default=8,
                        help='Number of threads (default: 8)')

    args = parser.parse_args()

    print("=" * 60, flush=True)
    mode_name = "binary" if args.binary else "multilevel"
    print(f"Photon Compression - Mode: {mode_name}", flush=True)
    print("=" * 60, flush=True)

    # Open input
    print(f"\nOpening {args.input}...", flush=True)
    ser = Ser(args.input)
    n_frames = ser.count
    height, width = ser.ysize, ser.xsize
    print(f"  {n_frames} frames, {width}x{height}", flush=True)

    # Get bias
    if args.bias_level is not None:
        master_bias = np.full((height, width), args.bias_level, dtype=np.float32)
        print(f"Using constant bias: {args.bias_level}", flush=True)
    else:
        print(f"Loading bias from {args.bias}...", flush=True)
        bias_ser = Ser(args.bias)
        bias_frames = [bias_ser.load_img(i).astype(np.float32)
                       for i in range(bias_ser.count)]
        bias_ser.close()
        master_bias = np.median(bias_frames, axis=0)
        print(f"  {len(bias_frames)} frames -> {master_bias.mean():.1f} ADU", flush=True)
        del bias_frames

    # Estimate EMCCD stats
    print("Analyzing statistics...", flush=True)
    sample = []
    for i in range(min(50, n_frames)):
        sample.append(ser.load_img(i).astype(np.float32) - master_bias)
    stats = analyze_emccd(np.array(sample).ravel(), verbose=False)
    del sample

    readout_noise = stats['readout_noise'] or 10.0
    em_gain = stats['em_gain'] or 50.0
    threshold = args.sigma * readout_noise

    print(f"  Readout: {readout_noise:.1f} ADU", flush=True)
    print(f"  EM gain: {em_gain:.1f} ADU/e-", flush=True)
    print(f"  Threshold ({args.sigma}σ): {threshold:.1f} ADU", flush=True)

    # Mode: 1=binary, 2=multilevel
    mode_byte = 1 if args.binary else 2

    def process_frame(frame_data):
        """Process single frame in thread."""
        frame = frame_data.astype(np.float32) - master_bias

        if args.binary:
            binary = (frame > threshold).astype(np.uint8)
            nonzero = np.sum(binary)
            raw = pack_bits(binary)
        else:
            counts = count_photons_hybrid(frame, em_gain, readout_noise)
            nonzero = np.sum(counts > 0)
            raw = counts.tobytes()

        compressed = compress_data(raw)
        return compressed, nonzero, frame.size

    # Load all frames
    print(f"\nProcessing {n_frames} frames ({args.threads} threads)...", flush=True)
    print("  Loading frames...", flush=True)
    frames = []
    for i in range(n_frames):
        frames.append(ser.load_img(i))
        if (i + 1) % 500 == 0:
            print(f"    {i+1}/{n_frames} loaded...", flush=True)
    ser.close()

    # Process in parallel
    print("  Compressing...", flush=True)
    results = [None] * n_frames
    total_ones = 0
    total_pixels = 0
    completed = 0
    lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        futures = {executor.submit(process_frame, frames[i]): i
                   for i in range(n_frames)}

        for future in as_completed(futures):
            idx = futures[future]
            compressed, nonzero, pixels = future.result()
            results[idx] = compressed

            with lock:
                total_ones += nonzero
                total_pixels += pixels
                completed += 1
                if completed % 200 == 0:
                    pct = 100 * total_ones / total_pixels
                    print(f"    {completed}/{n_frames} ({pct:.1f}% non-zero)...", flush=True)

    # Write output
    print("  Writing output...", flush=True)
    with open(args.output, 'wb') as f:
        f.write(MAGIC)
        header = struct.pack('<IIIffffBB',
            n_frames, height, width,
            threshold, readout_noise, em_gain, args.sigma,
            mode_byte, 0)
        f.write(struct.pack('<I', len(header)))
        f.write(header)

        for r in results:
            f.write(struct.pack('<I', len(r)))
        for r in results:
            f.write(r)

    # Summary
    import os
    in_size = os.path.getsize(args.input)
    out_size = os.path.getsize(args.output)

    print(f"\n" + "=" * 60, flush=True)
    print("SUMMARY", flush=True)
    print("=" * 60, flush=True)
    print(f"  Mode:        {mode_name}", flush=True)
    print(f"  Non-zero:    {100 * total_ones / total_pixels:.1f}%", flush=True)
    print(f"  Input:       {in_size / 1024**2:.1f} MB", flush=True)
    print(f"  Output:      {out_size / 1024**2:.1f} MB", flush=True)
    print(f"  Ratio:       {in_size / out_size:.1f}x", flush=True)


if __name__ == "__main__":
    main()
