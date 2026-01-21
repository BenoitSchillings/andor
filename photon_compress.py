#!/usr/bin/env python3
"""
Photon Counting Compression Tool

Compresses EMCCD SER files with bias subtraction.

Modes:
  lossless (default) - Preserves exact bias-subtracted values (int16)
  hybrid             - Photon counting: binary for faint, proportional for bright
  binary             - Binary detection only (0/1), best compression

Usage:
    python photon_compress.py *.ser -b bias.ser
    python photon_compress.py *.ser -b bias.ser --hybrid
    python photon_compress.py *.ser -b bias.ser --binary -o /output/dir
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


def compress_file(input_path, output_path, master_bias, em_gain, readout_noise,
                  sigma, mode, n_threads):
    """
    Compress a single SER file to PCC format.

    mode: 'lossless', 'hybrid', or 'binary'
    """
    threshold = sigma * readout_noise
    # Mode bytes: 0=lossless, 1=binary, 2=hybrid
    mode_byte = {'lossless': 0, 'binary': 1, 'hybrid': 2}[mode]

    # Open input
    ser = Ser(input_path)
    n_frames = ser.count
    height, width = ser.ysize, ser.xsize

    def process_frame(frame_data):
        """Process single frame in thread."""
        frame = frame_data.astype(np.float32) - master_bias

        if mode == 'lossless':
            # Preserve exact values as int16, but zero out sub-photon noise
            frame[frame < em_gain] = 0
            frame_int = np.clip(frame, -32768, 32767).astype(np.int16)
            nonzero = np.sum(frame_int != 0)
            raw = frame_int.tobytes()
        elif mode == 'binary':
            binary = (frame > threshold).astype(np.uint8)
            nonzero = np.sum(binary)
            raw = pack_bits(binary)
        else:  # hybrid
            counts = count_photons_hybrid(frame, em_gain, readout_noise)
            nonzero = np.sum(counts > 0)
            raw = counts.tobytes()

        compressed = compress_data(raw)
        return compressed, nonzero, frame.size

    # Load all frames
    frames = []
    for i in range(n_frames):
        frames.append(ser.load_img(i))
    ser.close()

    # Process in parallel
    results = [None] * n_frames
    total_ones = 0
    total_pixels = 0
    lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        futures = {executor.submit(process_frame, frames[i]): i
                   for i in range(n_frames)}

        for future in as_completed(futures):
            idx = futures[future]
            compressed, nonzero, pixels = future.result()
            results[idx] = compressed

            with lock:
                total_ones += nonzero
                total_pixels += pixels

    # Write output
    with open(output_path, 'wb') as f:
        f.write(MAGIC)
        header = struct.pack('<IIIffffBB',
            n_frames, height, width,
            threshold, readout_noise, em_gain, sigma,
            mode_byte, 0)
        f.write(struct.pack('<I', len(header)))
        f.write(header)

        for r in results:
            f.write(struct.pack('<I', len(r)))
        for r in results:
            f.write(r)

    return n_frames, total_ones, total_pixels


def main():
    parser = argparse.ArgumentParser(description='Compress EMCCD SER to photon counts')
    parser.add_argument('inputs', nargs='+', help='Input SER file(s)')
    parser.add_argument('--bias', '-b', required=True, help='Bias SER file')
    parser.add_argument('--output-dir', '-o', default=None,
                        help='Output directory (default: same as input)')
    parser.add_argument('--bias-level', type=float, default=None,
                        help='Use constant bias level instead of bias file')
    parser.add_argument('--sigma', type=float, default=5.0,
                        help='Detection threshold in sigma (default: 5.0)')
    # Compression modes (mutually exclusive)
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument('--lossless', action='store_true', default=True,
                        help='Lossless mode - preserve exact values (default)')
    mode_group.add_argument('--hybrid', action='store_true',
                        help='Hybrid photon counting (binary for faint, proportional for bright)')
    mode_group.add_argument('--binary', action='store_true',
                        help='Binary mode (0/1 only, best compression)')
    parser.add_argument('--threads', type=int, default=8,
                        help='Number of threads (default: 8)')

    args = parser.parse_args()

    import os
    from pathlib import Path

    # Determine mode
    if args.binary:
        mode = 'binary'
    elif args.hybrid:
        mode = 'hybrid'
    else:
        mode = 'lossless'
    print("=" * 60, flush=True)
    print(f"Photon Compression - Mode: {mode}", flush=True)
    print(f"  Files: {len(args.inputs)}", flush=True)
    print("=" * 60, flush=True)

    # Get first file dimensions for bias
    first_ser = Ser(args.inputs[0])
    height, width = first_ser.ysize, first_ser.xsize
    n_sample_frames = first_ser.count

    # Get bias (with caching)
    if args.bias_level is not None:
        master_bias = np.full((height, width), args.bias_level, dtype=np.float32)
        print(f"\nUsing constant bias: {args.bias_level}", flush=True)
    else:
        bias_path = Path(args.bias)
        cache_path = bias_path.with_suffix('.bias.npy')

        # Check if cache exists and is newer than source
        if cache_path.exists() and cache_path.stat().st_mtime > bias_path.stat().st_mtime:
            print(f"\nLoading cached bias from {cache_path.name}...", flush=True)
            master_bias = np.load(cache_path)
            print(f"  mean {master_bias.mean():.1f} ADU", flush=True)
        else:
            print(f"\nComputing bias from {args.bias}...", flush=True)
            bias_ser = Ser(args.bias)
            bias_frames = [bias_ser.load_img(i).astype(np.float32)
                           for i in range(bias_ser.count)]
            bias_ser.close()
            master_bias = np.median(bias_frames, axis=0)
            print(f"  {len(bias_frames)} frames -> mean {master_bias.mean():.1f} ADU", flush=True)
            del bias_frames
            # Cache for next time
            np.save(cache_path, master_bias)
            print(f"  Cached to {cache_path.name}", flush=True)

    # Estimate EMCCD stats from first file
    print("Analyzing statistics from first file...", flush=True)
    sample = []
    for i in range(min(50, n_sample_frames)):
        sample.append(first_ser.load_img(i).astype(np.float32) - master_bias)
    first_ser.close()

    stats = analyze_emccd(np.array(sample).ravel(), verbose=False)
    del sample

    readout_noise = stats['readout_noise'] or 10.0
    em_gain = stats['em_gain'] or 50.0

    print(f"  Readout: {readout_noise:.1f} ADU", flush=True)
    print(f"  EM gain: {em_gain:.1f} ADU/e-", flush=True)
    print(f"  Threshold ({args.sigma}σ): {args.sigma * readout_noise:.1f} ADU", flush=True)

    # Process each file
    total_in_size = 0
    total_out_size = 0
    total_frames = 0
    grand_total_ones = 0
    grand_total_pixels = 0

    for i, input_path in enumerate(args.inputs):
        input_path = Path(input_path)

        # Determine output path
        if args.output_dir:
            out_dir = Path(args.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            output_path = out_dir / input_path.with_suffix('.pcc').name
        else:
            output_path = input_path.with_suffix('.pcc')

        print(f"\n[{i+1}/{len(args.inputs)}] {input_path.name}", flush=True)

        # Compress file
        n_frames, file_ones, file_pixels = compress_file(
            str(input_path), str(output_path), master_bias,
            em_gain, readout_noise, args.sigma, mode, args.threads
        )

        # Stats
        in_size = os.path.getsize(input_path)
        out_size = os.path.getsize(output_path)
        ratio = in_size / out_size if out_size > 0 else 0
        pct_nonzero = 100 * file_ones / file_pixels if file_pixels > 0 else 0

        print(f"  {n_frames} frames, {pct_nonzero:.1f}% non-zero, {ratio:.1f}x compression", flush=True)
        print(f"  -> {output_path.name} ({out_size / 1024**2:.1f} MB)", flush=True)

        total_in_size += in_size
        total_out_size += out_size
        total_frames += n_frames
        grand_total_ones += file_ones
        grand_total_pixels += file_pixels

    # Summary
    print(f"\n" + "=" * 60, flush=True)
    print("TOTAL SUMMARY", flush=True)
    print("=" * 60, flush=True)
    print(f"  Files:       {len(args.inputs)}", flush=True)
    print(f"  Frames:      {total_frames}", flush=True)
    print(f"  Mode:        {mode}", flush=True)
    if grand_total_pixels > 0:
        print(f"  Non-zero:    {100 * grand_total_ones / grand_total_pixels:.1f}%", flush=True)
    print(f"  Input:       {total_in_size / 1024**2:.1f} MB", flush=True)
    print(f"  Output:      {total_out_size / 1024**2:.1f} MB", flush=True)
    if total_out_size > 0:
        print(f"  Ratio:       {total_in_size / total_out_size:.1f}x", flush=True)


if __name__ == "__main__":
    main()
