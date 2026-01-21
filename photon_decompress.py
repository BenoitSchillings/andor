#!/usr/bin/env python3
"""
Photon Counting Decompression Tool

Decompresses .pcc files back to SER format.

Usage:
    python photon_decompress.py input.pcc output.ser
    python photon_decompress.py input.pcc --info
"""

import sys
import argparse
import numpy as np
import zlib
import struct

from ser import SerWriter

sys.stdout.reconfigure(line_buffering=True)

MAGIC = b'PCC2'
MODE_NAMES = {0: 'lossless', 1: 'binary', 2: 'hybrid'}


def unpack_bits(packed_bytes, shape):
    """Unpack bytes back to binary array."""
    packed = np.frombuffer(packed_bytes, dtype=np.uint8)
    unpacked = np.unpackbits(packed)
    return unpacked[:shape[0] * shape[1]].reshape(shape)


def main():
    parser = argparse.ArgumentParser(description='Decompress PCC to SER')
    parser.add_argument('input', help='Input PCC file')
    parser.add_argument('output', nargs='?', help='Output SER file')
    parser.add_argument('--info', action='store_true', help='Show info only')
    parser.add_argument('--raw', action='store_true',
                        help='Output raw photon counts (not scaled)')

    args = parser.parse_args()

    if not args.info and not args.output:
        print("Error: output file required (or use --info)")
        sys.exit(1)

    print("=" * 60, flush=True)
    print("Photon Decompression", flush=True)
    print("=" * 60, flush=True)

    with open(args.input, 'rb') as f:
        magic = f.read(4)
        if magic != MAGIC:
            print(f"Error: Invalid file (got {magic})")
            sys.exit(1)

        header_size = struct.unpack('<I', f.read(4))[0]
        header = f.read(header_size)
        n_frames, height, width, threshold, readout, em_gain, sigma, mode, _ = \
            struct.unpack('<IIIffffBB', header)

        print(f"\n  Frames:    {n_frames}", flush=True)
        print(f"  Size:      {width}x{height}", flush=True)
        print(f"  Mode:      {MODE_NAMES.get(mode, mode)}", flush=True)
        print(f"  Threshold: {threshold:.1f} ADU ({sigma}σ)", flush=True)
        print(f"  EM Gain:   {em_gain:.1f} ADU/e-", flush=True)

        # Read chunk sizes
        chunk_sizes = [struct.unpack('<I', f.read(4))[0] for _ in range(n_frames)]

        if args.info:
            total = sum(chunk_sizes)
            print(f"\n  Compressed: {total / 1024**2:.1f} MB", flush=True)
            print(f"  Avg/frame:  {total / n_frames / 1024:.1f} KB", flush=True)
            return

        print(f"\nWriting {args.output}...", flush=True)
        out = SerWriter(args.output)
        out.set_sizes(width, height, 2)  # 16-bit output

        for i in range(n_frames):
            compressed = f.read(chunk_sizes[i])
            raw = zlib.decompress(compressed)

            if mode == 0:  # lossless
                # Stored as int16, convert to uint16 with offset
                frame_int = np.frombuffer(raw, dtype=np.int16).reshape(height, width)
                # Add offset to make non-negative (bias-subtracted values can be negative)
                frame = (frame_int.astype(np.int32) + 1000).clip(0, 65535).astype(np.uint16)
            elif mode == 1:  # binary
                frame = unpack_bits(raw, (height, width))
                if args.raw:
                    frame = frame.astype(np.uint16)
                else:
                    frame = (frame * 65535).astype(np.uint16)
            else:  # hybrid
                counts = np.frombuffer(raw, dtype=np.uint8).reshape(height, width)
                if args.raw:
                    # Raw photon counts (0, 1, 2, 3...)
                    frame = counts.astype(np.uint16)
                else:
                    # Scale for visualization (each photon = ~6500 ADU)
                    frame = (counts.astype(np.uint16) * 6500).clip(0, 65535)

            out.add_image(frame)

            if (i + 1) % 200 == 0:
                print(f"  {i+1}/{n_frames}...", flush=True)

        out.close()

    import os
    print(f"\n  Output: {os.path.getsize(args.output) / 1024**2:.1f} MB", flush=True)
    print("Done!", flush=True)


if __name__ == "__main__":
    main()
