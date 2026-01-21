#!/bin/bash
# Automated capture script
# Usage: ./capture.sh <target_name> [frames] [exp] [gain]
#
# Example:
#   ./capture.sh m31_field 15000 0.01 1000

TARGET=${1:-"capture"}
FRAMES=${2:-10000}
EXP=${3:-0.01}
GAIN=${4:-1000}
TEMP=${5:--60}

echo "=================================="
echo "Automated Capture"
echo "=================================="
echo "Target:      $TARGET"
echo "Frames:      $FRAMES"
echo "Exposure:    ${EXP}s"
echo "EM Gain:     $GAIN"
echo "Temperature: ${TEMP}°C"
echo "=================================="

cd "$(dirname "$0")"

python3 andor_ui.py \
    -f "${TARGET}_" \
    -exp "$EXP" \
    -gain "$GAIN" \
    -temp "$TEMP" \
    --capture "$FRAMES" \
    --auto-start \
    --wait-temp \
    --ser

echo "Capture complete!"
