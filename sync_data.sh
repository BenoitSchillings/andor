#!/bin/bash
# Sync data files to remote server
# Uses rsync to only copy new/modified files

REMOTE_HOST="benoit-ser.tail60bb60.ts.net"
REMOTE_USER="${REMOTE_USER:-benoit}"
REMOTE_DIR="${REMOTE_DIR:-/data/andor}"

LOCAL_DIR="${1:-./data}"

echo "=============================================="
echo "Syncing data to ${REMOTE_USER}@${REMOTE_HOST}"
echo "=============================================="
echo "Local:  $LOCAL_DIR"
echo "Remote: $REMOTE_DIR"
echo ""

# Check if local directory exists
if [ ! -d "$LOCAL_DIR" ]; then
    echo "Error: Local directory '$LOCAL_DIR' not found"
    exit 1
fi

# Show what would be transferred (dry run)
echo "Checking files to sync..."
rsync -avz --progress --stats --dry-run \
    --exclude='*.tmp' \
    --exclude='*.partial' \
    "$LOCAL_DIR/" "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/"

echo ""
read -p "Proceed with sync? [y/N] " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Syncing..."
    rsync -avz --progress --stats \
        --exclude='*.tmp' \
        --exclude='*.partial' \
        "$LOCAL_DIR/" "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/"
    echo ""
    echo "Done!"
else
    echo "Cancelled."
fi
