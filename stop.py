#!/usr/bin/env python3
"""Emergency stop - one command to trigger failsafe."""
import urllib.request
import sys

host = sys.argv[1] if len(sys.argv) > 1 else "localhost"
url = f"http://{host}:9999/stop"

print("!!! EMERGENCY STOP !!!")
try:
    with urllib.request.urlopen(url, timeout=10) as r:
        print(r.read().decode())
except Exception as e:
    print(f"ERROR: {e}")
    sys.exit(1)
