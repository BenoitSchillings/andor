#!/bin/bash
# Emergency stop - kills UI and controls hardware directly
# Usage: ./stop.sh

echo "!!! EMERGENCY STOP !!!"

# Kill any process using the camera
echo "Killing camera processes..."
pkill -f "andor_ui.py" 2>/dev/null && echo "  Killed andor_ui.py" || echo "  No UI running"
pkill -f "python.*andor" 2>/dev/null

sleep 5

cd "$(dirname "$0")"

echo "Securing hardware..."
python3 << 'EOF'
print("Camera...")
try:
    from andor import AndorCamera
    cam = AndorCamera()
    cam.abort_acquisition()
    cam.close_shutter()
    cam.set_temperature(10)
    print("  Shutter closed, temp -> +10°C")
except Exception as e:
    print(f"  {e}")

print("Mount...")
try:
    from skyx import sky6RASCOMTele
    mount = sky6RASCOMTele()
    mount.stop()
    mount.park()
    print("  Stopped and parking")
except Exception as e:
    print(f"  {e}")
EOF

echo "Done"
