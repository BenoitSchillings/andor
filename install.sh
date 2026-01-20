#!/bin/bash
#
# Andor iXon Camera Setup Script
# Installs dependencies and configures the system for camera access
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIB_DIR="$SCRIPT_DIR/lib"

echo "=============================================="
echo "Andor iXon Camera Setup"
echo "=============================================="
echo ""

# Check if running as root for system-wide changes
check_root() {
    if [ "$EUID" -ne 0 ]; then
        echo "Note: Some steps require root. You may be prompted for sudo password."
    fi
}

# Install system packages
install_system_packages() {
    echo ""
    echo "--- Installing system packages ---"

    if command -v apt-get &> /dev/null; then
        sudo apt-get update
        sudo apt-get install -y \
            python3 \
            python3-pip \
            python3-venv \
            libusb-1.0-0 \
            libusb-1.0-0-dev \
            libc6-i386 \
            lib32stdc++6
        echo "System packages installed."
    elif command -v dnf &> /dev/null; then
        sudo dnf install -y \
            python3 \
            python3-pip \
            libusb1 \
            libusb1-devel \
            glibc.i686 \
            libstdc++.i686
        echo "System packages installed."
    else
        echo "Warning: Unknown package manager. Please install manually:"
        echo "  - python3, python3-pip"
        echo "  - libusb-1.0"
        echo "  - 32-bit compatibility libraries (for 32-bit SDK if used)"
    fi
}

# Install Python packages
install_python_packages() {
    echo ""
    echo "--- Installing Python packages ---"

    pip3 install --user \
        numpy \
        matplotlib \
        scipy \
        PyQt5 \
        astropy

    echo "Python packages installed."
}

# Setup udev rules for USB camera access
setup_udev_rules() {
    echo ""
    echo "--- Setting up USB permissions (udev rules) ---"

    UDEV_RULE="/etc/udev/rules.d/99-andor.rules"

    # Andor USB vendor ID is 0x136e
    RULE_CONTENT='# Andor Technology cameras
SUBSYSTEM=="usb", ATTR{idVendor}=="136e", MODE="0666", GROUP="plugdev"
SUBSYSTEM=="usb_device", ATTR{idVendor}=="136e", MODE="0666", GROUP="plugdev"

# Andor iXon Ultra (alternative ID)
SUBSYSTEM=="usb", ATTR{idVendor}=="0547", MODE="0666", GROUP="plugdev"
'

    echo "$RULE_CONTENT" | sudo tee "$UDEV_RULE" > /dev/null

    # Reload udev rules
    sudo udevadm control --reload-rules
    sudo udevadm trigger

    # Add user to plugdev group if it exists
    if getent group plugdev > /dev/null; then
        sudo usermod -a -G plugdev "$USER" 2>/dev/null || true
        echo "Added $USER to plugdev group."
    fi

    echo "USB permissions configured."
    echo "Note: You may need to log out and back in for group changes to take effect."
}

# Setup library path
setup_library_path() {
    echo ""
    echo "--- Setting up library path ---"

    # Check if lib directory exists
    if [ ! -d "$LIB_DIR" ]; then
        echo "Warning: Library directory not found: $LIB_DIR"
        echo "Please ensure libandor.so is in $LIB_DIR"
        return
    fi

    # Check for libandor.so
    if [ ! -f "$LIB_DIR/libandor.so" ]; then
        echo "Warning: libandor.so not found in $LIB_DIR"
        echo "Please copy the Andor SDK library to $LIB_DIR"
        return
    fi

    # Create ld.so.conf.d entry
    LDCONF="/etc/ld.so.conf.d/andor.conf"
    echo "$LIB_DIR" | sudo tee "$LDCONF" > /dev/null
    sudo ldconfig

    echo "Library path configured: $LIB_DIR"

    # Also add to bashrc for LD_LIBRARY_PATH
    BASHRC="$HOME/.bashrc"
    if ! grep -q "ANDOR_LIB" "$BASHRC" 2>/dev/null; then
        echo "" >> "$BASHRC"
        echo "# Andor camera library" >> "$BASHRC"
        echo "export ANDOR_LIB=\"$LIB_DIR\"" >> "$BASHRC"
        echo "export LD_LIBRARY_PATH=\"\$ANDOR_LIB:\$LD_LIBRARY_PATH\"" >> "$BASHRC"
        echo "Added LD_LIBRARY_PATH to $BASHRC"
    fi
}

# Setup Andor configuration directory
setup_andor_config() {
    echo ""
    echo "--- Setting up Andor configuration ---"

    # Andor SDK looks for config in /usr/local/etc/andor or /etc/andor
    ANDOR_ETC="/usr/local/etc/andor"

    if [ ! -d "$ANDOR_ETC" ]; then
        sudo mkdir -p "$ANDOR_ETC"
    fi

    # Create detector.ini if it doesn't exist
    DETECTOR_INI="$ANDOR_ETC/detector.ini"
    if [ ! -f "$DETECTOR_INI" ]; then
        echo "[system]" | sudo tee "$DETECTOR_INI" > /dev/null
        echo "path=$LIB_DIR" | sudo tee -a "$DETECTOR_INI" > /dev/null
        echo "Created $DETECTOR_INI"
    fi

    # Link firmware files if they exist
    if [ -d "$LIB_DIR" ]; then
        for f in "$LIB_DIR"/*.cof "$LIB_DIR"/*.RBF "$LIB_DIR"/*.hex 2>/dev/null; do
            if [ -f "$f" ]; then
                fname=$(basename "$f")
                if [ ! -f "$ANDOR_ETC/$fname" ]; then
                    sudo ln -sf "$f" "$ANDOR_ETC/$fname" 2>/dev/null || true
                fi
            fi
        done
        echo "Firmware files linked."
    fi
}

# Verify installation
verify_installation() {
    echo ""
    echo "--- Verifying installation ---"

    # Check Python
    echo -n "Python: "
    python3 --version

    # Check required packages
    echo -n "NumPy: "
    python3 -c "import numpy; print(numpy.__version__)" 2>/dev/null || echo "NOT INSTALLED"

    echo -n "Matplotlib: "
    python3 -c "import matplotlib; print(matplotlib.__version__)" 2>/dev/null || echo "NOT INSTALLED"

    echo -n "SciPy: "
    python3 -c "import scipy; print(scipy.__version__)" 2>/dev/null || echo "NOT INSTALLED"

    echo -n "PyQt5: "
    python3 -c "from PyQt5 import QtCore; print(QtCore.PYQT_VERSION_STR)" 2>/dev/null || echo "NOT INSTALLED"

    # Check library
    echo -n "libandor.so: "
    if [ -f "$LIB_DIR/libandor.so" ]; then
        echo "Found in $LIB_DIR"
    else
        echo "NOT FOUND"
    fi

    # Check if camera is accessible
    echo -n "USB Camera: "
    if lsusb | grep -qi "andor\|136e\|0547"; then
        echo "Detected"
    else
        echo "Not detected (camera may be off or not connected)"
    fi
}

# Create convenience scripts
create_scripts() {
    echo ""
    echo "--- Creating convenience scripts ---"

    # Create run script
    cat > "$SCRIPT_DIR/run.sh" << 'EOF'
#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LD_LIBRARY_PATH="$SCRIPT_DIR/lib:$LD_LIBRARY_PATH"
cd "$SCRIPT_DIR"
python3 andor_ui.py "$@"
EOF
    chmod +x "$SCRIPT_DIR/run.sh"
    echo "Created run.sh"

    # Create info script
    cat > "$SCRIPT_DIR/camera_info.sh" << 'EOF'
#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LD_LIBRARY_PATH="$SCRIPT_DIR/lib:$LD_LIBRARY_PATH"
cd "$SCRIPT_DIR"
python3 info.py "$@"
EOF
    chmod +x "$SCRIPT_DIR/camera_info.sh"
    echo "Created camera_info.sh"
}

# Print summary
print_summary() {
    echo ""
    echo "=============================================="
    echo "Setup Complete"
    echo "=============================================="
    echo ""
    echo "To run the camera UI:"
    echo "  cd $SCRIPT_DIR"
    echo "  ./run.sh"
    echo ""
    echo "To get camera info:"
    echo "  ./camera_info.sh"
    echo ""
    echo "If you get permission errors:"
    echo "  1. Log out and log back in (for group changes)"
    echo "  2. Unplug and replug the camera"
    echo ""
    echo "If library not found:"
    echo "  export LD_LIBRARY_PATH=\"$LIB_DIR:\$LD_LIBRARY_PATH\""
    echo ""
}

# Main
main() {
    check_root
    install_system_packages
    install_python_packages
    setup_udev_rules
    setup_library_path
    setup_andor_config
    create_scripts
    verify_installation
    print_summary
}

# Run with options
case "${1:-}" in
    --packages-only)
        install_system_packages
        install_python_packages
        ;;
    --udev-only)
        setup_udev_rules
        ;;
    --verify)
        verify_installation
        ;;
    --help|-h)
        echo "Usage: $0 [option]"
        echo ""
        echo "Options:"
        echo "  (none)           Full installation"
        echo "  --packages-only  Install packages only"
        echo "  --udev-only      Setup USB permissions only"
        echo "  --verify         Verify installation"
        echo "  --help           Show this help"
        ;;
    *)
        main
        ;;
esac
