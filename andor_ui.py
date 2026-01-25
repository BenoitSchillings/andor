#!/usr/bin/env python3
"""
Andor iXon Ultra 888 Camera UI

A PyQt5/pyqtgraph-based interface for the Andor EMCCD camera.

Features:
- Live view with auto-stretch
- Zoom window tracking brightest star
- Crosshair and bullseye overlays
- Temperature monitoring
- EM gain control
- Exposure control
- FITS file capture

Usage:
    python andor_ui.py -exp 0.1 -gain 100
"""

import numpy as np
import time
from datetime import datetime
import argparse
import sys
import json
import configparser

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    from scipy.ndimage import zoom

# PyQt5 must be imported and QApplication created BEFORE pyqtgraph
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import QMenu, QMenuBar, QAction, QMainWindow, QStatusBar
from PyQt5.QtCore import QObject, QThread, pyqtSignal, QTimer, QPoint
from PyQt5.QtGui import QImage, QPixmap

# Create QApplication early (required before pyqtgraph import)
_app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

import pyqtgraph as pg

from andor import (
    get_camera, AndorCamera, SimulatedCamera,
    AcquisitionMode, ReadMode, TriggerMode,
    SpuriousNoiseFilterMode, ErrorCode
)
from session import SessionManager
import json
from pathlib import Path

CONFIG_FILE = Path.home() / ".andor_ui_config.json"
CALIBRATION_FILE = Path(__file__).parent / "calibration.json"

try:
    from astropy.io import fits
    HAS_ASTROPY = True
except ImportError:
    HAS_ASTROPY = False


def load_calibration_config():
    """Load calibration file paths from config."""
    config = {"dark_bias": "", "flat_field": "", "flat_bias": ""}
    try:
        if CALIBRATION_FILE.exists():
            with open(CALIBRATION_FILE, 'r') as f:
                config.update(json.load(f))
    except Exception as e:
        print(f"Could not load calibration config: {e}")
    return config


def load_calibration_frame(filepath):
    """Load a FITS calibration frame, returns None if not available."""
    if not filepath or not HAS_ASTROPY:
        return None
    try:
        path = Path(filepath)
        if path.exists():
            data = fits.getdata(filepath)
            print(f"Loaded calibration frame: {filepath} ({data.shape})")
            return data.astype(np.float32)
    except Exception as e:
        print(f"Could not load calibration frame {filepath}: {e}")
    return None

from util import (
    HighValueFinder,
    fit_gauss_circular,
    compute_hfd,
    filter_outliers_simple
)
from ser import SerWriter
from skyx import sky6RASCOMTele, SkyxConnectionError
from scipy.ndimage import gaussian_filter

# Focuser support (optional)
try:
    from fli_focuser import focuser as FLIFocuser
    HAS_FOCUSER = True
except ImportError:
    HAS_FOCUSER = False
except Exception as e:
    print(f"Focuser not available: {e}")
    HAS_FOCUSER = False


# ============================================================================
# Camera Worker Thread
# ============================================================================

HS = 1 

class CameraWorker(QObject):
    """Worker object that handles camera operations in a separate thread."""

    new_frame_ready = pyqtSignal(object)
    exposure_started = pyqtSignal(float)
    temperature_updated = pyqtSignal(float, float, float, float, str)  # sensor, target, ambient, volts, status

    def __init__(self, camera):
        super().__init__()
        self.camera = camera
        self.running = False
        self.exposure_time = 0.01
        self.em_gain = 100
        self.video_mode = True
        self.restart_needed = False

    def start_capture(self):
        """Start the continuous frame capture loop."""
        self.running = True
        self._last_frame_time = 0

        if self.video_mode:
            # Video mode: continuous acquisition
            self.camera.setup_video_mode()
            self.camera.set_exposure_time(self.exposure_time)
            self.camera.start_acquisition()

            last_image_index = 0
            while self.running:
                # Check if there are new images
                try:
                    first, last = self.camera.get_number_new_images()
                    if last > last_image_index:
                        # New frame available
                        frame = self.camera.get_latest_frame()
                        if frame is not None:
                            self.new_frame_ready.emit(frame)
                            last_image_index = last
                        time.sleep(0.001)
                    else:
                        # No new frame, wait a bit
                        time.sleep(0.001)
                except Exception as e:
                    # Log and fallback if get_number_new_images fails
                    print(f"Frame error: {e}")
                    time.sleep(0.01)

                # Update temperature periodically
                if hasattr(self, '_temp_counter'):
                    self._temp_counter += 1
                else:
                    self._temp_counter = 0

                if self._temp_counter % 50 == 0:
                    sensor, target, ambient, volts, status = self.camera.get_temperature_status()
                    status_map = {
                        ErrorCode.DRV_TEMPERATURE_OFF: "Off",
                        ErrorCode.DRV_TEMPERATURE_STABILIZED: "Stabilized",
                        ErrorCode.DRV_TEMPERATURE_NOT_STABILIZED: "Cooling...",
                        ErrorCode.DRV_TEMPERATURE_NOT_REACHED: "Cooling...",
                        ErrorCode.DRV_TEMPERATURE_DRIFT: "Drift",
                    }
                    self.temperature_updated.emit(sensor, target, ambient, volts, status_map.get(status, "Unknown"))

                # Check if we need to restart acquisition (e.g., exposure or gain changed)
                if self.restart_needed:
                    self.restart_needed = False
                    print(f"Restarting acquisition: exp={self.exposure_time:.4f}s, gain={self.em_gain}")
                    self.camera.abort_acquisition()
                    # Apply same settings as setup_video_mode()
                    self.camera.set_exposure_time(self.exposure_time)
                    self.camera.set_output_amplifier(0)  # EM amplifier
                    self.camera.set_hs_speed(HS, 0)  # index=1 (20 MHz), EM amp
                    self.camera.set_em_advanced(True)
                    self.camera.set_em_gain_mode(1)  # 12-bit DAC (0-4095)
                    # Clamp gain to valid range
                    low, high = self.camera.get_em_gain_range()
                    gain = max(low, min(high, self.em_gain))
                    if gain != self.em_gain:
                        print(f"Clamped gain {self.em_gain} to {gain} (range {low}-{high})")
                    self.camera.set_emccd_gain(gain)
                    self.camera.set_preamp_gain(1)  # 2x preamp
                    self.camera.set_vs_speed(1)  # VS speed index 1
                    self.camera.start_acquisition()
                    last_image_index = 0  # Reset so we don't wait for old frame count

            self.camera.abort_acquisition()
        else:
            # Single frame mode
            while self.running:
                self.exposure_started.emit(self.exposure_time)
                self.camera.setup_single_scan()
                self.camera.set_exposure_time(self.exposure_time)
                self.camera.start_acquisition()
                self.camera.wait_for_acquisition()

                if self.running:
                    frame = self.camera.get_acquired_data(dtype="uint16")
                    self.new_frame_ready.emit(frame)

                    # Update temperature
                    sensor, target, ambient, volts, status = self.camera.get_temperature_status()
                    status_map = {
                        ErrorCode.DRV_TEMPERATURE_OFF: "Off",
                        ErrorCode.DRV_TEMPERATURE_STABILIZED: "Stabilized",
                        ErrorCode.DRV_TEMPERATURE_NOT_STABILIZED: "Cooling...",
                        ErrorCode.DRV_TEMPERATURE_NOT_REACHED: "Cooling...",
                        ErrorCode.DRV_TEMPERATURE_DRIFT: "Drift",
                    }
                    self.temperature_updated.emit(sensor, target, ambient, volts, status_map.get(status, "Unknown"))

    def stop_capture(self):
        """Stop the frame capture loop."""
        self.running = False


# ============================================================================
# Custom Widgets
# ============================================================================

class ExposureProgressBar(QtWidgets.QProgressBar):
    """Progress bar that shows exposure time."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.video_mode = False
        self.setRange(0, 100)
        self.setValue(0)

    def text(self):
        if self.video_mode:
            return "Video mode"
        if self.maximum() == 0:
            return "Ready"
        current_sec = self.value() / 10.0
        max_sec = self.maximum() / 10.0
        return f"Exposure: {current_sec:.1f} / {max_sec:.1f} sec"


# ============================================================================
# Main Window
# ============================================================================

class AndorWindow(QMainWindow):
    """Main window for the Andor camera UI."""

    def __init__(self, ui_controller, camera_name="Andor iXon", parent=None):
        super().__init__(parent)
        self.ui_controller = ui_controller
        self.quit = False
        self._create_menu_bar()
        self.setWindowTitle(camera_name)

    def _create_menu_bar(self):
        menu_bar = QMenuBar(self)
        self.setMenuBar(menu_bar)

        # Action menu
        action_menu = menu_bar.addMenu('Action')

        auto_level_action = QAction('Auto-Level', self)
        auto_level_action.setShortcut('Ctrl+A')
        auto_level_action.triggered.connect(self.on_auto_level)
        action_menu.addAction(auto_level_action)

        denoise_action = QAction('Toggle CIC Filter', self)
        denoise_action.setShortcut('Ctrl+N')
        denoise_action.triggered.connect(self.on_denoise)
        action_menu.addAction(denoise_action)

        crosshair_action = QAction('Toggle Crosshair', self)
        crosshair_action.setShortcut('Ctrl+C')
        crosshair_action.triggered.connect(self.on_toggle_crosshair)
        action_menu.addAction(crosshair_action)

        bullseye_action = QAction('Toggle Bullseye', self)
        bullseye_action.setShortcut('Ctrl+B')
        bullseye_action.triggered.connect(self.on_toggle_bullseye)
        action_menu.addAction(bullseye_action)

        hotpixel_action = QAction('Toggle Hot Pixel Filter', self)
        hotpixel_action.setShortcut('Ctrl+H')
        hotpixel_action.triggered.connect(self.on_toggle_hotpixel)
        action_menu.addAction(hotpixel_action)

        average_action = QAction('Toggle Average View (20 frames)', self)
        average_action.setShortcut('Ctrl+G')
        average_action.triggered.connect(self.on_toggle_average)
        action_menu.addAction(average_action)

        action_menu.addSeparator()

        dark_action = QAction('Toggle Dark Subtract', self)
        dark_action.setShortcut('D')
        dark_action.triggered.connect(self.on_toggle_dark)
        action_menu.addAction(dark_action)

        flat_action = QAction('Toggle Flat Divide', self)
        flat_action.setShortcut('F')
        flat_action.triggered.connect(self.on_toggle_flat)
        action_menu.addAction(flat_action)

        action_menu.addSeparator()

        capture_action = QAction('Start/Stop Capture', self)
        capture_action.setShortcut('Space')
        capture_action.triggered.connect(self.on_capture)
        action_menu.addAction(capture_action)

        # Camera menu
        camera_menu = menu_bar.addMenu('Camera')

        cooler_action = QAction('Toggle Cooler', self)
        cooler_action.setShortcut('Ctrl+T')
        cooler_action.triggered.connect(self.on_toggle_cooler)
        camera_menu.addAction(cooler_action)

        cooler_shutdown_action = QAction('Cooler Shutdown (gradual)', self)
        cooler_shutdown_action.setShortcut('Ctrl+Shift+T')
        cooler_shutdown_action.triggered.connect(self.on_cooler_shutdown)
        camera_menu.addAction(cooler_shutdown_action)

    def on_auto_level(self):
        self.ui_controller.auto_level = True

    def on_denoise(self):
        self.ui_controller.toggle_denoise()

    def on_toggle_crosshair(self):
        self.ui_controller.toggle_crosshair()

    def on_toggle_bullseye(self):
        self.ui_controller.toggle_bullseye()

    def on_toggle_cooler(self):
        self.ui_controller.toggle_cooler()

    def on_cooler_shutdown(self):
        self.ui_controller.start_cooler_shutdown()

    def on_toggle_hotpixel(self):
        self.ui_controller.toggle_hotpixel()

    def on_toggle_average(self):
        self.ui_controller.toggle_average()

    def on_toggle_dark(self):
        self.ui_controller.toggle_dark()

    def on_toggle_flat(self):
        self.ui_controller.toggle_flat()

    def on_capture(self):
        self.ui_controller.toggle_capture()

    def closeEvent(self, event):
        self.quit = True
        self.ui_controller._cleanup()
        super().closeEvent(event)


# ============================================================================
# Main UI Controller
# ============================================================================

class AndorUI:
    """Main UI controller for the Andor camera."""

    def __init__(self, camera, args):
        self.camera = camera
        self.args = args

        # Image dimensions
        self.sx = camera.width
        self.sy = camera.height

        # State
        self.idx = 0
        self.cnt = 0
        self.capture_state = False
        self.auto_level = False
        self.denoise = False
        self.show_crosshair = False
        self.show_bullseye = False
        self.pos = QPoint(self.sx // 2, self.sy // 2)
        self.array = np.zeros((self.sy, self.sx), dtype=np.uint16)
        self.frame_per_file = args.count
        self.EDGE = 64
        self._pending_frame = None
        self._display_scheduled = False
        self._first_frame = True
        self._display_count = 0
        self.filter_hot_pixels = False
        self.save_as_ser = False  # False = FITS, True = SER
        self.ser_writer = None
        self.shutdown_in_progress = False
        self.shutdown_timer = None
        self.cooler_shutdown_complete = False  # Track if we did gradual shutdown

        # Batch capture mode
        self.batch_capture_target = args.capture if hasattr(args, 'capture') else 0
        self.batch_frames_captured = 0
        self.batch_auto_start = args.auto_start if hasattr(args, 'auto_start') else False
        self.batch_wait_temp = args.wait_temp if hasattr(args, 'wait_temp') else False
        self.batch_temp_ready = False

        # Session management
        data_dir = getattr(args, 'data_dir', '.')
        self.session = SessionManager(data_dir)
        self.session_active = False

        # Frame averaging
        self.show_average = False  # Toggle between live and averaged view
        self.avg_frame_count = 20  # Number of frames to average
        self._frame_buffer = []  # Circular buffer for averaging

        # Telescope/mount control via TheSkyX
        try:
            self.scope = sky6RASCOMTele()
            self.scope.Connect()
            print("Scope: connected to mount")
        except SkyxConnectionError as e:
            self.scope = None
            print(f"Scope: Failed to connect to TheSkyX: {e}")
        except Exception as e:
            self.scope = None
            print("Scope: not available")

        # Load calibration frames (dark/bias, flat, flat bias)
        # Command line overrides config file
        cal_config = load_calibration_config()
        dark_file = args.dark_file if hasattr(args, 'dark_file') and args.dark_file else cal_config.get("dark_bias")
        flat_file = args.flat_file if hasattr(args, 'flat_file') and args.flat_file else cal_config.get("flat_field")
        self.cal_dark = load_calibration_frame(dark_file)
        self.cal_flat = load_calibration_frame(flat_file)
        self.cal_flat_bias = load_calibration_frame(cal_config.get("flat_bias"))

        # Prepare normalized flat if available
        if self.cal_flat is not None:
            flat = self.cal_flat.copy()
            if self.cal_flat_bias is not None:
                flat = flat - self.cal_flat_bias
            flat_mean = np.mean(flat)
            if flat_mean > 0:
                self.cal_flat_norm = flat / flat_mean
            else:
                self.cal_flat_norm = None
        else:
            self.cal_flat_norm = None

        # Calibration apply flags (command line can override defaults)
        # Default: enable if calibration frame is available
        self.apply_dark = self.cal_dark is not None
        self.apply_flat = self.cal_flat_norm is not None
        # Command line overrides
        if hasattr(args, 'dark') and args.dark:
            self.apply_dark = True
        if hasattr(args, 'no_dark') and args.no_dark:
            self.apply_dark = False
        if hasattr(args, 'flat') and args.flat:
            self.apply_flat = True
        if hasattr(args, 'no_flat') and args.no_flat:
            self.apply_flat = False

        # Load saved config (target temp, cooler state)
        self._load_config()

        # Star finder with tracking (avoids full-image scan each frame)
        self.star_finder = HighValueFinder(search_box_size=64, blur_size=5)

        # Stats
        self.fwhm = 0.0
        self.hfd = 0.0
        self.rms = 0.0
        self.min_val = 0
        self.max_val = 0
        self.fps = 0.0
        self.temperature = 20.0
        self.temp_status = "Off"

        # Create main window
        self.win = AndorWindow(self, camera.head_model)
        # Size window for 1:1 image display + dock area
        self.win.resize(self.sx + 100, self.sy + 280)

        # Create image view
        self.imv = pg.ImageView()
        self.imv.setImage(self.array)
        self.imv.getImageItem().setAutoDownsample(active=True)
        # Lock aspect ratio for 1:1 pixel display
        self.imv.getView().setAspectLocked(True, ratio=1.0)
        self.imv.getView().setRange(xRange=(0, self.sx), yRange=(0, self.sy), padding=0)
        # Set histogram range for 16-bit data
        self.imv.setHistogramRange(0, 65535)
        self.imv.setLevels(0, 65535)

        # Setup overlays
        self._setup_overlays()

        self.win.setCentralWidget(self.imv)

        # Create status bar with controls
        self._setup_status_bar()

        # Setup click handler
        self.imv.getImageItem().mouseClickEvent = self.on_click

        # Setup worker thread
        self._setup_worker()

        # Timing
        self.t0 = time.perf_counter()

        # Restore cooler state from saved config
        self._restore_cooler_state()

        # Apply command line display options
        self._apply_startup_options()

        self.win.show()

    def _setup_overlays(self):
        """Setup crosshair and bullseye overlays."""
        vb = self.imv.getView()

        # Crosshair
        self.crosshair_v = pg.InfiniteLine(angle=90, movable=False, pen='g')
        self.crosshair_h = pg.InfiniteLine(angle=0, movable=False, pen='g')
        self.crosshair_v.setPos(self.sx / 2)
        self.crosshair_h.setPos(self.sy / 2)
        self.crosshair_v.setVisible(False)
        self.crosshair_h.setVisible(False)
        vb.addItem(self.crosshair_v)
        vb.addItem(self.crosshair_h)

        # Bullseye circles (red)
        self.bullseye_items = []
        radii = [20, 40, 60, 80, 100]
        red_pen = pg.mkPen('r', width=1)
        for r in radii:
            circle = QtWidgets.QGraphicsEllipseItem(-r, -r, r * 2, r * 2)
            circle.setPen(red_pen)
            circle.setPos(self.sx / 2, self.sy / 2)
            circle.setVisible(False)
            self.bullseye_items.append(circle)
            vb.addItem(circle)

        # Bullseye cross (red)
        cross_size = 100
        cx, cy = self.sx / 2, self.sy / 2
        self.bullseye_cross_v = pg.InfiniteLine(angle=90, movable=False, pen=red_pen)
        self.bullseye_cross_h = pg.InfiniteLine(angle=0, movable=False, pen=red_pen)
        self.bullseye_cross_v.setPos(cx)
        self.bullseye_cross_h.setPos(cy)
        self.bullseye_cross_v.setVisible(False)
        self.bullseye_cross_h.setVisible(False)
        vb.addItem(self.bullseye_cross_v)
        vb.addItem(self.bullseye_cross_h)

    def _setup_status_bar(self):
        """Setup status bar with controls."""
        self.statusBar = QStatusBar()

        # Left side: Zoom view
        left_widget = QtWidgets.QWidget()
        left_widget.setLayout(QtWidgets.QHBoxLayout())
        left_widget.setFixedSize(300, 200)

        self.zoom_view = QtWidgets.QLabel()
        self.zoom_view.setFixedSize(256, 256)
        left_widget.layout().addWidget(self.zoom_view)

        self.statusBar.addPermanentWidget(left_widget, 0)

        # Middle: Camera controls
        mid_widget = QtWidgets.QWidget()
        mid_widget.setLayout(QtWidgets.QFormLayout())
        mid_widget.setFixedSize(250, 200)

        # Exposure control
        self.exp_spinbox = QtWidgets.QDoubleSpinBox()
        self.exp_spinbox.setRange(0.0001, 300.0)
        self.exp_spinbox.setValue(self.args.exp)
        self.exp_spinbox.setSuffix(" sec")
        self.exp_spinbox.setDecimals(4)
        self.exp_spinbox.setSingleStep(0.001)
        self.exp_spinbox.valueChanged.connect(self.on_exposure_changed)
        mid_widget.layout().addRow("Exposure:", self.exp_spinbox)

        # EM Gain control
        gain_widget = QtWidgets.QWidget()
        gain_layout = QtWidgets.QHBoxLayout(gain_widget)
        gain_layout.setContentsMargins(0, 0, 0, 0)

        self.gain_spinbox = QtWidgets.QSpinBox()
        self.gain_spinbox.setRange(1, 3964)  # Mode 1: 12-bit DAC
        self.gain_spinbox.setValue(self.args.gain)
        self.gain_spinbox.valueChanged.connect(self.on_gain_changed)
        gain_layout.addWidget(self.gain_spinbox)

        self.advanced_checkbox = QtWidgets.QCheckBox("Adv")
        self.advanced_checkbox.setToolTip("Enable advanced mode (EM gain up to 1000)")
        self.advanced_checkbox.setChecked(True)  # Enable advanced by default
        self.advanced_checkbox.toggled.connect(self.on_advanced_toggled)
        gain_layout.addWidget(self.advanced_checkbox)

        mid_widget.layout().addRow("EM Gain:", gain_widget)

        # Temperature display
        self.temp_label = QtWidgets.QLabel("-- °C")
        mid_widget.layout().addRow("Temp:", self.temp_label)

        # Cooler button - initialize to actual state
        self.cooler_is_on = self.camera.is_cooler_on()
        self.cooler_button = QtWidgets.QPushButton("Cooler ON" if self.cooler_is_on else "Cooler OFF")
        self.cooler_button.clicked.connect(self.toggle_cooler)
        mid_widget.layout().addRow("", self.cooler_button)

        # Target temperature
        self.target_temp_spinbox = QtWidgets.QSpinBox()
        temp_range = self.camera.get_temperature_range()
        self.target_temp_spinbox.setRange(temp_range[0], temp_range[1])
        self.target_temp_spinbox.setValue(-60)
        self.target_temp_spinbox.setSuffix(" °C")
        self.target_temp_spinbox.valueChanged.connect(self.on_target_temp_changed)
        mid_widget.layout().addRow("Target:", self.target_temp_spinbox)

        self.statusBar.addPermanentWidget(mid_widget, 0)

        # Right side: Capture controls and stats
        right_widget = QtWidgets.QWidget()
        right_widget.setLayout(QtWidgets.QVBoxLayout())
        right_widget.setFixedSize(400, 220)

        # Target and Filter row
        target_row = QtWidgets.QWidget()
        target_layout = QtWidgets.QHBoxLayout(target_row)
        target_layout.setContentsMargins(0, 0, 0, 0)
        target_layout.addWidget(QtWidgets.QLabel("Target:"))
        self.target_edit = QtWidgets.QLineEdit(getattr(self.args, 'target', 'noname_target'))
        self.target_edit.setPlaceholderText("e.g., M42")
        self.target_edit.setFixedWidth(100)
        target_layout.addWidget(self.target_edit)
        target_layout.addWidget(QtWidgets.QLabel("Filter:"))
        self.filter_edit = QtWidgets.QLineEdit(getattr(self.args, 'filter', ''))
        self.filter_edit.setPlaceholderText("e.g., Ha")
        self.filter_edit.setFixedWidth(60)
        target_layout.addWidget(self.filter_edit)
        target_layout.addStretch()
        right_widget.layout().addWidget(target_row)

        # Data directory with disk space
        data_row = QtWidgets.QWidget()
        data_layout = QtWidgets.QHBoxLayout(data_row)
        data_layout.setContentsMargins(0, 0, 0, 0)
        data_layout.addWidget(QtWidgets.QLabel("Data:"))
        self.data_dir_edit = QtWidgets.QLineEdit(getattr(self.args, 'data_dir', '.'))
        self.data_dir_edit.setFixedWidth(150)
        data_layout.addWidget(self.data_dir_edit)
        self.disk_space_label = QtWidgets.QLabel("Disk: --")
        data_layout.addWidget(self.disk_space_label)
        data_layout.addStretch()
        right_widget.layout().addWidget(data_row)

        # Format selection
        format_widget = QtWidgets.QWidget()
        format_layout = QtWidgets.QHBoxLayout(format_widget)
        format_layout.setContentsMargins(0, 0, 0, 0)
        self.format_combo = QtWidgets.QComboBox()
        self.format_combo.addItems(["FITS (individual)", "SER (video)"])
        self.format_combo.currentIndexChanged.connect(self.on_format_changed)
        format_layout.addWidget(QtWidgets.QLabel("Format:"))
        format_layout.addWidget(self.format_combo)
        right_widget.layout().addWidget(format_widget)

        # Exposure progress bar
        self.exposure_progress = ExposureProgressBar()
        self.exposure_progress.setTextVisible(True)
        right_widget.layout().addWidget(self.exposure_progress)

        # Capture button
        self.capture_button = QtWidgets.QPushButton("Start Capture")
        self.capture_button.clicked.connect(self.toggle_capture)
        right_widget.layout().addWidget(self.capture_button)

        # Stats labels
        self.stats_label1 = QtWidgets.QLabel("FWHM: -- HFD: -- Min: -- Max: --")
        right_widget.layout().addWidget(self.stats_label1)

        self.stats_label2 = QtWidgets.QLabel("RMS: -- FPS: -- Frame: 0")
        right_widget.layout().addWidget(self.stats_label2)

        self.scope_label = QtWidgets.QLabel("Scope: --")
        right_widget.layout().addWidget(self.scope_label)

        self.statusBar.addPermanentWidget(right_widget, 1)

        # Timer for scope updates (every 2 seconds)
        self._scope_update_counter = 0

        self.win.setStatusBar(self.statusBar)

    def _setup_worker(self):
        """Setup camera worker thread."""
        self.thread = QThread()
        self.worker = CameraWorker(self.camera)
        self.worker.exposure_time = self.args.exp
        self.worker.em_gain = self.args.gain
        self.worker.video_mode = self.args.exp < 1.0
        self.exposure_progress.video_mode = self.worker.video_mode
        self.worker.moveToThread(self.thread)

        self.worker.new_frame_ready.connect(self.handle_new_frame)
        self.worker.exposure_started.connect(self.start_exposure_timer)
        self.worker.temperature_updated.connect(self.update_temperature)
        self.thread.started.connect(self.worker.start_capture)

        # Exposure timer for progress bar
        self.exposure_timer = QTimer()
        self.exposure_timer.timeout.connect(self.update_exposure_progress)

        # Initialize EM gain mode and value before starting
        self.camera.set_em_advanced(True)
        self.camera.set_em_gain_mode(1)  # 12-bit DAC (0-4095)
        # Query actual gain range from camera and update spinbox
        low, high = self.camera.get_em_gain_range()
        print(f"[CAM] EM Gain Range: {low}-{high}")
        self.gain_spinbox.setRange(low, high)
        # Clamp initial gain to valid range
        initial_gain = min(max(self.gain_spinbox.value(), low), high)
        self.gain_spinbox.setValue(initial_gain)
        self.camera.set_emccd_gain(initial_gain)
        print(f"Initial settings: exp={self.args.exp:.4f}s, gain={initial_gain}, mode=Real EM")

        self.thread.start()

    def on_click(self, event):
        """Handle mouse click on image."""
        event.accept()
        self.pos = event.pos()
        print(f"Click: ({int(self.pos.x())}, {int(self.pos.y())})")

    def on_exposure_changed(self, value):
        """Handle exposure time change."""
        print(f"Exposure: {value:.4f} sec")
        self.worker.exposure_time = value
        self.worker.restart_needed = True

    def on_gain_changed(self, value):
        """Handle EM gain change."""
        print(f"EM Gain: {value}")
        self.worker.em_gain = value
        self.worker.restart_needed = True

    def on_advanced_toggled(self, checked):
        """Handle advanced mode toggle."""
        print(f"EM Advanced: {checked}")
        self.camera.set_em_advanced(checked)
        # Query actual gain range from camera
        low, high = self.camera.get_em_gain_range()
        print(f"[CAM] EM Gain Range: {low}-{high}")
        self.gain_spinbox.setRange(low, high)

    def on_target_temp_changed(self, value):
        """Handle target temperature change."""
        print(f"Target Temp: {value}°C")
        self.camera.set_temperature(value)

    def toggle_cooler(self):
        """Toggle cooler on/off."""
        if self.cooler_is_on:
            self.camera.cooler_off()
            self.cooler_is_on = False
            self.cooler_button.setText("Cooler OFF")
            print("[UI] Cooler OFF")
        else:
            target = self.target_temp_spinbox.value()
            self.camera.set_temperature(target)
            self.camera.cooler_on()
            self.cooler_is_on = True
            self.cooler_button.setText("Cooler ON")
            print(f"[UI] Cooler ON, target {target}°C")
        # Update gain range (temperature-dependent)
        self._update_gain_range()

    def _update_gain_range(self):
        """Update gain spinbox range from camera."""
        try:
            em_adv = self.camera.get_em_advanced()
            print(f"[UI] EM Advanced: {em_adv}")
        except Exception:
            pass
        low, high = self.camera.get_em_gain_range()
        print(f"[UI] EM Gain range: {low}-{high}")
        current = self.gain_spinbox.value()
        self.gain_spinbox.setRange(low, high)
        # Clamp current value to new range
        if current > high:
            self.gain_spinbox.setValue(high)
        elif current < low:
            self.gain_spinbox.setValue(low)

    def start_cooler_shutdown(self):
        """Start gradual cooler shutdown procedure."""
        if self.shutdown_in_progress:
            print("Shutdown already in progress")
            return

        if not self.cooler_is_on:
            print("Cooler is already off")
            return

        self.shutdown_in_progress = True
        self.cooler_button.setText("Shutting down...")
        print("Starting gradual cooler shutdown (+10°C every 30s)")

        # Start the shutdown timer
        self.shutdown_timer = QTimer()
        self.shutdown_timer.timeout.connect(self._cooler_shutdown_step)
        self._cooler_shutdown_step()  # Do first step immediately
        self.shutdown_timer.start(30000)  # 30 seconds

    def _cooler_shutdown_step(self):
        """Execute one step of the cooler shutdown."""
        current_temp, _ = self.camera.get_temperature()
        new_target = current_temp + 10

        if new_target >= 20:
            # Reached ambient, turn off cooler
            self.shutdown_timer.stop()
            self.shutdown_timer = None
            self.camera.cooler_off()
            self.cooler_is_on = False
            self.cooler_button.setText("Cooler OFF")
            self.shutdown_in_progress = False
            self.cooler_shutdown_complete = True  # Mark that we did proper shutdown
            print(f"Cooler shutdown complete at {current_temp:.1f}°C")
        else:
            # Raise temperature by 10°C
            self.camera.set_temperature(int(new_target))
            print(f"Shutdown: {current_temp:.1f}°C → {new_target:.0f}°C target")

    def toggle_crosshair(self):
        """Toggle crosshair overlay."""
        self.show_crosshair = not self.show_crosshair
        self.crosshair_v.setVisible(self.show_crosshair)
        self.crosshair_h.setVisible(self.show_crosshair)

    def toggle_bullseye(self):
        """Toggle bullseye overlay (circles + cross)."""
        self.show_bullseye = not self.show_bullseye
        for item in self.bullseye_items:
            item.setVisible(self.show_bullseye)
        self.bullseye_cross_v.setVisible(self.show_bullseye)
        self.bullseye_cross_h.setVisible(self.show_bullseye)

    def toggle_denoise(self):
        """Toggle CIC filter."""
        self.denoise = not self.denoise
        if self.denoise:
            self.camera.set_filter_mode(SpuriousNoiseFilterMode.MEDIAN)
        else:
            self.camera.set_filter_mode(SpuriousNoiseFilterMode.OFF)

    def toggle_hotpixel(self):
        """Toggle hot pixel filter."""
        self.filter_hot_pixels = not self.filter_hot_pixels
        print(f"Hot pixel filter: {'ON' if self.filter_hot_pixels else 'OFF'}")

    def toggle_average(self):
        """Toggle between live view and averaged view (last 20 frames)."""
        self.show_average = not self.show_average
        if not self.show_average:
            self._frame_buffer.clear()
        print(f"Display mode: {'AVERAGE (20 frames)' if self.show_average else 'LIVE'}")

    def toggle_dark(self):
        """Toggle dark/bias subtraction."""
        if self.cal_dark is None:
            print("Dark subtract: No dark frame loaded")
            return
        self.apply_dark = not self.apply_dark
        print(f"Dark subtract: {'ON' if self.apply_dark else 'OFF'}")

    def toggle_flat(self):
        """Toggle flat field division."""
        if self.cal_flat_norm is None:
            print("Flat divide: No flat frame loaded")
            return
        self.apply_flat = not self.apply_flat
        print(f"Flat divide: {'ON' if self.apply_flat else 'OFF'}")

    def on_format_changed(self, index):
        """Handle format selection change."""
        self.save_as_ser = (index == 1)
        print(f"Save format: {'SER' if self.save_as_ser else 'FITS'}")

    def toggle_capture(self):
        """Toggle capture on/off."""
        if not self.capture_state:
            # Check disk space before starting
            can_capture, msg = self._check_disk_space()
            if not can_capture:
                print(f"[CAPTURE] Cannot start: {msg}")
                QtWidgets.QMessageBox.warning(self.win, "Disk Space", msg)
                return

            # Start or resume session
            if not self.session_active:
                self._start_session()

            self.capture_state = True
            self.capture_button.setText("Stop Capture")
            self.cnt = 0

            # Generate filename using session
            if self.save_as_ser:
                if self.session_active:
                    filepath = self.session.get_capture_path(
                        prefix=self.target_edit.text() or "capture",
                        extension=".ser"
                    )
                    fn = str(filepath)
                else:
                    fn = f"{self.filename_edit.text()}{time.time_ns()}.ser"

                self.ser_writer = SerWriter(fn)
                self.current_capture_path = Path(fn)
                # Determine depth: 2 bytes for uint16
                depth = 2 if self.array.dtype == np.uint16 else 1
                self.ser_writer.set_sizes(self.sx, self.sy, depth)
                print(f"Recording SER: {fn}")
        else:
            self.capture_state = False
            self.capture_button.setText("Start Capture")

            # Close SER file if active
            if self.ser_writer is not None:
                self.ser_writer.close()
                frames = self.ser_writer.count
                print(f"SER closed: {frames} frames")

                # Log capture to session
                if self.session_active and hasattr(self, 'current_capture_path'):
                    size = self.current_capture_path.stat().st_size if self.current_capture_path.exists() else 0
                    self.session.log_capture(self.current_capture_path, int(frames), int(size))
                    print(f"[SESSION] Logged: {self.current_capture_path.name} ({frames} frames, {size/(1024*1024):.1f} MB)")

                self.ser_writer = None

    def _roll_ser_file(self):
        """Close current SER file and start a new one."""
        if self.ser_writer is None:
            return

        # Close current file
        self.ser_writer.close()
        frames = self.ser_writer.count
        print(f"SER closed: {frames} frames")

        # Log to session
        if self.session_active and hasattr(self, 'current_capture_path'):
            size = self.current_capture_path.stat().st_size if self.current_capture_path.exists() else 0
            self.session.log_capture(self.current_capture_path, int(frames), int(size))
            print(f"[SESSION] Logged: {self.current_capture_path.name} ({frames} frames, {size/(1024*1024):.1f} MB)")

        # Reset frame counter
        self.cnt = 0

        # Open new file
        if self.session_active:
            filepath = self.session.get_capture_path(
                prefix=self.target_edit.text() or "capture",
                extension=".ser"
            )
            fn = str(filepath)
        else:
            fn = f"{self.filename_edit.text()}{time.time_ns()}.ser"

        self.ser_writer = SerWriter(fn)
        self.current_capture_path = Path(fn)
        depth = 2 if self.array.dtype == np.uint16 else 1
        self.ser_writer.set_sizes(self.sx, self.sy, depth)
        print(f"Recording SER: {fn}")

    def _start_session(self):
        """Start a new acquisition session."""
        target = self.target_edit.text().strip()
        filter_name = self.filter_edit.text().strip() or None

        # Update session manager base path from UI
        self.session.base_path = Path(self.data_dir_edit.text()).resolve()

        # Get camera settings for logging
        camera_settings = {
            "exposure": self.worker.exposure_time,
            "em_gain": self.worker.em_gain,
            "temperature": self.target_temp_spinbox.value(),
            "width": self.sx,
            "height": self.sy
        }

        session_dir = self.session.start_session(target, filter_name, camera_settings)
        self.session_active = True
        print(f"[SESSION] Started: {session_dir}")

    def _end_session(self, notes: str = ""):
        """End the current session."""
        if self.session_active:
            summary = self.session.end_session(notes)
            self.session_active = False
            print(f"[SESSION] Ended: {summary.get('total_frames', 0)} frames, {summary.get('total_size_mb', 0):.1f} MB")

    def _check_disk_space(self) -> tuple:
        """Check disk space and return (can_proceed, message)."""
        # Update session base path from UI
        self.session.base_path = Path(self.data_dir_edit.text()).resolve()

        # Check if we have a batch target for size estimation
        if self.batch_capture_target > 0:
            return self.session.can_capture(self.batch_capture_target, self.sx, self.sy)
        else:
            # Just check available space
            available_gb, status = self.session.check_disk_space()
            if status == "critical":
                return (False, f"Critical: only {available_gb:.1f} GB available")
            elif status == "warning":
                return (True, f"Warning: only {available_gb:.1f} GB available")
            return (True, f"OK: {available_gb:.1f} GB available")

    def _update_disk_space_display(self):
        """Update the disk space label."""
        available_gb, status = self.session.check_disk_space()
        if status == "critical":
            self.disk_space_label.setText(f"Disk: {available_gb:.0f}GB ⚠️")
            self.disk_space_label.setStyleSheet("color: red;")
        elif status == "warning":
            self.disk_space_label.setText(f"Disk: {available_gb:.0f}GB")
            self.disk_space_label.setStyleSheet("color: orange;")
        else:
            self.disk_space_label.setText(f"Disk: {available_gb:.0f}GB")
            self.disk_space_label.setStyleSheet("")

    def apply_calibration(self, frame):
        """Apply dark subtraction and flat field correction."""
        result = frame.astype(np.float32)

        # Dark/bias subtraction
        if self.apply_dark and self.cal_dark is not None:
            result = result - self.cal_dark

        # Flat field division
        if self.apply_flat and self.cal_flat_norm is not None:
            result = result / self.cal_flat_norm

        return result.clip(0, 65535).astype(np.uint16)

    def handle_new_frame(self, frame):
        """Handle new frame from worker thread."""
        self.exposure_timer.stop()
        if frame is None:
            return

        # Apply calibration (dark subtraction and/or flat field)
        if self.apply_dark or self.apply_flat:
            frame = self.apply_calibration(frame)

        self.array = frame
        self.idx += 1

        # Add to averaging buffer if enabled
        if self.show_average:
            self._frame_buffer.append(frame.astype(np.float32))
            if len(self._frame_buffer) > self.avg_frame_count:
                self._frame_buffer.pop(0)

        # Calculate FPS
        t1 = time.perf_counter()
        if self.idx > 0:
            self.fps = self.idx / (t1 - self.t0)

        # Save if capturing
        if self.capture_state:
            self.cnt += 1
            self.batch_frames_captured += 1
            self.save_frame(frame)

            # Check batch capture completion
            if self.batch_capture_target > 0 and self.batch_frames_captured >= self.batch_capture_target:
                print(f"[BATCH] Captured {self.batch_frames_captured} frames - target reached")
                self.toggle_capture()
                self._batch_complete()
                return

            # Roll to new file after frame_per_file frames
            if self.cnt >= self.frame_per_file:
                if self.save_as_ser:
                    self._roll_ser_file()
                else:
                    self.toggle_capture()  # FITS: stop and restart

        # Throttled display update - schedule if not already pending
        self._pending_frame = frame
        if not self._display_scheduled:
            self._display_scheduled = True
            QTimer.singleShot(67, self._do_display_update)  # ~15 FPS max

    def _do_display_update(self):
        """Deferred display update."""
        self._display_scheduled = False
        if self._pending_frame is not None:
            self.update_display()

    def save_frame(self, frame):
        """Save frame to file (FITS or SER depending on mode)."""
        # Add offset of 1000 to avoid negative values after calibration
        save_data = (frame.astype(np.int32) + 1000).clip(0, 65535).astype(np.uint16)

        # SER mode: append frame to open SER file
        if self.save_as_ser and self.ser_writer is not None:
            self.ser_writer.add_image(save_data)
            return

        # FITS mode: save individual files
        if not HAS_ASTROPY:
            fn = f"{self.filename_edit.text()}{time.time_ns()}.npy"
            np.save(fn, save_data)
            print(f"Saved: {fn}")
            return

        fn = f"{self.filename_edit.text()}{time.time_ns()}.fits"

        hdr = fits.Header()
        hdr['EXPTIME'] = self.exp_spinbox.value()
        hdr['EMGAIN'] = self.gain_spinbox.value()
        hdr['DATE-OBS'] = datetime.utcnow().isoformat()
        hdr['INSTRUME'] = self.camera.head_model
        hdr['CCD-TEMP'] = self.temperature
        hdr['XPIXSZ'] = self.camera.pixel_size[0]
        hdr['YPIXSZ'] = self.camera.pixel_size[1]
        hdr['PEDESTAL'] = (1000, 'Offset added to avoid negative values')

        fits.writeto(fn, save_data, hdr, overwrite=True)
        print(f"Saved: {fn}")

    def start_exposure_timer(self, duration):
        """Start exposure progress timer."""
        self.exposure_progress.setMaximum(int(duration * 10))
        self.exposure_progress.setValue(0)
        self.exposure_timer.start(100)

    def update_exposure_progress(self):
        """Update exposure progress bar."""
        current = self.exposure_progress.value()
        if current < self.exposure_progress.maximum():
            self.exposure_progress.setValue(current + 1)
        else:
            self.exposure_timer.stop()

    def update_temperature(self, sensor, target, ambient, volts, status):
        """Update temperature display."""
        self.temperature = sensor
        self.temp_status = status
        self.temp_label.setText(f"{sensor:.1f} °C ({status})")
        # Update disk space periodically (piggyback on temp updates)
        self._update_disk_space_display()

    def update_display(self):
        """Update the image display and stats."""
        if self.array is None:
            return

        self._display_count += 1

        # Use averaged frame or live frame
        if self.show_average and len(self._frame_buffer) > 0:
            # Compute average of buffered frames
            avg = np.mean(self._frame_buffer, axis=0)
            display_array = avg.astype(np.uint16)
        else:
            display_array = self.array

        # Optional hot pixel filtering (skip some frames to reduce load)
        if self.filter_hot_pixels and self._display_count % 2 == 0:
            display_array = filter_outliers_simple(
                display_array.astype(np.float64), sigma_threshold=4.0, in_place=False
            ).astype(np.uint16)

        # Update main image (rotate/flip for correct orientation)
        self.imv.setImage(
            np.rot90(display_array, k=-1),
            autoRange=False,
            autoLevels=False,
            autoHistogramRange=False
        )

        # Auto-level on first frame or if requested
        if self._first_frame or self.auto_level:
            vmin = np.percentile(display_array, 3)
            vmax = np.percentile(display_array, 97)
            self.imv.setLevels(vmin, vmax * 1.5)
            self._first_frame = False
            self.auto_level = False

        # Find brightest spot using tracked search
        star_x, star_y, _ = self.star_finder.find_high_value_element(display_array)
        self.pos.setX(star_x)
        self.pos.setY(star_y)
        self.pos = self._clip_pos(self.pos)

        # Extract subregion for zoom view
        x, y = int(self.pos.x()), int(self.pos.y())
        sub = display_array[
            y - self.EDGE:y + self.EDGE,
            x - self.EDGE:x + self.EDGE
        ].copy()

        # Calculate stats
        self.min_val = int(np.min(sub))
        self.max_val = int(np.max(sub))

        # Expensive stats only every 3rd display frame
        if self._display_count % 3 == 0:
            self.rms = np.std(display_array)
            fwhm_result = fit_gauss_circular(sub)
            self.fwhm = fwhm_result if isinstance(fwhm_result, float) else 0.0
            self.hfd = compute_hfd(sub)

        # Update stats labels
        self.stats_label1.setText(
            f"FWHM: {self.fwhm:.2f}  HFD: {self.hfd:.2f}  "
            f"Min: {self.min_val}  Max: {self.max_val}"
        )
        self.stats_label2.setText(
            f"RMS: {self.rms:.1f}  FPS: {self.fps:.2f}  Frame: {self.cnt}"
        )

        # Update scope RA/Dec (every ~30 frames to avoid slowdown)
        self._scope_update_counter += 1
        if self._scope_update_counter >= 30:
            self._scope_update_counter = 0
            if self.scope and self.scope.is_connected():
                try:
                    radec = self.scope.GetRaDec()
                    ra, dec = float(radec[0]), float(radec[1])
                except (IndexError, ValueError, SkyxConnectionError):
                    ra, dec = None, None
                if ra is not None and dec is not None:
                    # Convert RA from hours to H:M:S
                    ra_h = int(ra)
                    ra_m = int((ra - ra_h) * 60)
                    ra_s = ((ra - ra_h) * 60 - ra_m) * 60
                    # Convert Dec to D:M:S
                    dec_sign = '+' if dec >= 0 else '-'
                    dec_abs = abs(dec)
                    dec_d = int(dec_abs)
                    dec_m = int((dec_abs - dec_d) * 60)
                    dec_s = ((dec_abs - dec_d) * 60 - dec_m) * 60
                    self.scope_label.setText(
                        f"RA: {ra_h:02d}h{ra_m:02d}m{ra_s:04.1f}s  "
                        f"Dec: {dec_sign}{dec_d:02d}°{dec_m:02d}'{dec_s:04.1f}\""
                    )
                else:
                    self.scope_label.setText("Scope: no data")
            else:
                self.scope_label.setText("Scope: disconnected")

        # Update zoom view
        self._update_zoom_view(sub)

    def _clip_pos(self, pos):
        """Clip position to valid range."""
        x = max(self.EDGE, min(self.sx - self.EDGE, pos.x()))
        y = max(self.EDGE, min(self.sy - self.EDGE, pos.y()))
        return QPoint(int(x), int(y))

    def _update_zoom_view(self, sub):
        """Update the zoom view with the subregion and FWHM/HFD overlay."""
        # Normalize for display
        sub_disp = sub.astype(np.float32)
        sub_disp = sub_disp - self.min_val
        max_range = self.max_val - self.min_val
        if max_range > 0:
            sub_disp = sub_disp * (65535.0 / max_range)
        sub_disp = np.clip(sub_disp, 0, 65535).astype(np.uint16)

        # Resize for display
        if HAS_CV2:
            sub_resized = cv2.resize(sub_disp, (256, 256), interpolation=cv2.INTER_NEAREST)
        else:
            # Use scipy zoom
            zoom_factor = 256 / sub_disp.shape[0]
            sub_resized = zoom(sub_disp, zoom_factor, order=0).astype(np.uint16)

        # Convert to QPixmap
        h, w = sub_resized.shape
        qimg = QImage(sub_resized.data, w, h, w * 2, QImage.Format_Grayscale16)
        pixmap = QPixmap.fromImage(qimg)

        # Draw FWHM/HFD overlay
        painter = QtGui.QPainter(pixmap)
        painter.setPen(QtGui.QColor(0, 255, 0))  # Green text
        font = painter.font()
        font.setPointSize(10)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(5, 15, f"FWHM: {self.fwhm:.1f}")
        painter.drawText(5, 30, f"HFD: {self.hfd:.1f}")
        painter.end()

        self.zoom_view.setPixmap(pixmap)

    def mainloop(self):
        """Main event loop."""
        # Schedule autofocus if requested (after UI is up and frames are flowing)
        if self.args.autofocus:
            QTimer.singleShot(2000, self._run_autofocus_async)

        app = QtWidgets.QApplication.instance()
        app.exec_()

    def _run_autofocus_async(self):
        """Run autofocus in a separate thread to avoid blocking the UI."""
        import threading
        af_thread = threading.Thread(target=self.run_autofocus, daemon=True)
        af_thread.start()

    def _load_config(self):
        """Load saved configuration (target temp, cooler state)."""
        self._saved_target_temp = -60  # Default
        self._saved_cooler_on = False
        try:
            if CONFIG_FILE.exists():
                with open(CONFIG_FILE, 'r') as f:
                    config = json.load(f)
                self._saved_target_temp = config.get('target_temp', -60)
                self._saved_cooler_on = config.get('cooler_on', False)
                print(f"Loaded config: target={self._saved_target_temp}°C, cooler={'ON' if self._saved_cooler_on else 'OFF'}")
        except Exception as e:
            print(f"Could not load config: {e}")

    def _save_config(self):
        """Save current configuration."""
        try:
            config = {
                'target_temp': self.target_temp_spinbox.value(),
                'cooler_on': self.cooler_is_on and not self.cooler_shutdown_complete
            }
            with open(CONFIG_FILE, 'w') as f:
                json.dump(config, f)
            print(f"Saved config: target={config['target_temp']}°C, cooler={'ON' if config['cooler_on'] else 'OFF'}")
        except Exception as e:
            print(f"Could not save config: {e}")

    def _restore_cooler_state(self):
        """Restore cooler state from saved config (called after UI is ready)."""
        # Set target temperature from saved config
        self.target_temp_spinbox.setValue(self._saved_target_temp)
        self.camera.set_temperature(self._saved_target_temp)

        # Restore cooler state if it was on
        if self._saved_cooler_on:
            self.camera.cooler_on()
            self.cooler_is_on = True
            self.cooler_button.setText("Cooler ON")
            print(f"Restored cooler to {self._saved_target_temp}°C")

    def _apply_startup_options(self):
        """Apply command line options after UI is ready."""
        args = self.args

        # Display overlays
        if getattr(args, 'crosshair', False):
            self.toggle_crosshair()
        if getattr(args, 'bullseye', False):
            self.toggle_bullseye()

        # Frame averaging
        avg_count = getattr(args, 'average', 0)
        if avg_count > 0:
            self.avg_frame_count = avg_count
            self.show_average = True
            print(f"Frame averaging: ON ({avg_count} frames)")

        # Hot pixel filter
        if getattr(args, 'hot_pixel_filter', False):
            self.filter_hot_pixels = True
            print("Hot pixel filter: ON")

        # Output format
        if getattr(args, 'ser', False):
            self.save_as_ser = True
            self.format_combo.setCurrentIndex(1)
            print("Output format: SER")

        # Print startup status
        status = []
        if self.apply_dark:
            status.append("dark")
        if self.apply_flat:
            status.append("flat")
        if status:
            print(f"Processing pipeline: {' → '.join(status)}")

        # Batch capture mode
        if self.batch_capture_target > 0:
            print(f"[BATCH] Target: {self.batch_capture_target} frames")
            if self.batch_wait_temp:
                print("[BATCH] Waiting for temperature to stabilize...")
                QTimer.singleShot(1000, self._check_temp_for_batch)
            elif self.batch_auto_start:
                print("[BATCH] Auto-starting capture...")
                QTimer.singleShot(500, self._start_batch_capture)
        elif self.batch_auto_start:
            print("[BATCH] Auto-starting capture (no frame limit)...")
            QTimer.singleShot(500, self.toggle_capture)

    def _check_temp_for_batch(self):
        """Check if temperature is stable before starting batch capture."""
        temp, status = self.camera.get_temperature()
        target = self.target_temp_spinbox.value()

        # Consider stable if within 2 degrees or status indicates stable
        if abs(temp - target) <= 2 or status == 1:  # DRV_TEMPERATURE_STABILIZED
            print(f"[BATCH] Temperature stable at {temp:.1f}°C")
            self.batch_temp_ready = True
            if self.batch_auto_start:
                self._start_batch_capture()
        else:
            print(f"[BATCH] Waiting... {temp:.1f}°C (target: {target}°C)")
            QTimer.singleShot(5000, self._check_temp_for_batch)

    def _start_batch_capture(self):
        """Start capture for batch mode."""
        if not self.capture_state:
            print(f"[BATCH] Starting capture of {self.batch_capture_target} frames")
            self.toggle_capture()

    def _batch_complete(self):
        """Called when batch capture is complete."""
        print(f"[BATCH] Capture complete: {self.batch_frames_captured} frames saved")

        # End session
        if self.session_active:
            self._end_session()

        print("[BATCH] Closing application...")
        # Schedule application quit
        QTimer.singleShot(500, self.win.close)

    # ========================================================================
    # Autofocus
    # ========================================================================

    def load_autofocus_config(self):
        """Load autofocus parameters from detector.ini."""
        config = configparser.ConfigParser()
        config_path = Path(__file__).parent / "detector.ini"
        config.read(config_path)

        af = {}
        if config.has_section('Autofocus'):
            af['min_brightness'] = config.getint('Autofocus', 'MinBrightness', fallback=10000)
            af['samples_per_position'] = config.getint('Autofocus', 'SamplesPerPosition', fallback=25)
            af['step_size'] = config.getint('Autofocus', 'StepSize', fallback=50)
            af['positions_per_direction'] = config.getint('Autofocus', 'PositionsPerDirection', fallback=4)
            af['max_travel'] = config.getint('Autofocus', 'MaxTravel', fallback=400)
            af['roi_size'] = config.getint('Autofocus', 'ROISize', fallback=20)
        else:
            # Defaults
            af = {
                'min_brightness': 10000,
                'samples_per_position': 25,
                'step_size': 50,
                'positions_per_direction': 4,
                'max_travel': 400,
                'roi_size': 20
            }
        return af

    def find_brightest_star(self, frame):
        """Find brightest star in frame using Gaussian filtering."""
        filtered = gaussian_filter(frame.astype(float), sigma=2)
        y, x = np.unravel_index(np.argmax(filtered), filtered.shape)
        brightness = filtered[y, x]
        return y, x, brightness

    def extract_star_roi(self, frame, y, x, roi_size):
        """Extract ROI around star position."""
        y1 = max(0, y - roi_size)
        y2 = min(frame.shape[0], y + roi_size)
        x1 = max(0, x - roi_size)
        x2 = min(frame.shape[1], x + roi_size)
        roi = frame[y1:y2, x1:x2].copy()
        # Subtract background
        roi = roi.astype(float) - np.min(roi)
        return roi

    def measure_hfd_at_position(self, af_config, focuser_obj):
        """
        Measure HFD at current focus position by averaging multiple samples.
        Returns (mean_hfd, std_hfd) or (None, None) if star lost.
        """
        hfd_samples = []
        roi_size = af_config['roi_size']
        min_brightness = af_config['min_brightness']
        samples_needed = af_config['samples_per_position']

        # Wait for focuser to settle
        time.sleep(0.3)

        attempts = 0
        max_attempts = samples_needed * 3  # Allow some failed frames

        while len(hfd_samples) < samples_needed and attempts < max_attempts:
            attempts += 1

            # Get a frame - use the most recent frame from the display
            frame = self.array.copy()

            # Find star
            y, x, brightness = self.find_brightest_star(frame)

            if brightness < min_brightness:
                continue  # Skip this frame

            # Extract ROI and compute HFD
            roi = self.extract_star_roi(frame, y, x, roi_size)
            if roi.size == 0:
                continue

            hfd = compute_hfd(roi)
            if hfd > 0 and hfd < 50:  # Sanity check
                hfd_samples.append(hfd)

            # Small delay between samples
            time.sleep(0.05)

        if len(hfd_samples) < samples_needed // 2:
            return None, None  # Not enough valid samples

        # Filter outliers and compute mean
        hfd_array = np.array(hfd_samples)
        mean_hfd = np.mean(hfd_array)
        std_hfd = np.std(hfd_array)

        return mean_hfd, std_hfd

    def run_autofocus(self):
        """
        Run the autofocus routine.
        Returns True if successful, False otherwise.
        """
        print("\n" + "=" * 60)
        print("AUTOFOCUS STARTING")
        print("=" * 60)

        if not HAS_FOCUSER:
            print("ERROR: Focuser not available")
            return False

        # Load config
        af_config = self.load_autofocus_config()
        print(f"Config: step={af_config['step_size']}, "
              f"positions={af_config['positions_per_direction']}/dir, "
              f"samples={af_config['samples_per_position']}")

        # Initialize focuser
        try:
            foc = FLIFocuser()
            start_position = foc.get_abs_pos()
            print(f"Focuser connected, start position: {start_position}")
        except Exception as e:
            print(f"ERROR: Could not connect to focuser: {e}")
            return False

        # Wait for some frames to arrive
        print("Waiting for frames...")
        time.sleep(1.0)

        # Check for bright star
        frame = self.array.copy()
        y, x, brightness = self.find_brightest_star(frame)
        print(f"Brightest star at ({x}, {y}) with brightness {brightness:.0f} ADU")

        if brightness < af_config['min_brightness']:
            print(f"ERROR: Star too faint ({brightness:.0f} < {af_config['min_brightness']} ADU)")
            print("AUTOFOCUS ABORTED")
            return False

        # Measure baseline HFD
        print("\nMeasuring baseline HFD...")
        baseline_hfd, baseline_std = self.measure_hfd_at_position(af_config, foc)
        if baseline_hfd is None:
            print("ERROR: Could not measure baseline HFD")
            return False
        print(f"Baseline HFD: {baseline_hfd:.2f} ± {baseline_std:.2f}")

        # Collect measurements at multiple positions
        measurements = [(start_position, baseline_hfd)]
        step_size = af_config['step_size']
        n_positions = af_config['positions_per_direction']

        # Scan positive direction
        print(f"\nScanning +direction ({n_positions} positions)...")
        current_pos = start_position
        for i in range(n_positions):
            foc.move_focus(step_size)
            current_pos += step_size
            print(f"  Position {current_pos}: ", end="", flush=True)

            hfd, std = self.measure_hfd_at_position(af_config, foc)
            if hfd is not None:
                measurements.append((current_pos, hfd))
                print(f"HFD = {hfd:.2f} ± {std:.2f}")
            else:
                print("FAILED (star lost)")
                break

        # Return to start
        print("\nReturning to start position...")
        foc.move_to(start_position)
        time.sleep(0.5)

        # Scan negative direction
        print(f"\nScanning -direction ({n_positions} positions)...")
        current_pos = start_position
        for i in range(n_positions):
            foc.move_focus(-step_size)
            current_pos -= step_size
            print(f"  Position {current_pos}: ", end="", flush=True)

            hfd, std = self.measure_hfd_at_position(af_config, foc)
            if hfd is not None:
                measurements.append((current_pos, hfd))
                print(f"HFD = {hfd:.2f} ± {std:.2f}")
            else:
                print("FAILED (star lost)")
                break

        # Need at least 3 points for parabola fit
        if len(measurements) < 3:
            print("\nERROR: Not enough measurements for parabola fit")
            print("Returning to start position...")
            foc.move_to(start_position)
            return False

        # Fit parabola: HFD = a*pos^2 + b*pos + c
        print(f"\nFitting parabola to {len(measurements)} points...")
        positions = np.array([m[0] for m in measurements])
        hfds = np.array([m[1] for m in measurements])

        # Polynomial fit (degree 2)
        coeffs = np.polyfit(positions, hfds, 2)
        a, b, c = coeffs

        # Check parabola opens upward (a > 0)
        if a <= 0:
            print("WARNING: Parabola opens downward - focus curve may be invalid")
            print("Returning to start position...")
            foc.move_to(start_position)
            return False

        # Find minimum: derivative = 2*a*pos + b = 0 => pos = -b/(2*a)
        best_position = int(-b / (2 * a))
        predicted_hfd = a * best_position**2 + b * best_position + c

        print(f"Parabola: HFD = {a:.6f}*pos² + {b:.4f}*pos + {c:.2f}")
        print(f"Predicted best position: {best_position}")
        print(f"Predicted HFD: {predicted_hfd:.2f}")

        # Check if best position is within limits
        if abs(best_position - start_position) > af_config['max_travel']:
            print(f"WARNING: Best position exceeds max travel limit")
            best_position = start_position + np.sign(best_position - start_position) * af_config['max_travel']
            print(f"Clamping to: {best_position}")

        # Move to best position
        print(f"\nMoving to best position: {best_position}")
        foc.move_to(best_position)

        # Final verification
        print("Verifying final HFD...")
        final_hfd, final_std = self.measure_hfd_at_position(af_config, foc)
        if final_hfd is None:
            print("ERROR: Could not measure final HFD")
            print("Returning to start position...")
            foc.move_to(start_position)
            return False

        print(f"Final HFD: {final_hfd:.2f} ± {final_std:.2f}")

        # Compare to baseline
        if final_hfd >= baseline_hfd:
            print(f"\nNo improvement (final {final_hfd:.2f} >= baseline {baseline_hfd:.2f})")
            print("Returning to start position...")
            foc.move_to(start_position)
            print("\n" + "=" * 60)
            print("AUTOFOCUS COMPLETE - NO IMPROVEMENT")
            print("=" * 60 + "\n")
            return False

        improvement = ((baseline_hfd - final_hfd) / baseline_hfd) * 100
        print(f"\nImprovement: {improvement:.1f}% (HFD: {baseline_hfd:.2f} -> {final_hfd:.2f})")
        print(f"Best focus position: {best_position}")
        print("\n" + "=" * 60)
        print("AUTOFOCUS COMPLETE - SUCCESS")
        print("=" * 60 + "\n")

        return True

    def _cleanup(self):
        """Cleanup on exit."""
        # Save config before exiting (preserves cooler state unless shutdown was done)
        self._save_config()

        # Stop shutdown timer if running
        if self.shutdown_timer is not None:
            self.shutdown_timer.stop()
            self.shutdown_timer = None

        # Close SER file if recording
        if self.ser_writer is not None:
            self.ser_writer.close()
            print(f"SER closed on exit: {self.ser_writer.count} frames")
            self.ser_writer = None

        # End session if active
        if self.session_active:
            self._end_session()

        # Note: We do NOT turn off the cooler here - it keeps running
        # Only gradual shutdown turns it off

        if self.worker.running:
            self.worker.stop_capture()
            self.thread.quit()
            self.thread.wait()


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Andor iXon Ultra 888 Camera UI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Keyboard shortcuts:
  D        Toggle dark subtraction
  F        Toggle flat field division
  Space    Start/stop capture
  Ctrl+A   Auto-level display
  Ctrl+B   Toggle bullseye overlay
  Ctrl+C   Toggle crosshair overlay
  Ctrl+G   Toggle frame averaging
  Ctrl+H   Toggle hot pixel filter
  Ctrl+N   Toggle CIC filter
  Ctrl+T   Toggle cooler
"""
    )

    # Basic camera settings
    parser.add_argument("-f", "--filename", type=str, default="ixon_",
                        help="Base filename for captures (default: ixon_)")
    parser.add_argument("-exp", type=float, default=0.001,
                        help="Exposure time in seconds (default: 0.1)")
    parser.add_argument("-gain", type=int, default=1000,
                        help="EM gain DAC value (0-4095) (default: 1000)")
    parser.add_argument("-count", type=int, default=100,
                        help="Number of frames to capture per file (default: 100)")
    parser.add_argument("-temp", type=int, default=-60,
                        help="Target temperature in Celsius (default: -60)")
    parser.add_argument("-sim", "--simulate", action="store_true",
                        help="Force simulated camera")

    # Calibration options
    parser.add_argument("--dark", action="store_true",
                        help="Enable dark/bias subtraction on startup")
    parser.add_argument("--no-dark", action="store_true",
                        help="Disable dark/bias subtraction on startup")
    parser.add_argument("--flat", action="store_true",
                        help="Enable flat field division on startup")
    parser.add_argument("--no-flat", action="store_true",
                        help="Disable flat field division on startup")
    parser.add_argument("--dark-file", type=str, default=None,
                        help="Path to dark/bias FITS file (overrides config)")
    parser.add_argument("--flat-file", type=str, default=None,
                        help="Path to flat field FITS file (overrides config)")

    # Display options
    parser.add_argument("--crosshair", action="store_true",
                        help="Show crosshair overlay on startup")
    parser.add_argument("--bullseye", action="store_true",
                        help="Show bullseye overlay on startup")
    parser.add_argument("--average", type=int, default=0, metavar="N",
                        help="Enable frame averaging with N frames (default: off)")
    parser.add_argument("--hot-pixel-filter", action="store_true",
                        help="Enable hot pixel filter on startup")

    # Output format
    parser.add_argument("--ser", action="store_true",
                        help="Use SER format instead of FITS for capture")

    # Session/data management
    parser.add_argument("--target", type=str, default="noname_target",
                        help="Target name (e.g., M42, NGC7000)")
    parser.add_argument("--filter", type=str, default="",
                        help="Filter name (e.g., Ha, OIII, L)")
    parser.add_argument("--data-dir", type=str, default=".",
                        help="Base directory for data (default: current dir)")

    # Batch/scripted capture mode
    parser.add_argument("--capture", type=int, default=500000, metavar="N",
                        help="Capture N total frames then exit (default: 500000)")
    parser.add_argument("--auto-start", action="store_true",
                        help="Start capturing immediately on startup")
    parser.add_argument("--wait-temp", action="store_true",
                        help="Wait for temperature to stabilize before capturing")

    # Autofocus
    parser.add_argument("--autofocus", action="store_true",
                        help="Run autofocus routine before starting")

    args = parser.parse_args()

    # Use the QApplication created at module level (required for pyqtgraph)
    app = _app

    # Get camera (real or simulated)
    camera = get_camera(simulate=args.simulate)

    # Configure camera
    camera.set_exposure_time(args.exp)
    camera.set_output_amplifier(0)  # EM amplifier
    camera.set_hs_speed(HS, 0)  # index=1 (20 MHz), EM amp
    camera.set_em_gain_mode(1)  # 12-bit DAC (0-4095)

    camera.set_preamp_gain(1)  # 2x preamp
    camera.set_vs_speed(0)  # VS speed index 1
    #camera.set_emccd_gain(args.gain)
    camera.set_temperature(args.temp)
    #camera.optimize_readout_speed(use_em_amplifier=True)

    # Create and run UI
    ui = AndorUI(camera, args)

    try:
        ui.mainloop()
    finally:
        # Set cooler mode based on whether we did a proper shutdown
        if ui.cooler_shutdown_complete:
            # Cooler was already warmed up and turned off
            camera.set_cooler_mode(False)
            print("Camera closing (cooler was shut down)")
        else:
            # Keep cooler running after ShutDown
            camera.set_cooler_mode(True)
            print("Camera closing (cooler will stay on)")
        camera.close_shutter()
        camera.close()
