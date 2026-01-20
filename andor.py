"""
Python bindings for Andor SDK 2.91 (Linux)

This module provides a Pythonic interface to the Andor camera SDK for
controlling CCD, EMCCD, and ICCD cameras.

Example usage:
    from andor import AndorCamera

    with AndorCamera() as cam:
        cam.set_temperature(-60)
        cam.cooler_on()
        cam.set_exposure_time(1.0)
        cam.set_acquisition_mode(AcquisitionMode.SINGLE_SCAN)
        cam.set_read_mode(ReadMode.IMAGE)

        cam.start_acquisition()
        cam.wait_for_acquisition()
        image = cam.get_acquired_data()
"""

import ctypes
from ctypes import (
    c_int, c_uint, c_short, c_ushort, c_long, c_ulong, c_longlong, c_ulonglong,
    c_float, c_double, c_char, c_char_p, c_void_p, c_ubyte,
    POINTER, byref, create_string_buffer, Structure
)
import numpy as np
import time
from enum import IntEnum, IntFlag
from pathlib import Path
from typing import Optional, Tuple, List, Union
import os

# Try to import Cython-accelerated simulation module
try:
    import sim_frame as _sim_frame
    _USE_CYTHON_SIM = True
    print("Using Cython-accelerated frame simulation")
except ImportError:
    _USE_CYTHON_SIM = False


# ============================================================================
# Type definitions (matching atmcdLXd.h)
# ============================================================================

at_u16 = c_ushort
at_32 = c_int      # 64-bit Linux
at_u32 = c_uint    # 64-bit Linux
at_64 = c_longlong
at_u64 = c_ulonglong


# ============================================================================
# Structures
# ============================================================================

class AndorCapabilities(Structure):
    """Camera capabilities structure."""
    _fields_ = [
        ("ulSize", at_u32),
        ("ulAcqModes", at_u32),
        ("ulReadModes", at_u32),
        ("ulTriggerModes", at_u32),
        ("ulCameraType", at_u32),
        ("ulPixelMode", at_u32),
        ("ulSetFunctions", at_u32),
        ("ulGetFunctions", at_u32),
        ("ulFeatures", at_u32),
        ("ulPCICard", at_u32),
        ("ulEMGainCapability", at_u32),
        ("ulFTReadModes", at_u32),
    ]


class ColorDemosaicInfo(Structure):
    """Color demosaicing parameters."""
    _fields_ = [
        ("iX", c_int),
        ("iY", c_int),
        ("iAlgorithm", c_int),
        ("iXPhase", c_int),
        ("iYPhase", c_int),
        ("iBackground", c_int),
    ]


class WhiteBalanceInfo(Structure):
    """White balance parameters."""
    _fields_ = [
        ("iSize", c_int),
        ("iX", c_int),
        ("iY", c_int),
        ("iAlgorithm", c_int),
        ("iROI_left", c_int),
        ("iROI_right", c_int),
        ("iROI_top", c_int),
        ("iROI_bottom", c_int),
        ("iOperation", c_int),
    ]


# ============================================================================
# Enumerations
# ============================================================================

class ErrorCode(IntEnum):
    """Andor SDK error codes."""
    DRV_ERROR_CODES = 20001
    DRV_SUCCESS = 20002
    DRV_VXDNOTINSTALLED = 20003
    DRV_ERROR_SCAN = 20004
    DRV_ERROR_CHECK_SUM = 20005
    DRV_ERROR_FILELOAD = 20006
    DRV_UNKNOWN_FUNCTION = 20007
    DRV_ERROR_VXD_INIT = 20008
    DRV_ERROR_ADDRESS = 20009
    DRV_ERROR_PAGELOCK = 20010
    DRV_ERROR_PAGEUNLOCK = 20011
    DRV_ERROR_BOARDTEST = 20012
    DRV_ERROR_ACK = 20013
    DRV_ERROR_UP_FIFO = 20014
    DRV_ERROR_PATTERN = 20015
    DRV_ACQUISITION_ERRORS = 20017
    DRV_ACQ_BUFFER = 20018
    DRV_ACQ_DOWNFIFO_FULL = 20019
    DRV_PROC_UNKNOWN_INSTRUCTION = 20020
    DRV_ILLEGAL_OP_CODE = 20021
    DRV_KINETIC_TIME_NOT_MET = 20022
    DRV_ACCUM_TIME_NOT_MET = 20023
    DRV_NO_NEW_DATA = 20024
    KERN_MEM_ERROR = 20025
    DRV_SPOOLERROR = 20026
    DRV_SPOOLSETUPERROR = 20027
    DRV_FILESIZELIMITERROR = 20028
    DRV_ERROR_FILESAVE = 20029
    DRV_TEMPERATURE_CODES = 20033
    DRV_TEMPERATURE_OFF = 20034
    DRV_TEMPERATURE_NOT_STABILIZED = 20035
    DRV_TEMPERATURE_STABILIZED = 20036
    DRV_TEMPERATURE_NOT_REACHED = 20037
    DRV_TEMPERATURE_OUT_RANGE = 20038
    DRV_TEMPERATURE_NOT_SUPPORTED = 20039
    DRV_TEMPERATURE_DRIFT = 20040
    DRV_GENERAL_ERRORS = 20049
    DRV_INVALID_AUX = 20050
    DRV_COF_NOTLOADED = 20051
    DRV_FPGAPROG = 20052
    DRV_FLEXERROR = 20053
    DRV_GPIBERROR = 20054
    DRV_EEPROMVERSIONERROR = 20055
    DRV_DATATYPE = 20064
    DRV_DRIVER_ERRORS = 20065
    DRV_P1INVALID = 20066
    DRV_P2INVALID = 20067
    DRV_P3INVALID = 20068
    DRV_P4INVALID = 20069
    DRV_INIERROR = 20070
    DRV_COFERROR = 20071
    DRV_ACQUIRING = 20072
    DRV_IDLE = 20073
    DRV_TEMPCYCLE = 20074
    DRV_NOT_INITIALIZED = 20075
    DRV_P5INVALID = 20076
    DRV_P6INVALID = 20077
    DRV_INVALID_MODE = 20078
    DRV_INVALID_FILTER = 20079
    DRV_I2CERRORS = 20080
    DRV_I2CDEVNOTFOUND = 20081
    DRV_I2CTIMEOUT = 20082
    DRV_P7INVALID = 20083
    DRV_P8INVALID = 20084
    DRV_P9INVALID = 20085
    DRV_P10INVALID = 20086
    DRV_P11INVALID = 20087
    DRV_USBERROR = 20089
    DRV_IOCERROR = 20090
    DRV_VRMVERSIONERROR = 20091
    DRV_GATESTEPERROR = 20092
    DRV_USB_INTERRUPT_ENDPOINT_ERROR = 20093
    DRV_RANDOM_TRACK_ERROR = 20094
    DRV_INVALID_TRIGGER_MODE = 20095
    DRV_LOAD_FIRMWARE_ERROR = 20096
    DRV_DIVIDE_BY_ZERO_ERROR = 20097
    DRV_INVALID_RINGEXPOSURES = 20098
    DRV_BINNING_ERROR = 20099
    DRV_INVALID_AMPLIFIER = 20100
    DRV_INVALID_COUNTCONVERT_MODE = 20101
    DRV_ERROR_NOCAMERA = 20990
    DRV_NOT_SUPPORTED = 20991
    DRV_NOT_AVAILABLE = 20992
    DRV_ERROR_MAP = 20115
    DRV_ERROR_UNMAP = 20116
    DRV_ERROR_MDL = 20117
    DRV_ERROR_UNMDL = 20118
    DRV_ERROR_BUFFSIZE = 20119
    DRV_ERROR_NOHANDLE = 20121
    DRV_GATING_NOT_AVAILABLE = 20130
    DRV_FPGA_VOLTAGE_ERROR = 20131
    DRV_PROCESSING_FAILED = 20211


class AcquisitionMode(IntEnum):
    """Camera acquisition modes."""
    SINGLE_SCAN = 1
    ACCUMULATE = 2
    KINETICS = 3
    FAST_KINETICS = 4
    RUN_TILL_ABORT = 5


class ReadMode(IntEnum):
    """Camera read modes."""
    FULL_VERTICAL_BINNING = 0
    MULTI_TRACK = 1
    RANDOM_TRACK = 2
    SINGLE_TRACK = 3
    IMAGE = 4


class TriggerMode(IntEnum):
    """Camera trigger modes."""
    INTERNAL = 0
    EXTERNAL = 1
    EXTERNAL_START = 6
    EXTERNAL_EXPOSURE = 7
    EXTERNAL_FVB_EM = 9
    SOFTWARE = 10
    EXTERNAL_CHARGE_SHIFTING = 12


class ShutterMode(IntEnum):
    """Shutter operating modes."""
    AUTO = 0
    OPEN = 1
    CLOSE = 2


class ShutterType(IntEnum):
    """Shutter types."""
    TTL_LOW = 0
    TTL_HIGH = 1


class CameraType(IntEnum):
    """Camera type identifiers."""
    PDA = 0
    IXON = 1
    ICCD = 2
    EMCCD = 3
    CCD = 4
    ISTAR = 5
    VIDEO = 6
    IDUS = 7
    NEWTON = 8
    SURCAM = 9
    USBICCD = 10
    LUCA = 11
    RESERVED = 12
    IKON = 13
    INGAAS = 14
    IVAC = 15
    UNPROGRAMMED = 16
    CLARA = 17
    USBISTAR = 18
    SIMCAM = 19
    NEO = 20
    XTREME = 21


class CameraStatus(IntEnum):
    """Camera status codes."""
    IDLE = 20073
    TEMPCYCLE = 20074
    ACQUIRING = 20072
    ACCUM_TIME_NOT_MET = 20023
    KINETIC_TIME_NOT_MET = 20022
    ACQ_BUFFER = 20018
    SPOOLERROR = 20026


class AcqModeCapability(IntFlag):
    """Acquisition mode capability flags."""
    SINGLE = 1
    VIDEO = 2
    ACCUMULATE = 4
    KINETIC = 8
    FRAMETRANSFER = 16
    FASTKINETICS = 32
    OVERLAP = 64


class ReadModeCapability(IntFlag):
    """Read mode capability flags."""
    FULLIMAGE = 1
    SUBIMAGE = 2
    SINGLETRACK = 4
    FVB = 8
    MULTITRACK = 16
    RANDOMTRACK = 32
    MULTITRACKSCAN = 64


class TriggerModeCapability(IntFlag):
    """Trigger mode capability flags."""
    INTERNAL = 1
    EXTERNAL = 2
    EXTERNAL_FVB_EM = 4
    CONTINUOUS = 8
    EXTERNALSTART = 16
    EXTERNALEXPOSURE = 32
    INVERTED = 0x40
    EXTERNAL_CHARGESHIFTING = 0x80


class FeatureCapability(IntFlag):
    """Feature capability flags."""
    POLLING = 1
    EVENTS = 2
    SPOOLING = 4
    SHUTTER = 8
    SHUTTEREX = 16
    EXTERNAL_I2C = 32
    SATURATIONEVENT = 64
    FANCONTROL = 128
    MIDFANCONTROL = 256
    TEMPERATUREDURINGACQUISITION = 512
    KEEPCLEANCONTROL = 1024
    DDGLITE = 0x0800
    FTEXTERNALEXPOSURE = 0x1000
    KINETICEXTERNALEXPOSURE = 0x2000
    DACCONTROL = 0x4000
    METADATA = 0x8000
    IOCONTROL = 0x10000
    PHOTONCOUNTING = 0x20000
    COUNTCONVERT = 0x40000
    DUALMODE = 0x80000


class FanMode(IntEnum):
    """Fan operating modes."""
    FULL = 0
    LOW = 1
    OFF = 2


class SpuriousNoiseFilterMode(IntEnum):
    """Spurious noise (CIC) filter modes."""
    OFF = 0
    MEDIAN = 1
    LEVEL_ABOVE = 2


class DataAveragingMode(IntEnum):
    """Data averaging modes for noise reduction."""
    OFF = 0
    RECURSIVE = 1
    FRAME = 2


# ============================================================================
# Exceptions
# ============================================================================

class AndorError(Exception):
    """Base exception for Andor SDK errors."""

    def __init__(self, code: int, message: str = ""):
        self.code = code
        try:
            self.error_name = ErrorCode(code).name
        except ValueError:
            self.error_name = f"UNKNOWN_ERROR_{code}"

        if message:
            super().__init__(f"{self.error_name} ({code}): {message}")
        else:
            super().__init__(f"{self.error_name} ({code})")


class AndorNotInitializedError(AndorError):
    """Camera not initialized."""
    pass


class AndorAcquiringError(AndorError):
    """Camera is currently acquiring."""
    pass


class AndorTemperatureError(AndorError):
    """Temperature-related error."""
    pass


# ============================================================================
# Library Loading
# ============================================================================

def _find_library() -> str:
    """Find the Andor library path."""
    import sys

    # Try to find library in common locations
    search_paths = [
        # Conda environment lib directory (highest priority)
        Path(sys.prefix) / "lib" / "libandor.so",
        # Local package lib directory
        Path(__file__).parent / "lib" / "libandor.so",
        Path(__file__).parent / "lib" / "libandor.so.2",
        # System locations
        Path("/usr/local/lib/libandor.so"),
        Path("/usr/lib/libandor.so"),
        Path("/opt/andor/lib/libandor.so"),
    ]

    # Also check LD_LIBRARY_PATH
    ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    for p in ld_path.split(":"):
        if p:
            search_paths.append(Path(p) / "libandor.so")

    for path in search_paths:
        if path.exists():
            return str(path)

    # Fall back to letting ctypes find it
    return "libandor.so.2"


def _load_library() -> ctypes.CDLL:
    """Load the Andor shared library."""
    lib_path = _find_library()
    try:
        # RTLD_GLOBAL is required for Andor's internal linking
        lib = ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
    except OSError as e:
        raise ImportError(
            f"Could not load Andor library from {lib_path}. "
            "Make sure the library is installed and LD_LIBRARY_PATH is set correctly. "
            "If you get permission errors, try running 'sudo udevadm trigger'."
        ) from e

    # Set up function signatures for type safety
    _setup_function_signatures(lib)
    return lib


def _setup_function_signatures(lib: ctypes.CDLL) -> None:
    """Define function signatures for type checking."""

    # Initialization
    lib.Initialize.argtypes = [c_char_p]
    lib.Initialize.restype = c_uint

    lib.InitializeDevice.argtypes = [c_char_p]
    lib.InitializeDevice.restype = c_uint

    lib.ShutDown.argtypes = []
    lib.ShutDown.restype = c_uint

    # Camera info
    lib.GetAvailableCameras.argtypes = [POINTER(at_32)]
    lib.GetAvailableCameras.restype = c_uint

    lib.GetCameraHandle.argtypes = [at_32, POINTER(at_32)]
    lib.GetCameraHandle.restype = c_uint

    lib.SetCurrentCamera.argtypes = [at_32]
    lib.SetCurrentCamera.restype = c_uint

    lib.GetCurrentCamera.argtypes = [POINTER(at_32)]
    lib.GetCurrentCamera.restype = c_uint

    lib.GetCameraSerialNumber.argtypes = [POINTER(c_int)]
    lib.GetCameraSerialNumber.restype = c_uint

    lib.GetHeadModel.argtypes = [c_char_p]
    lib.GetHeadModel.restype = c_uint

    lib.GetDetector.argtypes = [POINTER(c_int), POINTER(c_int)]
    lib.GetDetector.restype = c_uint

    lib.GetCapabilities.argtypes = [POINTER(AndorCapabilities)]
    lib.GetCapabilities.restype = c_uint

    lib.GetPixelSize.argtypes = [POINTER(c_float), POINTER(c_float)]
    lib.GetPixelSize.restype = c_uint

    # Temperature
    lib.GetTemperature.argtypes = [POINTER(c_int)]
    lib.GetTemperature.restype = c_uint

    lib.GetTemperatureF.argtypes = [POINTER(c_float)]
    lib.GetTemperatureF.restype = c_uint

    lib.GetTemperatureRange.argtypes = [POINTER(c_int), POINTER(c_int)]
    lib.GetTemperatureRange.restype = c_uint

    lib.SetTemperature.argtypes = [c_int]
    lib.SetTemperature.restype = c_uint

    lib.CoolerON.argtypes = []
    lib.CoolerON.restype = c_uint

    lib.CoolerOFF.argtypes = []
    lib.CoolerOFF.restype = c_uint

    lib.IsCoolerOn.argtypes = [POINTER(c_int)]
    lib.IsCoolerOn.restype = c_uint

    lib.SetCoolerMode.argtypes = [c_int]
    lib.SetCoolerMode.restype = c_uint

    lib.GetTECStatus.argtypes = [POINTER(c_int)]
    lib.GetTECStatus.restype = c_uint

    lib.SetFanMode.argtypes = [c_int]
    lib.SetFanMode.restype = c_uint

    # Acquisition setup
    lib.SetAcquisitionMode.argtypes = [c_int]
    lib.SetAcquisitionMode.restype = c_uint

    lib.SetReadMode.argtypes = [c_int]
    lib.SetReadMode.restype = c_uint

    lib.SetTriggerMode.argtypes = [c_int]
    lib.SetTriggerMode.restype = c_uint

    lib.SetExposureTime.argtypes = [c_float]
    lib.SetExposureTime.restype = c_uint

    lib.SetAccumulationCycleTime.argtypes = [c_float]
    lib.SetAccumulationCycleTime.restype = c_uint

    lib.SetKineticCycleTime.argtypes = [c_float]
    lib.SetKineticCycleTime.restype = c_uint

    lib.SetNumberAccumulations.argtypes = [c_int]
    lib.SetNumberAccumulations.restype = c_uint

    lib.SetNumberKinetics.argtypes = [c_int]
    lib.SetNumberKinetics.restype = c_uint

    # Image settings
    lib.SetImage.argtypes = [c_int, c_int, c_int, c_int, c_int, c_int]
    lib.SetImage.restype = c_uint

    lib.SetFullImage.argtypes = [c_int, c_int]
    lib.SetFullImage.restype = c_uint

    lib.SetImageFlip.argtypes = [c_int, c_int]
    lib.SetImageFlip.restype = c_uint

    lib.GetImageFlip.argtypes = [POINTER(c_int), POINTER(c_int)]
    lib.GetImageFlip.restype = c_uint

    lib.SetImageRotate.argtypes = [c_int]
    lib.SetImageRotate.restype = c_uint

    lib.GetImageRotate.argtypes = [POINTER(c_int)]
    lib.GetImageRotate.restype = c_uint

    # Shutter
    lib.SetShutter.argtypes = [c_int, c_int, c_int, c_int]
    lib.SetShutter.restype = c_uint

    lib.SetShutterEx.argtypes = [c_int, c_int, c_int, c_int, c_int]
    lib.SetShutterEx.restype = c_uint

    # Acquisition control
    lib.PrepareAcquisition.argtypes = []
    lib.PrepareAcquisition.restype = c_uint

    lib.StartAcquisition.argtypes = []
    lib.StartAcquisition.restype = c_uint

    lib.AbortAcquisition.argtypes = []
    lib.AbortAcquisition.restype = c_uint

    lib.WaitForAcquisition.argtypes = []
    lib.WaitForAcquisition.restype = c_uint

    lib.WaitForAcquisitionTimeOut.argtypes = [c_int]
    lib.WaitForAcquisitionTimeOut.restype = c_uint

    lib.CancelWait.argtypes = []
    lib.CancelWait.restype = c_uint

    lib.SendSoftwareTrigger.argtypes = []
    lib.SendSoftwareTrigger.restype = c_uint

    # Status
    lib.GetStatus.argtypes = [POINTER(c_int)]
    lib.GetStatus.restype = c_uint

    lib.GetAcquisitionProgress.argtypes = [POINTER(at_32), POINTER(at_32)]
    lib.GetAcquisitionProgress.restype = c_uint

    lib.GetAcquisitionTimings.argtypes = [POINTER(c_float), POINTER(c_float), POINTER(c_float)]
    lib.GetAcquisitionTimings.restype = c_uint

    # Data acquisition
    lib.GetAcquiredData.argtypes = [POINTER(at_32), at_u32]
    lib.GetAcquiredData.restype = c_uint

    lib.GetAcquiredData16.argtypes = [POINTER(c_ushort), at_u32]
    lib.GetAcquiredData16.restype = c_uint

    lib.GetAcquiredFloatData.argtypes = [POINTER(c_float), at_u32]
    lib.GetAcquiredFloatData.restype = c_uint

    lib.GetMostRecentImage.argtypes = [POINTER(at_32), at_u32]
    lib.GetMostRecentImage.restype = c_uint

    lib.GetMostRecentImage16.argtypes = [POINTER(c_ushort), at_u32]
    lib.GetMostRecentImage16.restype = c_uint

    lib.GetOldestImage.argtypes = [POINTER(at_32), at_u32]
    lib.GetOldestImage.restype = c_uint

    lib.GetOldestImage16.argtypes = [POINTER(c_ushort), at_u32]
    lib.GetOldestImage16.restype = c_uint

    lib.GetNumberNewImages.argtypes = [POINTER(at_32), POINTER(at_32)]
    lib.GetNumberNewImages.restype = c_uint

    lib.GetNumberAvailableImages.argtypes = [POINTER(at_32), POINTER(at_32)]
    lib.GetNumberAvailableImages.restype = c_uint

    lib.GetImages.argtypes = [at_32, at_32, POINTER(at_32), at_u32, POINTER(at_32), POINTER(at_32)]
    lib.GetImages.restype = c_uint

    lib.GetImages16.argtypes = [at_32, at_32, POINTER(c_ushort), at_u32, POINTER(at_32), POINTER(at_32)]
    lib.GetImages16.restype = c_uint

    # Speeds
    lib.GetNumberHSSpeeds.argtypes = [c_int, c_int, POINTER(c_int)]
    lib.GetNumberHSSpeeds.restype = c_uint

    lib.GetHSSpeed.argtypes = [c_int, c_int, c_int, POINTER(c_float)]
    lib.GetHSSpeed.restype = c_uint

    lib.SetHSSpeed.argtypes = [c_int, c_int]
    lib.SetHSSpeed.restype = c_uint

    lib.GetNumberVSSpeeds.argtypes = [POINTER(c_int)]
    lib.GetNumberVSSpeeds.restype = c_uint

    lib.GetVSSpeed.argtypes = [c_int, POINTER(c_float)]
    lib.GetVSSpeed.restype = c_uint

    lib.SetVSSpeed.argtypes = [c_int]
    lib.SetVSSpeed.restype = c_uint

    lib.GetFastestRecommendedVSSpeed.argtypes = [POINTER(c_int), POINTER(c_float)]
    lib.GetFastestRecommendedVSSpeed.restype = c_uint

    # VS Amplitude (vertical clock voltage)
    lib.GetNumberVSAmplitudes.argtypes = [POINTER(c_int)]
    lib.GetNumberVSAmplitudes.restype = c_uint

    lib.GetVSAmplitudeValue.argtypes = [c_int, POINTER(c_int)]
    lib.GetVSAmplitudeValue.restype = c_uint

    lib.GetVSAmplitudeString.argtypes = [c_int, c_char_p]
    lib.GetVSAmplitudeString.restype = c_uint

    lib.SetVSAmplitude.argtypes = [c_int]
    lib.SetVSAmplitude.restype = c_uint

    # Gain
    lib.GetNumberPreAmpGains.argtypes = [POINTER(c_int)]
    lib.GetNumberPreAmpGains.restype = c_uint

    lib.GetPreAmpGain.argtypes = [c_int, POINTER(c_float)]
    lib.GetPreAmpGain.restype = c_uint

    lib.SetPreAmpGain.argtypes = [c_int]
    lib.SetPreAmpGain.restype = c_uint

    lib.GetEMCCDGain.argtypes = [POINTER(c_int)]
    lib.GetEMCCDGain.restype = c_uint

    lib.SetEMCCDGain.argtypes = [c_int]
    lib.SetEMCCDGain.restype = c_uint

    lib.GetEMGainRange.argtypes = [POINTER(c_int), POINTER(c_int)]
    lib.GetEMGainRange.restype = c_uint

    lib.SetEMGainMode.argtypes = [c_int]
    lib.SetEMGainMode.restype = c_uint

    lib.SetEMAdvanced.argtypes = [c_int]
    lib.SetEMAdvanced.restype = c_uint

    lib.GetEMAdvanced.argtypes = [POINTER(c_int)]
    lib.GetEMAdvanced.restype = c_uint

    lib.SetEMClockCompensation.argtypes = [c_int]
    lib.SetEMClockCompensation.restype = c_uint

    # Sensor Compensation (EM gain calibration)
    # Note: Only EnableSensorCompensation is exported. The other SensorCompensation_*
    # functions exist in the library but are local symbols (not exported).
    lib.EnableSensorCompensation.argtypes = [c_int]
    lib.EnableSensorCompensation.restype = c_uint

    # AD channel
    lib.GetNumberADChannels.argtypes = [POINTER(c_int)]
    lib.GetNumberADChannels.restype = c_uint

    lib.GetBitDepth.argtypes = [c_int, POINTER(c_int)]
    lib.GetBitDepth.restype = c_uint

    lib.SetADChannel.argtypes = [c_int]
    lib.SetADChannel.restype = c_uint

    # Amplifier
    lib.GetNumberAmp.argtypes = [POINTER(c_int)]
    lib.GetNumberAmp.restype = c_uint

    lib.SetOutputAmplifier.argtypes = [c_int]
    lib.SetOutputAmplifier.restype = c_uint

    lib.GetAmpDesc.argtypes = [c_int, c_char_p, c_int]
    lib.GetAmpDesc.restype = c_uint

    # Frame transfer
    lib.SetFrameTransferMode.argtypes = [c_int]
    lib.SetFrameTransferMode.restype = c_uint

    # Crop modes (for faster frame rates)
    lib.SetCropMode.argtypes = [c_int, c_int, c_int]  # active, cropheight, reserved
    lib.SetCropMode.restype = c_uint

    lib.SetIsolatedCropMode.argtypes = [c_int, c_int, c_int, c_int, c_int]  # active, cropheight, cropwidth, vbin, hbin
    lib.SetIsolatedCropMode.restype = c_uint

    # Note: SetIsolatedCropModeEx not available in SDK 2.91

    # Baseline
    lib.SetBaselineClamp.argtypes = [c_int]
    lib.SetBaselineClamp.restype = c_uint

    lib.GetBaselineClamp.argtypes = [POINTER(c_int)]
    lib.GetBaselineClamp.restype = c_uint

    # Spurious Noise / CIC Filter
    lib.Filter_SetMode.argtypes = [c_uint]
    lib.Filter_SetMode.restype = c_uint

    lib.Filter_GetMode.argtypes = [POINTER(c_uint)]
    lib.Filter_GetMode.restype = c_uint

    lib.Filter_SetThreshold.argtypes = [c_float]
    lib.Filter_SetThreshold.restype = c_uint

    lib.Filter_GetThreshold.argtypes = [POINTER(c_float)]
    lib.Filter_GetThreshold.restype = c_uint

    lib.Filter_SetDataAveragingMode.argtypes = [c_int]
    lib.Filter_SetDataAveragingMode.restype = c_uint

    lib.Filter_GetDataAveragingMode.argtypes = [POINTER(c_int)]
    lib.Filter_GetDataAveragingMode.restype = c_uint

    lib.Filter_SetAveragingFrameCount.argtypes = [c_int]
    lib.Filter_SetAveragingFrameCount.restype = c_uint

    lib.Filter_GetAveragingFrameCount.argtypes = [POINTER(c_int)]
    lib.Filter_GetAveragingFrameCount.restype = c_uint

    lib.Filter_SetAveragingFactor.argtypes = [c_int]
    lib.Filter_SetAveragingFactor.restype = c_uint

    lib.Filter_GetAveragingFactor.argtypes = [POINTER(c_int)]
    lib.Filter_GetAveragingFactor.restype = c_uint

    # Save functions
    lib.SaveAsFITS.argtypes = [c_char_p, c_int]
    lib.SaveAsFITS.restype = c_uint

    lib.SaveAsTiff.argtypes = [c_char_p, c_char_p, c_int, c_int]
    lib.SaveAsTiff.restype = c_uint

    lib.SaveAsSif.argtypes = [c_char_p]
    lib.SaveAsSif.restype = c_uint

    lib.SaveAsRaw.argtypes = [c_char_p, c_int]
    lib.SaveAsRaw.restype = c_uint

    lib.SaveEEPROMToFile.argtypes = [c_char_p]
    lib.SaveEEPROMToFile.restype = c_uint

    # Version info
    lib.GetSoftwareVersion.argtypes = [POINTER(c_uint)] * 6
    lib.GetSoftwareVersion.restype = c_uint

    lib.GetHardwareVersion.argtypes = [POINTER(c_uint)] * 6
    lib.GetHardwareVersion.restype = c_uint


# Load library at module import
_lib = _load_library()


# ============================================================================
# Helper functions
# ============================================================================

def _check_error(code: int, context: str = "") -> None:
    """Check error code and raise appropriate exception."""
    if code == ErrorCode.DRV_SUCCESS:
        return

    # Temperature status codes are not errors
    if code in (
        ErrorCode.DRV_TEMPERATURE_OFF,
        ErrorCode.DRV_TEMPERATURE_NOT_STABILIZED,
        ErrorCode.DRV_TEMPERATURE_STABILIZED,
        ErrorCode.DRV_TEMPERATURE_NOT_REACHED,
        ErrorCode.DRV_TEMPERATURE_DRIFT,
    ):
        return

    if code == ErrorCode.DRV_NOT_INITIALIZED:
        raise AndorNotInitializedError(code, context)
    elif code == ErrorCode.DRV_ACQUIRING:
        raise AndorAcquiringError(code, context)
    elif code in (
        ErrorCode.DRV_TEMPERATURE_OUT_RANGE,
        ErrorCode.DRV_TEMPERATURE_NOT_SUPPORTED,
    ):
        raise AndorTemperatureError(code, context)
    else:
        raise AndorError(code, context)


def get_available_cameras() -> int:
    """Get the number of available Andor cameras."""
    count = at_32()
    _check_error(_lib.GetAvailableCameras(byref(count)))
    return count.value


# ============================================================================
# Main Camera Class
# ============================================================================

class AndorCamera:
    """
    High-level interface to an Andor camera.

    This class provides a Pythonic interface to control Andor CCD, EMCCD,
    and ICCD cameras. It supports context manager protocol for proper
    resource management.

    Parameters
    ----------
    camera_index : int, optional
        Index of the camera to use (0-based). Default is 0.
    init_dir : str, optional
        Directory containing detector configuration files.
        Default is "" which uses the library's default location.

    Examples
    --------
    >>> with AndorCamera() as cam:
    ...     cam.set_temperature(-60)
    ...     cam.cooler_on()
    ...     print(f"Detector: {cam.width}x{cam.height}")
    ...     cam.set_exposure_time(1.0)
    ...     cam.start_acquisition()
    ...     cam.wait_for_acquisition()
    ...     image = cam.get_acquired_data()
    """

    def __init__(self, camera_index: int = 0, init_dir: str = "/usr/local/etc/andor"):
        self._initialized = False
        self._handle = at_32()
        self._width = 0
        self._height = 0
        self._camera_index = camera_index

        # Get camera handle if multiple cameras
        num_cameras = get_available_cameras()
        if camera_index >= num_cameras:
            raise AndorError(
                ErrorCode.DRV_ERROR_NOCAMERA,
                f"Camera index {camera_index} out of range (0-{num_cameras-1})"
            )

        if num_cameras > 1:
            _check_error(
                _lib.GetCameraHandle(at_32(camera_index), byref(self._handle)),
                "GetCameraHandle"
            )
            _check_error(_lib.SetCurrentCamera(self._handle), "SetCurrentCamera")

        # Initialize
        init_path = init_dir.encode() if init_dir else b""
        code = _lib.Initialize(init_path)
        _check_error(code, "Initialize")
        self._initialized = True

        # Get detector size
        width = c_int()
        height = c_int()
        _check_error(_lib.GetDetector(byref(width), byref(height)), "GetDetector")
        self._width = width.value
        self._height = height.value

    def __enter__(self) -> "AndorCamera":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def close(self) -> None:
        """Shut down the camera and release resources."""
        if self._initialized:
            self.abort_acquisition()
            _lib.ShutDown()
            self._initialized = False

    # ========================================================================
    # Properties
    # ========================================================================

    @property
    def width(self) -> int:
        """Detector width in pixels."""
        return self._width

    @property
    def height(self) -> int:
        """Detector height in pixels."""
        return self._height

    @property
    def serial_number(self) -> int:
        """Camera serial number."""
        serial = c_int()
        _check_error(_lib.GetCameraSerialNumber(byref(serial)), "GetCameraSerialNumber")
        return serial.value

    @property
    def head_model(self) -> str:
        """Camera head model name."""
        model = create_string_buffer(256)
        _check_error(_lib.GetHeadModel(model), "GetHeadModel")
        return model.value.decode()

    @property
    def pixel_size(self) -> Tuple[float, float]:
        """Pixel size in microns (x, y)."""
        x_size = c_float()
        y_size = c_float()
        _check_error(_lib.GetPixelSize(byref(x_size), byref(y_size)), "GetPixelSize")
        return (x_size.value, y_size.value)

    @property
    def status(self) -> CameraStatus:
        """Current camera status."""
        status = c_int()
        _check_error(_lib.GetStatus(byref(status)), "GetStatus")
        return CameraStatus(status.value)

    @property
    def is_acquiring(self) -> bool:
        """True if camera is currently acquiring."""
        return self.status == CameraStatus.ACQUIRING

    # ========================================================================
    # Capabilities
    # ========================================================================

    def get_capabilities(self) -> AndorCapabilities:
        """Get camera capabilities structure."""
        caps = AndorCapabilities()
        caps.ulSize = ctypes.sizeof(AndorCapabilities)
        _check_error(_lib.GetCapabilities(byref(caps)), "GetCapabilities")
        return caps

    def get_camera_type(self) -> CameraType:
        """Get camera type."""
        caps = self.get_capabilities()
        return CameraType(caps.ulCameraType)

    # ========================================================================
    # Temperature Control
    # ========================================================================

    def get_temperature(self) -> Tuple[float, int]:
        """
        Get current sensor temperature.

        Returns
        -------
        temperature : float
            Current temperature in degrees Celsius.
        status_code : int
            Temperature status code (see ErrorCode for meanings).
        """
        temp = c_float()
        code = _lib.GetTemperatureF(byref(temp))
        return (temp.value, code)

    def get_temperature_range(self) -> Tuple[int, int]:
        """Get valid temperature range (min, max) in degrees Celsius."""
        min_temp = c_int()
        max_temp = c_int()
        _check_error(
            _lib.GetTemperatureRange(byref(min_temp), byref(max_temp)),
            "GetTemperatureRange"
        )
        return (min_temp.value, max_temp.value)

    def set_temperature(self, temperature: int) -> None:
        """Set target temperature in degrees Celsius."""
        print(f"[CAM] Temperature: {temperature}°C")
        _check_error(_lib.SetTemperature(c_int(temperature)), "SetTemperature")

    def cooler_on(self) -> None:
        """Turn on the cooler."""
        print("[CAM] Cooler: ON")
        _check_error(_lib.CoolerON(), "CoolerON")

    def cooler_off(self) -> None:
        """Turn off the cooler."""
        print("[CAM] Cooler: OFF")
        _check_error(_lib.CoolerOFF(), "CoolerOFF")

    def is_cooler_on(self) -> bool:
        """Check if cooler is on."""
        status = c_int()
        _check_error(_lib.IsCoolerOn(byref(status)), "IsCoolerOn")
        return bool(status.value)

    def get_tec_status(self) -> bool:
        """
        Check TEC (thermoelectric cooler) status.

        Returns
        -------
        bool
            True if TEC has overheated (error condition), False if OK.
        """
        flag = c_int()
        _check_error(_lib.GetTECStatus(byref(flag)), "GetTECStatus")
        return bool(flag.value)

    def set_cooler_mode(self, maintain: bool) -> None:
        """
        Set whether cooler stays on after ShutDown.

        Parameters
        ----------
        maintain : bool
            If True (1), cooler is maintained on after ShutDown.
            If False (0), cooler is turned off during ShutDown.
        """
        mode = 1 if maintain else 0
        _check_error(_lib.SetCoolerMode(c_int(mode)), "SetCoolerMode")

    def set_fan_mode(self, mode: FanMode) -> None:
        """Set fan operating mode."""
        _check_error(_lib.SetFanMode(c_int(mode)), "SetFanMode")

    # ========================================================================
    # Acquisition Setup
    # ========================================================================

    def set_acquisition_mode(self, mode: AcquisitionMode) -> None:
        """Set acquisition mode."""
        _check_error(_lib.SetAcquisitionMode(c_int(mode)), "SetAcquisitionMode")

    def set_read_mode(self, mode: ReadMode) -> None:
        """Set read mode."""
        _check_error(_lib.SetReadMode(c_int(mode)), "SetReadMode")

    def set_trigger_mode(self, mode: TriggerMode) -> None:
        """Set trigger mode."""
        _check_error(_lib.SetTriggerMode(c_int(mode)), "SetTriggerMode")

    def set_exposure_time(self, seconds: float) -> None:
        """Set exposure time in seconds."""
        print(f"[CAM] Exposure: {seconds:.4f} sec")
        _check_error(_lib.SetExposureTime(c_float(seconds)), "SetExposureTime")

    def set_accumulation_cycle_time(self, seconds: float) -> None:
        """Set accumulation cycle time in seconds."""
        _check_error(
            _lib.SetAccumulationCycleTime(c_float(seconds)),
            "SetAccumulationCycleTime"
        )

    def set_kinetic_cycle_time(self, seconds: float) -> None:
        """Set kinetic cycle time in seconds."""
        _check_error(
            _lib.SetKineticCycleTime(c_float(seconds)),
            "SetKineticCycleTime"
        )

    def set_number_accumulations(self, number: int) -> None:
        """Set number of accumulations."""
        _check_error(
            _lib.SetNumberAccumulations(c_int(number)),
            "SetNumberAccumulations"
        )

    def set_number_kinetics(self, number: int) -> None:
        """Set number of kinetic series frames."""
        _check_error(_lib.SetNumberKinetics(c_int(number)), "SetNumberKinetics")

    def get_acquisition_timings(self) -> Tuple[float, float, float]:
        """
        Get actual acquisition timings.

        Returns
        -------
        exposure : float
            Actual exposure time in seconds.
        accumulate : float
            Accumulate cycle time in seconds.
        kinetic : float
            Kinetic cycle time in seconds.
        """
        exposure = c_float()
        accumulate = c_float()
        kinetic = c_float()
        _check_error(
            _lib.GetAcquisitionTimings(
                byref(exposure), byref(accumulate), byref(kinetic)
            ),
            "GetAcquisitionTimings"
        )
        return (exposure.value, accumulate.value, kinetic.value)

    # ========================================================================
    # Image Settings
    # ========================================================================

    def set_image(
        self,
        hbin: int = 1,
        vbin: int = 1,
        hstart: int = 1,
        hend: Optional[int] = None,
        vstart: int = 1,
        vend: Optional[int] = None
    ) -> None:
        """
        Set image format and binning.

        Parameters
        ----------
        hbin : int
            Horizontal binning factor.
        vbin : int
            Vertical binning factor.
        hstart : int
            Horizontal start pixel (1-based).
        hend : int, optional
            Horizontal end pixel. Default is detector width.
        vstart : int
            Vertical start pixel (1-based).
        vend : int, optional
            Vertical end pixel. Default is detector height.
        """
        if hend is None:
            hend = self._width
        if vend is None:
            vend = self._height

        _check_error(
            _lib.SetImage(
                c_int(hbin), c_int(vbin),
                c_int(hstart), c_int(hend),
                c_int(vstart), c_int(vend)
            ),
            "SetImage"
        )

    def set_full_image(self, hbin: int = 1, vbin: int = 1) -> None:
        """Set full image with specified binning."""
        _check_error(
            _lib.SetFullImage(c_int(hbin), c_int(vbin)),
            "SetFullImage"
        )

    def set_image_flip(self, horizontal: bool = False, vertical: bool = False) -> None:
        """Set image flip options."""
        _check_error(
            _lib.SetImageFlip(c_int(horizontal), c_int(vertical)),
            "SetImageFlip"
        )

    def get_image_flip(self) -> Tuple[bool, bool]:
        """Get current image flip settings (horizontal, vertical)."""
        h_flip = c_int()
        v_flip = c_int()
        _check_error(_lib.GetImageFlip(byref(h_flip), byref(v_flip)), "GetImageFlip")
        return (bool(h_flip.value), bool(v_flip.value))

    def set_image_rotate(self, rotation: int) -> None:
        """Set image rotation (0, 1, 2, or 3 for 0, 90, 180, 270 degrees)."""
        _check_error(_lib.SetImageRotate(c_int(rotation)), "SetImageRotate")

    # ========================================================================
    # Shutter Control
    # ========================================================================

    def set_shutter(
        self,
        shutter_type: ShutterType = ShutterType.TTL_HIGH,
        mode: ShutterMode = ShutterMode.AUTO,
        closing_time: int = 0,
        opening_time: int = 0
    ) -> None:
        """
        Configure the shutter.

        Parameters
        ----------
        shutter_type : ShutterType
            TTL signal type for external shutter.
        mode : ShutterMode
            Shutter operating mode.
        closing_time : int
            Time to close shutter in milliseconds.
        opening_time : int
            Time to open shutter in milliseconds.
        """
        mode_names = {0: "AUTO", 1: "OPEN", 2: "CLOSED"}
        print(f"[CAM] Shutter: mode={mode_names.get(int(mode), mode)}, close={closing_time}ms, open={opening_time}ms")
        _check_error(
            _lib.SetShutter(
                c_int(shutter_type), c_int(mode),
                c_int(closing_time), c_int(opening_time)
            ),
            "SetShutter"
        )

    # ========================================================================
    # Acquisition Control
    # ========================================================================

    def prepare_acquisition(self) -> None:
        """Prepare for acquisition (allocates buffers)."""
        _check_error(_lib.PrepareAcquisition(), "PrepareAcquisition")

    def start_acquisition(self) -> None:
        """Start acquisition."""
        _check_error(_lib.StartAcquisition(), "StartAcquisition")

    def abort_acquisition(self) -> None:
        """Abort current acquisition."""
        code = _lib.AbortAcquisition()
        # Don't raise error if not acquiring
        if code != ErrorCode.DRV_IDLE:
            _check_error(code, "AbortAcquisition")

    def wait_for_acquisition(self, timeout_ms: Optional[int] = None) -> None:
        """
        Wait for acquisition to complete.

        Parameters
        ----------
        timeout_ms : int, optional
            Timeout in milliseconds. If None, waits indefinitely.
        """
        if timeout_ms is not None:
            _check_error(
                _lib.WaitForAcquisitionTimeOut(c_int(timeout_ms)),
                "WaitForAcquisitionTimeOut"
            )
        else:
            _check_error(_lib.WaitForAcquisition(), "WaitForAcquisition")

    def cancel_wait(self) -> None:
        """Cancel a WaitForAcquisition call."""
        _check_error(_lib.CancelWait(), "CancelWait")

    def send_software_trigger(self) -> None:
        """Send a software trigger."""
        _check_error(_lib.SendSoftwareTrigger(), "SendSoftwareTrigger")

    def get_acquisition_progress(self) -> Tuple[int, int]:
        """
        Get acquisition progress.

        Returns
        -------
        accumulations : int
            Number of accumulations completed.
        series : int
            Number of kinetic series frames completed.
        """
        acc = at_32()
        series = at_32()
        _check_error(
            _lib.GetAcquisitionProgress(byref(acc), byref(series)),
            "GetAcquisitionProgress"
        )
        return (acc.value, series.value)

    # ========================================================================
    # Data Retrieval
    # ========================================================================

    def get_acquired_data(self, dtype: str = "int32") -> np.ndarray:
        """
        Get acquired image data.

        Parameters
        ----------
        dtype : str
            Data type: "int32", "uint16", or "float32".

        Returns
        -------
        data : np.ndarray
            Image data as 2D numpy array.
        """
        size = self._width * self._height

        if dtype == "uint16":
            buffer = (c_ushort * size)()
            _check_error(
                _lib.GetAcquiredData16(buffer, at_u32(size)),
                "GetAcquiredData16"
            )
            return np.ctypeslib.as_array(buffer).reshape(self._height, self._width).copy()
        elif dtype == "float32":
            buffer = (c_float * size)()
            _check_error(
                _lib.GetAcquiredFloatData(buffer, at_u32(size)),
                "GetAcquiredFloatData"
            )
            return np.ctypeslib.as_array(buffer).reshape(self._height, self._width).copy()
        else:  # int32
            buffer = (at_32 * size)()
            _check_error(
                _lib.GetAcquiredData(buffer, at_u32(size)),
                "GetAcquiredData"
            )
            return np.ctypeslib.as_array(buffer).reshape(self._height, self._width).copy()

    def get_most_recent_image(self, dtype: str = "uint16") -> np.ndarray:
        """Get most recent image from circular buffer."""
        size = self._width * self._height

        if dtype == "uint16":
            buffer = (c_ushort * size)()
            _check_error(
                _lib.GetMostRecentImage16(buffer, at_u32(size)),
                "GetMostRecentImage16"
            )
        else:
            buffer = (at_32 * size)()
            _check_error(
                _lib.GetMostRecentImage(buffer, at_u32(size)),
                "GetMostRecentImage"
            )

        return np.ctypeslib.as_array(buffer).reshape(self._height, self._width).copy()

    def get_oldest_image(self, dtype: str = "uint16") -> np.ndarray:
        """Get oldest image from circular buffer."""
        size = self._width * self._height

        if dtype == "uint16":
            buffer = (c_ushort * size)()
            _check_error(
                _lib.GetOldestImage16(buffer, at_u32(size)),
                "GetOldestImage16"
            )
        else:
            buffer = (at_32 * size)()
            _check_error(
                _lib.GetOldestImage(buffer, at_u32(size)),
                "GetOldestImage"
            )

        return np.ctypeslib.as_array(buffer).reshape(self._height, self._width).copy()

    def get_number_new_images(self) -> Tuple[int, int]:
        """Get range of new images available (first, last)."""
        first = at_32()
        last = at_32()
        _check_error(
            _lib.GetNumberNewImages(byref(first), byref(last)),
            "GetNumberNewImages"
        )
        return (first.value, last.value)

    def get_images(
        self,
        first: int,
        last: int,
        dtype: str = "uint16"
    ) -> Tuple[np.ndarray, int, int]:
        """
        Get a range of images from the circular buffer.

        Parameters
        ----------
        first : int
            First image index.
        last : int
            Last image index.
        dtype : str
            Data type: "int32" or "uint16".

        Returns
        -------
        images : np.ndarray
            3D array of images (num_images, height, width).
        valid_first : int
            First valid image index.
        valid_last : int
            Last valid image index.
        """
        num_images = last - first + 1
        size = self._width * self._height * num_images
        valid_first = at_32()
        valid_last = at_32()

        if dtype == "uint16":
            buffer = (c_ushort * size)()
            _check_error(
                _lib.GetImages16(
                    at_32(first), at_32(last),
                    buffer, at_u32(size),
                    byref(valid_first), byref(valid_last)
                ),
                "GetImages16"
            )
        else:
            buffer = (at_32 * size)()
            _check_error(
                _lib.GetImages(
                    at_32(first), at_32(last),
                    buffer, at_u32(size),
                    byref(valid_first), byref(valid_last)
                ),
                "GetImages"
            )

        data = np.ctypeslib.as_array(buffer).reshape(num_images, self._height, self._width).copy()
        return (data, valid_first.value, valid_last.value)

    # ========================================================================
    # Speed Settings
    # ========================================================================

    def get_number_hs_speeds(self, channel: int = 0, output_amp: int = 0) -> int:
        """Get number of horizontal shift speeds available."""
        num = c_int()
        _check_error(
            _lib.GetNumberHSSpeeds(c_int(channel), c_int(output_amp), byref(num)),
            "GetNumberHSSpeeds"
        )
        return num.value

    def get_hs_speed(self, index: int, channel: int = 0, output_amp: int = 0) -> float:
        """Get horizontal shift speed in MHz for given index."""
        speed = c_float()
        _check_error(
            _lib.GetHSSpeed(c_int(channel), c_int(output_amp), c_int(index), byref(speed)),
            "GetHSSpeed"
        )
        return speed.value

    def set_hs_speed(self, index: int, output_amp: int = 0) -> None:
        """Set horizontal shift speed by index."""
        print(f"[CAM] HS Speed: amp={output_amp}, index={index}")
        _check_error(_lib.SetHSSpeed(c_int(output_amp), c_int(index)), "SetHSSpeed")

    def get_number_vs_speeds(self) -> int:
        """Get number of vertical shift speeds available."""
        num = c_int()
        _check_error(_lib.GetNumberVSSpeeds(byref(num)), "GetNumberVSSpeeds")
        return num.value

    def get_vs_speed(self, index: int) -> float:
        """Get vertical shift speed in microseconds for given index."""
        speed = c_float()
        _check_error(_lib.GetVSSpeed(c_int(index), byref(speed)), "GetVSSpeed")
        return speed.value

    def set_vs_speed(self, index: int) -> None:
        """Set vertical shift speed by index."""
        print(f"[CAM] VS Speed: index={index}")
        _check_error(_lib.SetVSSpeed(c_int(index)), "SetVSSpeed")

    def get_fastest_recommended_vs_speed(self) -> Tuple[int, float]:
        """Get fastest recommended vertical shift speed (index, speed)."""
        index = c_int()
        speed = c_float()
        _check_error(
            _lib.GetFastestRecommendedVSSpeed(byref(index), byref(speed)),
            "GetFastestRecommendedVSSpeed"
        )
        return (index.value, speed.value)

    # ========================================================================
    # Vertical Clock Voltage (VS Amplitude)
    # ========================================================================

    def get_number_vs_amplitudes(self) -> int:
        """Get number of vertical clock voltage amplitudes available."""
        num = c_int()
        _check_error(_lib.GetNumberVSAmplitudes(byref(num)), "GetNumberVSAmplitudes")
        return num.value

    def get_vs_amplitude_value(self, index: int) -> int:
        """Get vertical clock voltage amplitude value for given index."""
        value = c_int()
        _check_error(
            _lib.GetVSAmplitudeValue(c_int(index), byref(value)),
            "GetVSAmplitudeValue"
        )
        return value.value

    def get_vs_amplitude_string(self, index: int) -> str:
        """Get vertical clock voltage amplitude description for given index."""
        desc = create_string_buffer(256)
        _check_error(
            _lib.GetVSAmplitudeString(c_int(index), desc),
            "GetVSAmplitudeString"
        )
        return desc.value.decode()

    def set_vs_amplitude(self, index: int) -> None:
        """
        Set vertical clock voltage amplitude.

        Higher amplitudes can help with charge transfer efficiency at faster
        vertical shift speeds, but may increase clock-induced charge (CIC).

        Parameters
        ----------
        index : int
            Amplitude index (0 = Normal, higher = increased amplitude).
            Use get_number_vs_amplitudes() to find available options.
        """
        _check_error(_lib.SetVSAmplitude(c_int(index)), "SetVSAmplitude")

    def get_vs_amplitudes(self) -> List[str]:
        """
        Get list of all available vertical clock voltage amplitude descriptions.

        Returns
        -------
        amplitudes : list of str
            List of amplitude descriptions (e.g., ["Normal", "+1", "+2", ...]).
        """
        num = self.get_number_vs_amplitudes()
        amplitudes = []
        for i in range(num):
            try:
                amplitudes.append(self.get_vs_amplitude_string(i))
            except AndorError:
                amplitudes.append(f"Level {i}")
        return amplitudes

    # ========================================================================
    # Gain Settings
    # ========================================================================

    def get_number_preamp_gains(self) -> int:
        """Get number of pre-amplifier gains available."""
        num = c_int()
        _check_error(_lib.GetNumberPreAmpGains(byref(num)), "GetNumberPreAmpGains")
        return num.value

    def get_preamp_gain(self, index: int) -> float:
        """Get pre-amplifier gain factor for given index."""
        gain = c_float()
        _check_error(_lib.GetPreAmpGain(c_int(index), byref(gain)), "GetPreAmpGain")
        return gain.value

    def set_preamp_gain(self, index: int) -> None:
        """Set pre-amplifier gain by index."""
        print(f"[CAM] Preamp Gain: index={index}")
        _check_error(_lib.SetPreAmpGain(c_int(index)), "SetPreAmpGain")

    def get_emccd_gain(self) -> int:
        """Get current EM gain value."""
        gain = c_int()
        _check_error(_lib.GetEMCCDGain(byref(gain)), "GetEMCCDGain")
        return gain.value

    def set_emccd_gain(self, gain: int) -> None:
        """
        Set EM gain value.

        Parameters
        ----------
        gain : int
            EM gain value. Use get_em_gain_range() to find valid range.
            Typical range is 0-300 (or 0-1000 with advanced mode enabled).
        """
        print(f"[CAM] EM Gain: {gain}")
        _check_error(_lib.SetEMCCDGain(c_int(gain)), "SetEMCCDGain")

    # Alias for consistency
    em_gain = property(get_emccd_gain, set_emccd_gain, doc="Current EM gain value (read/write property).")

    def get_em_gain_range(self) -> Tuple[int, int]:
        """Get valid EM gain range (low, high)."""
        low = c_int()
        high = c_int()
        _check_error(_lib.GetEMGainRange(byref(low), byref(high)), "GetEMGainRange")
        return (low.value, high.value)

    def set_em_gain_mode(self, mode: int) -> None:
        """
        Set EM gain mode.

        Parameters
        ----------
        mode : int
            0 = DAC 8-bit (default, 0-255 range)
            1 = DAC 12-bit (0-4095 range)
            2 = Linear mode
            3 = Real EM gain (calibrated, actual multiplication factor)
        """
        mode_names = {0: "8-bit DAC", 1: "12-bit DAC", 2: "Linear", 3: "Real EM"}
        print(f"[CAM] EM Gain Mode: {mode} ({mode_names.get(mode, 'unknown')})")
        _check_error(_lib.SetEMGainMode(c_int(mode)), "SetEMGainMode")

    def set_em_advanced(self, enabled: bool) -> None:
        """
        Enable/disable access to higher EM gain levels.

        WARNING: Higher EM gain levels can cause accelerated aging of the
        EM register. Use with caution and only when necessary.

        Parameters
        ----------
        enabled : bool
            If True, allows access to gain levels above 300.
        """
        print(f"[CAM] EM Advanced: {enabled}")
        _check_error(_lib.SetEMAdvanced(c_int(enabled)), "SetEMAdvanced")

    def get_em_advanced(self) -> bool:
        """Check if EM advanced mode (high gain) is enabled."""
        state = c_int()
        _check_error(_lib.GetEMAdvanced(byref(state)), "GetEMAdvanced")
        return bool(state.value)

    def set_em_clock_compensation(self, enabled: bool) -> None:
        """
        Enable/disable EM clock compensation for register aging.

        The EM register degrades with use over time, reducing gain.
        This feature compensates by adjusting clock voltages.

        Parameters
        ----------
        enabled : bool
            If True, enables automatic aging compensation.
        """
        print(f"[CAM] EM Clock Compensation: {enabled}")
        _check_error(_lib.SetEMClockCompensation(c_int(enabled)), "SetEMClockCompensation")

    # ========================================================================
    # Sensor Compensation (EM Gain Calibration)
    # ========================================================================

    def enable_sensor_compensation(self, enabled: bool) -> None:
        """
        Enable/disable sensor compensation (EM gain calibration).

        When enabled, the camera uses stored calibration coefficients to
        apply temperature-dependent gain correction.
        """
        print(f"[CAM] Sensor Compensation: {enabled}")
        _check_error(_lib.EnableSensorCompensation(c_int(enabled)), "EnableSensorCompensation")

    def configure_em_gain(
        self,
        gain: int,
        mode: int = 0,
        advanced: bool = False
    ) -> None:
        """
        Convenience method to configure EM gain settings.

        Parameters
        ----------
        gain : int
            EM gain value to set.
        mode : int
            EM gain mode (0=8-bit DAC, 1=12-bit DAC, 2=Linear, 3=Real).
        advanced : bool
            If True, enable access to higher gain levels (>300).
        """
        if advanced:
            self.set_em_advanced(True)
        self.set_em_gain_mode(mode)
        self.set_emccd_gain(gain)

    # ========================================================================
    # AD Channel Settings
    # ========================================================================

    def get_number_ad_channels(self) -> int:
        """Get number of AD channels available."""
        num = c_int()
        _check_error(_lib.GetNumberADChannels(byref(num)), "GetNumberADChannels")
        return num.value

    def get_bit_depth(self, channel: int = 0) -> int:
        """Get bit depth for specified AD channel."""
        depth = c_int()
        _check_error(_lib.GetBitDepth(c_int(channel), byref(depth)), "GetBitDepth")
        return depth.value

    def set_ad_channel(self, channel: int) -> None:
        """Set AD channel."""
        _check_error(_lib.SetADChannel(c_int(channel)), "SetADChannel")

    # ========================================================================
    # Output Amplifier
    # ========================================================================

    def get_number_amplifiers(self) -> int:
        """Get number of output amplifiers available."""
        num = c_int()
        _check_error(_lib.GetNumberAmp(byref(num)), "GetNumberAmp")
        return num.value

    def get_amplifier_description(self, index: int) -> str:
        """Get description of output amplifier."""
        desc = create_string_buffer(256)
        _check_error(_lib.GetAmpDesc(c_int(index), desc, 256), "GetAmpDesc")
        return desc.value.decode()

    def set_output_amplifier(self, index: int) -> None:
        """Set output amplifier (0 = EMCCD, 1 = Conventional)."""
        amp_names = {0: "EMCCD", 1: "Conventional"}
        print(f"[CAM] Output Amplifier: {index} ({amp_names.get(index, 'unknown')})")
        _check_error(_lib.SetOutputAmplifier(c_int(index)), "SetOutputAmplifier")

    # ========================================================================
    # Frame Transfer
    # ========================================================================

    def set_frame_transfer_mode(self, enabled: bool) -> None:
        """Enable/disable frame transfer mode."""
        print(f"[CAM] Frame Transfer: {enabled}")
        _check_error(
            _lib.SetFrameTransferMode(c_int(enabled)),
            "SetFrameTransferMode"
        )

    def set_crop_mode(self, active: bool, crop_height: int) -> None:
        """
        Enable/disable crop mode for faster frame rates.

        Parameters
        ----------
        active : bool
            Enable (True) or disable (False) crop mode.
        crop_height : int
            Height of the crop region in pixels.
        """
        _check_error(
            _lib.SetCropMode(c_int(active), c_int(crop_height), c_int(0)),
            "SetCropMode"
        )

    def set_isolated_crop_mode(
        self, active: bool, crop_height: int, crop_width: int,
        vbin: int = 1, hbin: int = 1
    ) -> None:
        """
        Enable/disable isolated crop mode for fastest frame rates.

        Isolated crop mode reads out only the specified region, ignoring
        the rest of the sensor for maximum speed.

        Parameters
        ----------
        active : bool
            Enable (True) or disable (False) isolated crop mode.
        crop_height : int
            Height of the crop region in pixels.
        crop_width : int
            Width of the crop region in pixels.
        vbin : int
            Vertical binning factor (default 1).
        hbin : int
            Horizontal binning factor (default 1).
        """
        _check_error(
            _lib.SetIsolatedCropMode(
                c_int(active), c_int(crop_height), c_int(crop_width),
                c_int(vbin), c_int(hbin)
            ),
            "SetIsolatedCropMode"
        )

    # Note: set_isolated_crop_mode_ex not available in SDK 2.91

    # ========================================================================
    # Baseline
    # ========================================================================

    def set_baseline_clamp(self, enabled: bool) -> None:
        """Enable/disable baseline clamp."""
        _check_error(_lib.SetBaselineClamp(c_int(enabled)), "SetBaselineClamp")

    def get_baseline_clamp(self) -> bool:
        """Get baseline clamp state."""
        state = c_int()
        _check_error(_lib.GetBaselineClamp(byref(state)), "GetBaselineClamp")
        return bool(state.value)

    # ========================================================================
    # Spurious Noise / CIC Filter
    # ========================================================================

    def get_filter_mode(self) -> SpuriousNoiseFilterMode:
        """
        Get current spurious noise (CIC) filter mode.

        Returns
        -------
        mode : SpuriousNoiseFilterMode
            OFF (0), MEDIAN (1), or LEVEL_ABOVE (2).
        """
        mode = c_uint()
        _check_error(_lib.Filter_GetMode(byref(mode)), "Filter_GetMode")
        return SpuriousNoiseFilterMode(mode.value)

    def set_filter_mode(self, mode: Union[SpuriousNoiseFilterMode, int]) -> None:
        """
        Set spurious noise (CIC) filter mode.

        Parameters
        ----------
        mode : SpuriousNoiseFilterMode or int
            OFF (0) - No filtering (default)
            MEDIAN (1) - Median filter for CIC removal
            LEVEL_ABOVE (2) - Threshold-based filter
        """
        _check_error(_lib.Filter_SetMode(c_uint(int(mode))), "Filter_SetMode")

    def get_filter_threshold(self) -> float:
        """
        Get spurious noise filter threshold.

        Returns
        -------
        threshold : float
            Current threshold value (default: 0.0).
        """
        threshold = c_float()
        _check_error(_lib.Filter_GetThreshold(byref(threshold)), "Filter_GetThreshold")
        return threshold.value

    def set_filter_threshold(self, threshold: float) -> None:
        """
        Set spurious noise filter threshold (for LEVEL_ABOVE mode).

        Parameters
        ----------
        threshold : float
            Threshold value. Pixels above this threshold relative to
            neighbors are considered spurious noise.
        """
        _check_error(_lib.Filter_SetThreshold(c_float(threshold)), "Filter_SetThreshold")

    def get_data_averaging_mode(self) -> DataAveragingMode:
        """
        Get data averaging mode.

        Returns
        -------
        mode : DataAveragingMode
            OFF (0), RECURSIVE (1), or FRAME (2).
        """
        mode = c_int()
        _check_error(_lib.Filter_GetDataAveragingMode(byref(mode)), "Filter_GetDataAveragingMode")
        return DataAveragingMode(mode.value)

    def set_data_averaging_mode(self, mode: Union[DataAveragingMode, int]) -> None:
        """
        Set data averaging mode for noise reduction.

        Parameters
        ----------
        mode : DataAveragingMode or int
            OFF (0) - No averaging (default)
            RECURSIVE (1) - Recursive averaging using averaging factor
            FRAME (2) - Average over fixed number of frames
        """
        _check_error(_lib.Filter_SetDataAveragingMode(c_int(int(mode))), "Filter_SetDataAveragingMode")

    def get_averaging_frame_count(self) -> int:
        """Get number of frames to average (for FRAME mode)."""
        count = c_int()
        _check_error(_lib.Filter_GetAveragingFrameCount(byref(count)), "Filter_GetAveragingFrameCount")
        return count.value

    def set_averaging_frame_count(self, count: int) -> None:
        """
        Set number of frames to average (for FRAME mode).

        Parameters
        ----------
        count : int
            Number of frames to average (default: 1).
        """
        _check_error(_lib.Filter_SetAveragingFrameCount(c_int(count)), "Filter_SetAveragingFrameCount")

    def get_averaging_factor(self) -> int:
        """Get recursive averaging factor."""
        factor = c_int()
        _check_error(_lib.Filter_GetAveragingFactor(byref(factor)), "Filter_GetAveragingFactor")
        return factor.value

    def set_averaging_factor(self, factor: int) -> None:
        """
        Set recursive averaging factor (for RECURSIVE mode).

        Parameters
        ----------
        factor : int
            Averaging factor (default: 1). Higher values = more smoothing.
        """
        _check_error(_lib.Filter_SetAveragingFactor(c_int(factor)), "Filter_SetAveragingFactor")

    def configure_cic_filter(
        self,
        mode: Union[SpuriousNoiseFilterMode, int] = SpuriousNoiseFilterMode.OFF,
        threshold: float = 0.0
    ) -> None:
        """
        Convenience method to configure CIC (spurious noise) filtering.

        Parameters
        ----------
        mode : SpuriousNoiseFilterMode or int
            Filter mode: OFF (0), MEDIAN (1), or LEVEL_ABOVE (2).
        threshold : float
            Threshold for LEVEL_ABOVE mode (default: 0.0).

        Examples
        --------
        >>> cam.configure_cic_filter(SpuriousNoiseFilterMode.MEDIAN)
        >>> cam.configure_cic_filter(SpuriousNoiseFilterMode.LEVEL_ABOVE, threshold=10.0)
        """
        self.set_filter_mode(mode)
        if mode == SpuriousNoiseFilterMode.LEVEL_ABOVE:
            self.set_filter_threshold(threshold)

    def get_filter_settings(self) -> dict:
        """
        Get all current filter settings.

        Returns
        -------
        settings : dict
            Dictionary with all filter parameters.
        """
        return {
            "filter_mode": self.get_filter_mode().name,
            "filter_threshold": self.get_filter_threshold(),
            "averaging_mode": self.get_data_averaging_mode().name,
            "averaging_frame_count": self.get_averaging_frame_count(),
            "averaging_factor": self.get_averaging_factor(),
        }

    # ========================================================================
    # File Saving
    # ========================================================================

    def save_as_fits(self, filename: str, data_type: int = 0) -> None:
        """
        Save acquired data as FITS file.

        Parameters
        ----------
        filename : str
            Output filename.
        data_type : int
            0 = signed 32-bit, 1 = unsigned 16-bit.
        """
        _check_error(
            _lib.SaveAsFITS(filename.encode(), c_int(data_type)),
            "SaveAsFITS"
        )

    def save_as_tiff(
        self,
        filename: str,
        palette: str = "",
        position: int = 1,
        data_type: int = 0
    ) -> None:
        """
        Save acquired data as TIFF file.

        Parameters
        ----------
        filename : str
            Output filename.
        palette : str
            Path to palette file (empty for grayscale).
        position : int
            Image position for multi-image file.
        data_type : int
            0 = unsigned 16-bit, 1 = unsigned 32-bit.
        """
        _check_error(
            _lib.SaveAsTiff(
                filename.encode(),
                palette.encode() if palette else b"",
                c_int(position),
                c_int(data_type)
            ),
            "SaveAsTiff"
        )

    def save_as_sif(self, filename: str) -> None:
        """Save acquired data as Andor SIF file."""
        _check_error(_lib.SaveAsSif(filename.encode()), "SaveAsSif")

    def save_as_raw(self, filename: str, data_type: int = 0) -> None:
        """
        Save acquired data as raw binary file.

        Parameters
        ----------
        filename : str
            Output filename.
        data_type : int
            0 = signed 32-bit, 1 = unsigned 16-bit.
        """
        _check_error(_lib.SaveAsRaw(filename.encode(), c_int(data_type)), "SaveAsRaw")

    def save_eeprom_to_file(self, filename: str) -> None:
        """
        Save camera EEPROM contents to a file.

        The EEPROM contains factory calibration data including EM gain
        calibration curves, sensor parameters, and serial number.

        Parameters
        ----------
        filename : str
            Output filename for EEPROM backup.
        """
        print(f"[CAM] Saving EEPROM to: {filename}")
        _check_error(_lib.SaveEEPROMToFile(filename.encode()), "SaveEEPROMToFile")

    # ========================================================================
    # Video Mode & Convenience Methods
    # ========================================================================

    def setup_single_scan(self) -> None:
        """Configure camera for single scan acquisition."""
        self.set_acquisition_mode(AcquisitionMode.SINGLE_SCAN)
        self.set_read_mode(ReadMode.IMAGE)
        self.set_trigger_mode(TriggerMode.INTERNAL)


    def close_shutter(self):
        self.set_shutter(ShutterType.TTL_HIGH, ShutterMode.CLOSE, 0, 0)


    def setup_video_mode(self, frame_transfer: bool = True, use_em: bool = True) -> None:
        """
        Configure camera for continuous video mode (run till abort).

        Parameters
        ----------
        frame_transfer : bool
            Enable frame transfer mode for fastest frame rates (default True).
        use_em : bool
            Use EM amplifier (True, default) or conventional amplifier (False).
        """
        self.set_read_mode(ReadMode.IMAGE)
        self.set_acquisition_mode(AcquisitionMode.RUN_TILL_ABORT)
        self.set_trigger_mode(TriggerMode.INTERNAL)
        self.set_frame_transfer_mode(frame_transfer)
        # Set output amplifier: 0 = EM, 1 = Conventional
        self.set_output_amplifier(0 if use_em else 1)
        # Set HS speed for selected amplifier (index 1 = 20 MHz for EM)
        self.set_hs_speed(1, 0)
        # Set EM gain mode if using EM amplifier (gain value set by UI)
        if use_em:
            self.set_em_advanced(True)
            self.set_em_gain_mode(1)  # 12-bit DAC (0-4095)
            try:
                self.set_em_clock_compensation(True)  # Enable aging compensation
            except AndorError:
                print("[CAM] EM Clock Compensation: not supported")
        # Set preamp gain to 4x (index 2)
        self.set_preamp_gain(1)
        # Set vertical shift speed (index 1 = 0.7  us)
        self.set_vs_speed(1)
        # Set full image region (required before acquisition)
        self.set_image(1, 1, 1, self._width, 1, self._height)
        # Keep shutter permanently open
        self.set_shutter(ShutterType.TTL_HIGH, ShutterMode.OPEN, 0, 0)

    def setup_kinetic_series(self, num_frames: int, cycle_time: float = 0.0) -> None:
        """
        Configure camera for kinetic series acquisition.

        Parameters
        ----------
        num_frames : int
            Number of frames to acquire.
        cycle_time : float
            Time between frames in seconds (0 = as fast as possible).
        """
        self.set_read_mode(ReadMode.IMAGE)
        self.set_acquisition_mode(AcquisitionMode.KINETICS)
        self.set_trigger_mode(TriggerMode.INTERNAL)
        self.set_number_kinetics(num_frames)
        self.set_kinetic_cycle_time(cycle_time)

    def optimize_readout_speed(self, use_em_amplifier: bool = True) -> Tuple[float, float]:
        """
        Automatically select the fastest horizontal and vertical readout speeds.

        Parameters
        ----------
        use_em_amplifier : bool
            If True, use EM amplifier (index 0). If False, use conventional (index 1).

        Returns
        -------
        hs_speed : float
            Selected horizontal speed in MHz.
        vs_speed : float
            Selected vertical speed in microseconds.
        """
        amp_idx = 0 if use_em_amplifier else 1
        self.set_output_amplifier(amp_idx)
        self.set_ad_channel(0)

        # Find fastest horizontal speed (highest MHz value)
        num_hs = self.get_number_hs_speeds(channel=0, output_amp=amp_idx)
        max_speed_val = 0.0
        max_speed_idx = 0

        for i in range(num_hs):
            speed = self.get_hs_speed(i, channel=0, output_amp=amp_idx)
            if speed > max_speed_val:
                max_speed_val = speed
                max_speed_idx = i

        self.set_hs_speed(max_speed_idx, output_amp=amp_idx)

        # Find fastest vertical speed (lowest microseconds value)
        num_vs = self.get_number_vs_speeds()
        min_vs_val = 99999.0
        min_vs_idx = 0

        for i in range(num_vs):
            speed = self.get_vs_speed(i)
            if speed < min_vs_val:
                min_vs_val = speed
                min_vs_idx = i

        self.set_vs_speed(min_vs_idx)

        return (max_speed_val, min_vs_val)

    def get_latest_frame(self, dtype: str = "uint16") -> Optional[np.ndarray]:
        """
        Get the most recent frame from the circular buffer (for video mode).

        Returns None if no new data is available.

        Parameters
        ----------
        dtype : str
            Data type: "uint16" or "int32".

        Returns
        -------
        frame : np.ndarray or None
            Image data as 2D numpy array, or None if no new data.
        """
        size = self._width * self._height

        if dtype == "uint16":
            buffer = (c_ushort * size)()
            code = _lib.GetMostRecentImage16(buffer, at_u32(size))
        else:
            buffer = (at_32 * size)()
            code = _lib.GetMostRecentImage(buffer, at_u32(size))

        if code == ErrorCode.DRV_SUCCESS:
            return np.ctypeslib.as_array(buffer).reshape(self._height, self._width).copy()
        elif code == ErrorCode.DRV_NO_NEW_DATA:
            return None
        else:
            _check_error(code, "GetMostRecentImage")
            return None

    def capture_single_frame(self, exposure: Optional[float] = None) -> np.ndarray:
        """
        Convenience method to capture a single frame.

        Parameters
        ----------
        exposure : float, optional
            Exposure time in seconds. If None, uses current setting.

        Returns
        -------
        frame : np.ndarray
            Captured image as 2D numpy array.
        """
        self.setup_single_scan()
        if exposure is not None:
            self.set_exposure_time(exposure)

        self.start_acquisition()
        self.wait_for_acquisition()
        return self.get_acquired_data(dtype="uint16")

    # ========================================================================
    # Version Information
    # ========================================================================

    def get_software_version(self) -> dict:
        """Get software version information."""
        eprom = c_uint()
        coffile = c_uint()
        vxdrev = c_uint()
        vxdver = c_uint()
        dllrev = c_uint()
        dllver = c_uint()

        _check_error(
            _lib.GetSoftwareVersion(
                byref(eprom), byref(coffile),
                byref(vxdrev), byref(vxdver),
                byref(dllrev), byref(dllver)
            ),
            "GetSoftwareVersion"
        )

        return {
            "eprom": eprom.value,
            "coffile": coffile.value,
            "driver_rev": vxdrev.value,
            "driver_ver": vxdver.value,
            "dll_rev": dllrev.value,
            "dll_ver": dllver.value,
        }

    def get_hardware_version(self) -> dict:
        """Get hardware version information."""
        pcb = c_uint()
        decode = c_uint()
        dummy1 = c_uint()
        dummy2 = c_uint()
        fw_ver = c_uint()
        fw_build = c_uint()

        _check_error(
            _lib.GetHardwareVersion(
                byref(pcb), byref(decode),
                byref(dummy1), byref(dummy2),
                byref(fw_ver), byref(fw_build)
            ),
            "GetHardwareVersion"
        )

        return {
            "pcb": pcb.value,
            "decode": decode.value,
            "firmware_version": fw_ver.value,
            "firmware_build": fw_build.value,
        }


# ============================================================================
# Convenience Functions
# ============================================================================

def list_cameras() -> List[dict]:
    """
    List all available Andor cameras.

    Returns
    -------
    cameras : list of dict
        List of camera info dictionaries with serial and model.
    """
    cameras = []
    num = get_available_cameras()

    for i in range(num):
        try:
            with AndorCamera(camera_index=i) as cam:
                cameras.append({
                    "index": i,
                    "serial": cam.serial_number,
                    "model": cam.head_model,
                    "width": cam.width,
                    "height": cam.height,
                })
        except AndorError:
            cameras.append({
                "index": i,
                "serial": None,
                "model": "Unknown (in use?)",
            })

    return cameras


def quick_capture(
    exposure: float = 1.0,
    temperature: Optional[int] = None,
    binning: int = 1,
    camera_index: int = 0
) -> np.ndarray:
    """
    Quickly capture a single image.

    Parameters
    ----------
    exposure : float
        Exposure time in seconds.
    temperature : int, optional
        Target temperature. If None, uses current temperature.
    binning : int
        Binning factor (applies to both axes).
    camera_index : int
        Camera index to use.

    Returns
    -------
    image : np.ndarray
        Captured image as 2D numpy array.
    """
    with AndorCamera(camera_index=camera_index) as cam:
        if temperature is not None:
            cam.set_temperature(temperature)
            cam.cooler_on()
            # Wait for temperature to stabilize
            import time
            while True:
                temp, status = cam.get_temperature()
                if status == ErrorCode.DRV_TEMPERATURE_STABILIZED:
                    break
                time.sleep(1)

        cam.set_acquisition_mode(AcquisitionMode.SINGLE_SCAN)
        cam.set_read_mode(ReadMode.IMAGE)
        cam.set_trigger_mode(TriggerMode.INTERNAL)
        cam.set_exposure_time(exposure)
        cam.set_image(hbin=binning, vbin=binning)

        cam.start_acquisition()
        cam.wait_for_acquisition()

        return cam.get_acquired_data(dtype="uint16")


# ============================================================================
# Simulated Camera (for testing without hardware)
# ============================================================================

class SimulatedCamera:
    """
    Simulated iXon Ultra 888 camera for testing without hardware.

    Generates realistic EMCCD images with:
    - Read noise
    - Dark current
    - Clock-induced charge (CIC)
    - EM gain with stochastic multiplication
    - Simulated star field

    Parameters
    ----------
    width : int
        Detector width (default: 1024 for iXon 888).
    height : int
        Detector height (default: 1024 for iXon 888).
    seed : int, optional
        Random seed for reproducibility.
    """

    def __init__(self, width: int = 1024, height: int = 1024, seed: Optional[int] = None):
        self._width = width
        self._height = height
        self._rng = np.random.default_rng(seed)

        # Camera state
        self._initialized = True
        self._acquiring = False
        self._cooler_on = False
        self._last_frame_time = 0.0  # For frame rate limiting
        self._frame_count = 0  # Frame counter for video mode

        # Settings with iXon Ultra 888 defaults
        self._temperature = 20.0  # Current temp (starts at ambient)
        self._target_temperature = -60  # Target temp
        self._exposure_time = 1.0  # seconds
        self._em_gain = 1  # EM gain multiplier
        self._em_gain_mode = 0  # 0=8-bit DAC
        self._em_advanced = False
        self._preamp_gain_index = 0
        self._hs_speed_index = 0
        self._vs_speed_index = 0
        self._vs_amplitude_index = 0
        self._output_amplifier = 0  # 0=EM, 1=Conventional
        self._acquisition_mode = AcquisitionMode.SINGLE_SCAN
        self._read_mode = ReadMode.IMAGE
        self._trigger_mode = TriggerMode.INTERNAL
        self._hbin = 1
        self._vbin = 1
        self._hstart = 1
        self._hend = width
        self._vstart = 1
        self._vend = height
        self._frame_transfer = False
        self._baseline_clamp = True

        # Filter settings
        self._filter_mode = SpuriousNoiseFilterMode.OFF
        self._filter_threshold = 0.0
        self._averaging_mode = DataAveragingMode.OFF
        self._averaging_frame_count = 1
        self._averaging_factor = 1

        # Noise parameters (typical iXon Ultra 888 values)
        self._read_noise_em = 50.0  # e- RMS at high speed with EM
        self._read_noise_conv = 6.0  # e- RMS conventional
        self._dark_current = 0.0001  # e-/pixel/sec at -60C (very low)
        self._cic_rate = 0.002  # CIC events per pixel per frame
        self._bias_level = 100  # ADU - dark background
        self._gain_e_per_adu = 4.5  # e-/ADU at 1x preamp
        self._saturation = 65535  # 16-bit

        # Preamp gains
        self._preamp_gains = [1.0, 2.0, 4.0]

        # HS speeds in MHz (EM amplifier)
        self._hs_speeds_em = [30.0, 20.0, 10.0, 1.0]
        # HS speeds in MHz (Conventional)
        self._hs_speeds_conv = [3.0, 1.0, 0.08]

        # VS speeds in microseconds
        self._vs_speeds = [0.3, 0.5, 0.9, 1.3, 1.8, 3.3]

        # VS amplitudes
        self._vs_amplitudes = ["Normal", "+1", "+2", "+3", "+4"]

        # Generate star field
        self._stars = self._generate_star_field()
        self._galaxy = self._generate_galaxy()

        # Frame buffer for kinetic series
        self._frame_buffer = []
        self._num_kinetics = 1
        self._kinetic_cycle_time = 0.0

    def _generate_star_field(self, num_stars: int = 12) -> List[dict]:
        """Generate random star positions for photon-counting EMCCD imaging.

        Flux values are in photons/sec. With typical 30ms exposures:
        - Bright star: 50 ph/s -> ~1.5 photons/frame -> visible speckle pattern
        - Faint star: 5 ph/s -> ~0.15 photons/frame -> occasional single photon
        """
        stars = []
        # Bright stars - will show as speckle clusters
        for _ in range(num_stars):
            stars.append({
                'x': self._rng.uniform(80, self._width - 80),
                'y': self._rng.uniform(80, self._height - 80),
                'flux': self._rng.uniform(30, 100),  # ~1-3 photons per 30ms frame
                'fwhm': self._rng.uniform(2.5, 4.0),
            })
        # Medium stars
        for _ in range(num_stars * 2):
            stars.append({
                'x': self._rng.uniform(50, self._width - 50),
                'y': self._rng.uniform(50, self._height - 50),
                'flux': self._rng.uniform(5, 30),  # ~0.15-1 photon per frame
                'fwhm': self._rng.uniform(2.0, 3.5),
            })
        # Very faint stars - occasional single photon events
        for _ in range(num_stars * 3):
            stars.append({
                'x': self._rng.uniform(30, self._width - 30),
                'y': self._rng.uniform(30, self._height - 30),
                'flux': self._rng.uniform(1, 8),  # rare single photons
                'fwhm': self._rng.uniform(1.8, 3.0),
            })
        return stars

    def _generate_galaxy(self) -> dict:
        """Generate a faint galaxy at the center - visible mainly in stacked images."""
        return {
            'x': self._width / 2,
            'y': self._height / 2,
            'flux': 450,  # total flux - 3x brighter, more visible in average mode
            'r_eff': 20.0,  # effective radius in pixels
            'ellipticity': 0.5,
            'position_angle': self._rng.uniform(0, 180),
            'sersic_n': 1.0,  # exponential disk profile
        }

    def _apply_em_gain(self, electrons: np.ndarray) -> np.ndarray:
        """
        Apply stochastic EM gain multiplication.

        EM gain follows an exponential distribution for single electrons,
        approximated here with gamma distribution for multiple electrons.
        """
        if self._em_gain <= 1 or self._output_amplifier == 1:
            return electrons

        # For each pixel with signal, apply stochastic multiplication
        # Mean = em_gain, variance = em_gain^2 for single electron
        # Use gamma distribution: shape = electrons, scale = em_gain
        mask = electrons > 0
        result = electrons.copy()

        # Gamma distribution approximates EM multiplication
        # shape parameter = number of electrons
        # scale parameter = em_gain
        if np.any(mask):
            result[mask] = self._rng.gamma(
                shape=electrons[mask],
                scale=self._em_gain
            )

        return result

    def _generate_frame(self) -> np.ndarray:
        """Generate a simulated EMCCD frame."""
        # Calculate actual image dimensions with binning
        img_width = (self._hend - self._hstart + 1) // self._hbin
        img_height = (self._vend - self._vstart + 1) // self._vbin

        # Use fast mode for short exposures (video mode)
        if self._exposure_time < 1.0:
            return self._generate_frame_fast(img_width, img_height)

        # Full simulation for longer exposures
        # Start with bias
        frame = np.full((img_height, img_width), self._bias_level, dtype=np.float64)

        # Temperature-dependent dark current
        # Doubles every 7 degrees (typical)
        temp_factor = 2 ** ((self._temperature + 60) / 7.0)
        dark_rate = self._dark_current * temp_factor
        dark_electrons = self._rng.poisson(
            dark_rate * self._exposure_time * self._hbin * self._vbin,
            size=(img_height, img_width)
        ).astype(np.float64)

        # Add stars (Gaussian PSF)
        star_electrons = np.zeros((img_height, img_width), dtype=np.float64)
        y_coords, x_coords = np.ogrid[:img_height, :img_width]

        for star in self._stars:
            # Adjust star position for ROI and binning
            star_x = (star['x'] - self._hstart + 1) / self._hbin
            star_y = (star['y'] - self._vstart + 1) / self._vbin

            # Skip if star is outside ROI
            if star_x < 0 or star_x >= img_width or star_y < 0 or star_y >= img_height:
                continue

            # Gaussian PSF
            sigma = star['fwhm'] / (2.355 * self._hbin)  # FWHM to sigma, adjust for binning
            dist_sq = (x_coords - star_x)**2 + (y_coords - star_y)**2
            psf = np.exp(-dist_sq / (2 * sigma**2))
            psf /= psf.sum()  # Normalize

            # Add star photons (Poisson)
            total_photons = star['flux'] * self._exposure_time
            star_electrons += psf * total_photons

        # Add galaxy (Sersic profile)
        gal = self._galaxy
        gal_x = (gal['x'] - self._hstart + 1) / self._hbin
        gal_y = (gal['y'] - self._vstart + 1) / self._vbin
        r_eff = gal['r_eff'] / self._hbin

        if 0 <= gal_x < img_width and 0 <= gal_y < img_height:
            radius = max(1, int(r_eff * 5))
            x_min = max(0, int(gal_x) - radius)
            x_max = min(img_width, int(gal_x) + radius + 1)
            y_min = max(0, int(gal_y) - radius)
            y_max = min(img_height, int(gal_y) + radius + 1)

            yy, xx = np.ogrid[y_min:y_max, x_min:x_max]

            pa_rad = np.deg2rad(gal['position_angle'])
            cos_pa, sin_pa = np.cos(pa_rad), np.sin(pa_rad)
            dx = xx - gal_x
            dy = yy - gal_y
            x_rot = dx * cos_pa + dy * sin_pa
            y_rot = -dx * sin_pa + dy * cos_pa

            q = 1.0 - gal['ellipticity']
            r = np.sqrt(x_rot**2 + (y_rot / q)**2)

            n = gal['sersic_n']
            b_n = 2 * n - 1.0 / 3 + 0.009876 / n
            profile = np.exp(-b_n * ((r / max(0.1, r_eff))**(1.0/n) - 1))
            profile *= gal['flux'] * self._exposure_time / max(1e-10, profile.sum())

            star_electrons[y_min:y_max, x_min:x_max] += profile

        # Apply Poisson statistics to star signal
        star_electrons = self._rng.poisson(star_electrons.clip(0))

        # Clock-induced charge (CIC) - more likely at faster VS speeds
        vs_speed = self._vs_speeds[self._vs_speed_index]
        cic_factor = 1.0 + (0.3 - vs_speed) / 0.3 if vs_speed < 0.5 else 1.0
        cic_electrons = self._rng.poisson(
            self._cic_rate * cic_factor,
            size=(img_height, img_width)
        ).astype(np.float64)

        # Total signal electrons before EM gain
        total_electrons = dark_electrons + star_electrons + cic_electrons

        # Apply EM gain (stochastic multiplication)
        if self._output_amplifier == 0:  # EM mode
            amplified = self._apply_em_gain(total_electrons)
        else:  # Conventional mode
            amplified = total_electrons

        # Read noise
        if self._output_amplifier == 0:  # EM mode
            read_noise = self._read_noise_em
        else:  # Conventional
            read_noise = self._read_noise_conv

        noise = self._rng.normal(0, read_noise, size=(img_height, img_width))

        # Convert to ADU
        preamp = self._preamp_gains[self._preamp_gain_index]
        adu = amplified / (self._gain_e_per_adu / preamp) + noise / (self._gain_e_per_adu / preamp)

        # Add to bias level
        frame += adu

        # Apply CIC filter if enabled
        if self._filter_mode == SpuriousNoiseFilterMode.MEDIAN:
            from scipy.ndimage import median_filter
            frame = median_filter(frame, size=3)
        elif self._filter_mode == SpuriousNoiseFilterMode.LEVEL_ABOVE:
            # Simple threshold filter
            from scipy.ndimage import median_filter
            median = median_filter(frame, size=3)
            mask = (frame - median) > self._filter_threshold
            frame[mask] = median[mask]

        # Clip to valid range
        frame = np.clip(frame, 0, self._saturation)

        return frame.astype(np.uint16)

    def _generate_frame_fast(self, img_width: int, img_height: int) -> np.ndarray:
        """Generate a fast simulated frame with simplified noise model for video mode."""
        # Pre-compute star image if not cached or size changed
        cache_key = (img_width, img_height, self._hbin, self._hstart, self._vstart)
        if not hasattr(self, '_star_cache') or getattr(self, '_star_cache_key', None) != cache_key:
            self._star_cache = self._compute_star_image(img_width, img_height)
            self._star_cache_key = cache_key

        # Use Cython implementation if available
        if _USE_CYTHON_SIM:
            return _sim_frame.generate_frame_fast(
                self._star_cache, img_width, img_height,
                self._exposure_time, self._em_gain,
                self._bias_level, self._cic_rate,
                self._rng
            )

        # Fallback to pure Python
        frame = np.full((img_height, img_width), self._bias_level, dtype=np.float32)
        star_signal = self._star_cache * self._exposure_time * self._em_gain
        frame += star_signal

        # Add minimal read noise for photon-starved look
        noise_level = 2.0 + np.sqrt(max(1, self._em_gain)) * 0.5
        frame += self._rng.normal(0, noise_level, size=(img_height, img_width)).astype(np.float32)

        # Add sparse CIC/cosmic ray events
        num_cic = int(self._cic_rate * img_width * img_height * 0.1)
        if num_cic > 0:
            cic_x = self._rng.integers(0, img_width, num_cic)
            cic_y = self._rng.integers(0, img_height, num_cic)
            cic_vals = self._rng.exponential(self._em_gain * 10, num_cic).astype(np.float32)
            frame[cic_y, cic_x] += cic_vals

        # Clip and convert
        return np.clip(frame, 0, self._saturation).astype(np.uint16)

    def _compute_star_image(self, img_width: int, img_height: int) -> np.ndarray:
        """Pre-compute the star pattern for fast mode."""
        # Use Cython implementation if available
        if _USE_CYTHON_SIM:
            return _sim_frame.compute_star_image(
                self._stars, self._galaxy,
                img_width, img_height,
                self._hbin, self._hstart, self._vstart
            )

        # Fallback to pure Python
        star_image = np.zeros((img_height, img_width), dtype=np.float32)

        for star in self._stars:
            star_x = (star['x'] - self._hstart + 1) / self._hbin
            star_y = (star['y'] - self._vstart + 1) / self._hbin

            if star_x < 0 or star_x >= img_width or star_y < 0 or star_y >= img_height:
                continue

            sigma = star['fwhm'] / (2.355 * self._hbin)

            # Only compute within a reasonable radius
            radius = max(1, int(sigma * 4))
            x_min = max(0, int(star_x) - radius)
            x_max = min(img_width, int(star_x) + radius + 1)
            y_min = max(0, int(star_y) - radius)
            y_max = min(img_height, int(star_y) + radius + 1)

            yy, xx = np.ogrid[y_min:y_max, x_min:x_max]
            local_dist_sq = (xx - star_x)**2 + (yy - star_y)**2
            psf = np.exp(-local_dist_sq / (2 * sigma**2))
            psf *= star['flux'] / max(1e-10, psf.sum())

            star_image[y_min:y_max, x_min:x_max] += psf.astype(np.float32)

        # Add galaxy (Sersic profile)
        gal = self._galaxy
        gal_x = (gal['x'] - self._hstart + 1) / self._hbin
        gal_y = (gal['y'] - self._vstart + 1) / self._hbin
        r_eff = gal['r_eff'] / self._hbin

        if 0 <= gal_x < img_width and 0 <= gal_y < img_height:
            # Compute galaxy within larger radius
            radius = max(1, int(r_eff * 5))
            x_min = max(0, int(gal_x) - radius)
            x_max = min(img_width, int(gal_x) + radius + 1)
            y_min = max(0, int(gal_y) - radius)
            y_max = min(img_height, int(gal_y) + radius + 1)

            yy, xx = np.ogrid[y_min:y_max, x_min:x_max]

            # Rotation for ellipticity
            pa_rad = np.deg2rad(gal['position_angle'])
            cos_pa, sin_pa = np.cos(pa_rad), np.sin(pa_rad)
            dx = xx - gal_x
            dy = yy - gal_y
            x_rot = dx * cos_pa + dy * sin_pa
            y_rot = -dx * sin_pa + dy * cos_pa

            # Elliptical radius
            q = 1.0 - gal['ellipticity']  # axis ratio
            r = np.sqrt(x_rot**2 + (y_rot / q)**2)

            # Sersic profile: I(r) = I_e * exp(-b_n * ((r/r_eff)^(1/n) - 1))
            n = gal['sersic_n']
            b_n = 2 * n - 1.0 / 3 + 0.009876 / n  # approximation
            profile = np.exp(-b_n * ((r / max(0.1, r_eff))**(1.0/n) - 1))
            profile *= gal['flux'] / max(1e-10, profile.sum())

            star_image[y_min:y_max, x_min:x_max] += profile.astype(np.float32)

        return star_image

    # ========================================================================
    # Properties
    # ========================================================================

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def serial_number(self) -> int:
        return 99999  # Simulated

    @property
    def head_model(self) -> str:
        return "DU888_BV (Simulated)"

    @property
    def pixel_size(self) -> Tuple[float, float]:
        return (13.0, 13.0)  # iXon Ultra 888 has 13µm pixels

    @property
    def status(self) -> CameraStatus:
        return CameraStatus.ACQUIRING if self._acquiring else CameraStatus.IDLE

    @property
    def is_acquiring(self) -> bool:
        return self._acquiring

    @property
    def em_gain(self) -> int:
        return self._em_gain

    @em_gain.setter
    def em_gain(self, value: int):
        self.set_emccd_gain(value)

    # ========================================================================
    # Context Manager
    # ========================================================================

    def __enter__(self) -> "SimulatedCamera":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def close(self) -> None:
        self._initialized = False

    # ========================================================================
    # Camera Info
    # ========================================================================

    def get_capabilities(self) -> AndorCapabilities:
        caps = AndorCapabilities()
        caps.ulSize = ctypes.sizeof(AndorCapabilities)
        caps.ulCameraType = CameraType.IXON
        caps.ulAcqModes = 0x7F  # All modes
        caps.ulReadModes = 0x1F  # All read modes
        caps.ulTriggerModes = 0xFF  # All trigger modes
        caps.ulFeatures = (
            FeatureCapability.SHUTTER |
            FeatureCapability.SHUTTEREX |
            FeatureCapability.FANCONTROL |
            FeatureCapability.TEMPERATUREDURINGACQUISITION
        )
        return caps

    def get_camera_type(self) -> CameraType:
        return CameraType.IXON

    # ========================================================================
    # Temperature
    # ========================================================================

    def get_temperature(self) -> Tuple[float, int]:
        # Simulate cooling
        if self._cooler_on:
            diff = self._temperature - self._target_temperature
            if abs(diff) > 0.5:
                self._temperature -= np.sign(diff) * 0.5  # Cool/warm 0.5°C per call
                return (self._temperature, ErrorCode.DRV_TEMPERATURE_NOT_REACHED)
            else:
                return (self._temperature, ErrorCode.DRV_TEMPERATURE_STABILIZED)
        else:
            return (self._temperature, ErrorCode.DRV_TEMPERATURE_OFF)

    def get_temperature_range(self) -> Tuple[int, int]:
        return (-100, 20)

    def set_temperature(self, temperature: int) -> None:
        self._target_temperature = temperature

    def cooler_on(self) -> None:
        self._cooler_on = True

    def cooler_off(self) -> None:
        self._cooler_on = False

    def is_cooler_on(self) -> bool:
        return self._cooler_on

    def get_tec_status(self) -> bool:
        """Check TEC status (simulated - always returns False/OK)."""
        return False

    def set_cooler_mode(self, maintain: bool) -> None:
        """Set whether cooler stays on after ShutDown (simulated)."""
        self._maintain_cooler = maintain

    def set_fan_mode(self, mode: FanMode) -> None:
        pass  # Simulated

    # ========================================================================
    # Acquisition Setup
    # ========================================================================

    def set_acquisition_mode(self, mode: AcquisitionMode) -> None:
        self._acquisition_mode = mode

    def set_read_mode(self, mode: ReadMode) -> None:
        self._read_mode = mode

    def set_trigger_mode(self, mode: TriggerMode) -> None:
        self._trigger_mode = mode

    def set_exposure_time(self, seconds: float) -> None:
        self._exposure_time = seconds

    def set_accumulation_cycle_time(self, seconds: float) -> None:
        pass

    def set_kinetic_cycle_time(self, seconds: float) -> None:
        self._kinetic_cycle_time = seconds

    def set_number_accumulations(self, number: int) -> None:
        pass

    def set_number_kinetics(self, number: int) -> None:
        self._num_kinetics = number

    def get_acquisition_timings(self) -> Tuple[float, float, float]:
        return (self._exposure_time, self._exposure_time, self._exposure_time)

    # ========================================================================
    # Image Settings
    # ========================================================================

    def set_image(
        self,
        hbin: int = 1,
        vbin: int = 1,
        hstart: int = 1,
        hend: Optional[int] = None,
        vstart: int = 1,
        vend: Optional[int] = None
    ) -> None:
        self._hbin = hbin
        self._vbin = vbin
        self._hstart = hstart
        self._hend = hend if hend else self._width
        self._vstart = vstart
        self._vend = vend if vend else self._height

    def set_full_image(self, hbin: int = 1, vbin: int = 1) -> None:
        self.set_image(hbin, vbin, 1, self._width, 1, self._height)

    def set_image_flip(self, horizontal: bool = False, vertical: bool = False) -> None:
        pass

    def get_image_flip(self) -> Tuple[bool, bool]:
        return (False, False)

    def set_image_rotate(self, rotation: int) -> None:
        pass

    def set_shutter(self, shutter_type: ShutterType = ShutterType.TTL_HIGH,
                    mode: ShutterMode = ShutterMode.AUTO,
                    closing_time: int = 0, opening_time: int = 0) -> None:
        pass

    # ========================================================================
    # Acquisition Control
    # ========================================================================

    def prepare_acquisition(self) -> None:
        pass

    def start_acquisition(self) -> None:
        self._acquiring = True
        self._frame_buffer = []
        self._frame_count = 0

        if self._acquisition_mode == AcquisitionMode.KINETICS:
            for _ in range(self._num_kinetics):
                self._frame_buffer.append(self._generate_frame())
            self._frame_count = self._num_kinetics
        elif self._acquisition_mode == AcquisitionMode.RUN_TILL_ABORT:
            # Generate first frame for video mode
            self._frame_buffer.append(self._generate_frame())
            self._frame_count = 1  # Signal first frame is ready
        else:
            self._frame_buffer.append(self._generate_frame())
            self._frame_count = 1

    def abort_acquisition(self) -> None:
        self._acquiring = False

    def wait_for_acquisition(self, timeout_ms: Optional[int] = None) -> None:
        import time
        time.sleep(self._exposure_time * 0.1)  # Simulate some wait
        self._acquiring = False

    def cancel_wait(self) -> None:
        pass

    def send_software_trigger(self) -> None:
        pass

    def get_acquisition_progress(self) -> Tuple[int, int]:
        return (len(self._frame_buffer), len(self._frame_buffer))

    # ========================================================================
    # Data Retrieval
    # ========================================================================

    def get_acquired_data(self, dtype: str = "uint16") -> np.ndarray:
        if self._frame_buffer:
            return self._frame_buffer[0]
        return self._generate_frame()

    def get_most_recent_image(self, dtype: str = "uint16") -> np.ndarray:
        return self.get_acquired_data(dtype)

    def get_oldest_image(self, dtype: str = "uint16") -> np.ndarray:
        return self.get_acquired_data(dtype)

    def get_latest_frame(self, dtype: str = "uint16") -> Optional[np.ndarray]:
        if self._acquiring:
            # Rate limit to 25 FPS (40ms between frames)
            now = time.perf_counter()
            if now - self._last_frame_time < 0.040:
                return None  # Too soon, no new frame
            self._last_frame_time = now
            # Generate new frame for video mode
            frame = self._generate_frame()
            self._frame_buffer = [frame]
            self._frame_count += 1
            return frame
        return None

    def get_number_new_images(self) -> Tuple[int, int]:
        return (1, self._frame_count)

    def get_images(self, first: int, last: int, dtype: str = "uint16") -> Tuple[np.ndarray, int, int]:
        if self._frame_buffer:
            frames = np.stack(self._frame_buffer[first-1:last])
            return (frames, first, last)
        return (np.stack([self._generate_frame()]), 1, 1)

    # ========================================================================
    # Speed Settings
    # ========================================================================

    def get_number_hs_speeds(self, channel: int = 0, output_amp: int = 0) -> int:
        if output_amp == 0:
            return len(self._hs_speeds_em)
        return len(self._hs_speeds_conv)

    def get_hs_speed(self, index: int, channel: int = 0, output_amp: int = 0) -> float:
        if output_amp == 0:
            return self._hs_speeds_em[index]
        return self._hs_speeds_conv[index]

    def set_hs_speed(self, index: int, output_amp: int = 0) -> None:
        self._hs_speed_index = index

    def get_number_vs_speeds(self) -> int:
        return len(self._vs_speeds)

    def get_vs_speed(self, index: int) -> float:
        return self._vs_speeds[index]

    def set_vs_speed(self, index: int) -> None:
        self._vs_speed_index = index

    def get_fastest_recommended_vs_speed(self) -> Tuple[int, float]:
        return (0, self._vs_speeds[0])

    # ========================================================================
    # VS Amplitude
    # ========================================================================

    def get_number_vs_amplitudes(self) -> int:
        return len(self._vs_amplitudes)

    def get_vs_amplitude_value(self, index: int) -> int:
        return index

    def get_vs_amplitude_string(self, index: int) -> str:
        return self._vs_amplitudes[index]

    def set_vs_amplitude(self, index: int) -> None:
        self._vs_amplitude_index = index

    def get_vs_amplitudes(self) -> List[str]:
        return self._vs_amplitudes.copy()

    # ========================================================================
    # Gain Settings
    # ========================================================================

    def get_number_preamp_gains(self) -> int:
        return len(self._preamp_gains)

    def get_preamp_gain(self, index: int) -> float:
        return self._preamp_gains[index]

    def set_preamp_gain(self, index: int) -> None:
        self._preamp_gain_index = index

    def get_emccd_gain(self) -> int:
        return self._em_gain

    def set_emccd_gain(self, gain: int) -> None:
        max_gain = 1000 if self._em_advanced else 300
        self._em_gain = min(max(1, gain), max_gain)

    def get_em_gain_range(self) -> Tuple[int, int]:
        return (1, 1000 if self._em_advanced else 300)

    def set_em_gain_mode(self, mode: int) -> None:
        self._em_gain_mode = mode

    def set_em_advanced(self, enabled: bool) -> None:
        self._em_advanced = enabled

    def configure_em_gain(self, gain: int, mode: int = 0, advanced: bool = False) -> None:
        self._em_advanced = advanced
        self._em_gain_mode = mode
        self.set_emccd_gain(gain)

    # ========================================================================
    # AD Channel / Amplifier
    # ========================================================================

    def get_number_ad_channels(self) -> int:
        return 1

    def get_bit_depth(self, channel: int = 0) -> int:
        return 16

    def set_ad_channel(self, channel: int) -> None:
        pass

    def get_number_amplifiers(self) -> int:
        return 2

    def get_amplifier_description(self, index: int) -> str:
        return "Electron Multiplying" if index == 0 else "Conventional"

    def set_output_amplifier(self, index: int) -> None:
        self._output_amplifier = index

    # ========================================================================
    # Frame Transfer / Baseline
    # ========================================================================

    def set_frame_transfer_mode(self, enabled: bool) -> None:
        self._frame_transfer = enabled

    def set_crop_mode(self, active: bool, crop_height: int) -> None:
        pass  # Simulated

    def set_isolated_crop_mode(
        self, active: bool, crop_height: int, crop_width: int,
        vbin: int = 1, hbin: int = 1
    ) -> None:
        pass  # Simulated

    def set_baseline_clamp(self, enabled: bool) -> None:
        self._baseline_clamp = enabled

    def get_baseline_clamp(self) -> bool:
        return self._baseline_clamp

    # ========================================================================
    # CIC / Spurious Noise Filter
    # ========================================================================

    def get_filter_mode(self) -> SpuriousNoiseFilterMode:
        return self._filter_mode

    def set_filter_mode(self, mode: Union[SpuriousNoiseFilterMode, int]) -> None:
        self._filter_mode = SpuriousNoiseFilterMode(int(mode))

    def get_filter_threshold(self) -> float:
        return self._filter_threshold

    def set_filter_threshold(self, threshold: float) -> None:
        self._filter_threshold = threshold

    def get_data_averaging_mode(self) -> DataAveragingMode:
        return self._averaging_mode

    def set_data_averaging_mode(self, mode: Union[DataAveragingMode, int]) -> None:
        self._averaging_mode = DataAveragingMode(int(mode))

    def get_averaging_frame_count(self) -> int:
        return self._averaging_frame_count

    def set_averaging_frame_count(self, count: int) -> None:
        self._averaging_frame_count = count

    def get_averaging_factor(self) -> int:
        return self._averaging_factor

    def set_averaging_factor(self, factor: int) -> None:
        self._averaging_factor = factor

    def configure_cic_filter(self, mode: Union[SpuriousNoiseFilterMode, int] = SpuriousNoiseFilterMode.OFF,
                             threshold: float = 0.0) -> None:
        self.set_filter_mode(mode)
        self._filter_threshold = threshold

    def get_filter_settings(self) -> dict:
        return {
            "filter_mode": self.get_filter_mode().name,
            "filter_threshold": self.get_filter_threshold(),
            "averaging_mode": self.get_data_averaging_mode().name,
            "averaging_frame_count": self.get_averaging_frame_count(),
            "averaging_factor": self.get_averaging_factor(),
        }

    # ========================================================================
    # File Saving (just save numpy array)
    # ========================================================================

    def save_as_fits(self, filename: str, data_type: int = 0) -> None:
        try:
            from astropy.io import fits
            if self._frame_buffer:
                fits.writeto(filename, self._frame_buffer[0], overwrite=True)
        except ImportError:
            np.save(filename.replace('.fits', '.npy'), self._frame_buffer[0] if self._frame_buffer else self._generate_frame())

    def save_as_tiff(self, filename: str, palette: str = "", position: int = 1, data_type: int = 0) -> None:
        try:
            from PIL import Image
            if self._frame_buffer:
                Image.fromarray(self._frame_buffer[0]).save(filename)
        except ImportError:
            pass

    def save_as_sif(self, filename: str) -> None:
        pass  # Not implemented for simulation

    def save_as_raw(self, filename: str, data_type: int = 0) -> None:
        if self._frame_buffer:
            self._frame_buffer[0].tofile(filename)

    # ========================================================================
    # Version Info
    # ========================================================================

    def get_software_version(self) -> dict:
        return {"dll_ver": 291, "dll_rev": 30001, "driver_ver": 0, "driver_rev": 0, "eprom": 0, "coffile": 0}

    def get_hardware_version(self) -> dict:
        return {"pcb": 0, "decode": 0, "firmware_version": 0, "firmware_build": 0}

    # ========================================================================
    # Convenience Methods
    # ========================================================================

    def setup_single_scan(self) -> None:
        self.set_acquisition_mode(AcquisitionMode.SINGLE_SCAN)
        self.set_read_mode(ReadMode.IMAGE)
        self.set_trigger_mode(TriggerMode.INTERNAL)

    def setup_video_mode(self, frame_transfer: bool = True, use_em: bool = True) -> None:
        """Configure camera for continuous video mode."""
        self.set_read_mode(ReadMode.IMAGE)
        self.set_acquisition_mode(AcquisitionMode.RUN_TILL_ABORT)
        self.set_trigger_mode(TriggerMode.INTERNAL)
        self._frame_transfer = frame_transfer
        self._output_amplifier = 0 if use_em else 1

    def close_shutter(self) -> None:
        """Close the shutter (simulated - no-op)."""
        pass

    def optimize_readout_speed(self, use_em_amplifier: bool = True) -> Tuple[float, float]:
        self._output_amplifier = 0 if use_em_amplifier else 1
        self._hs_speed_index = 0  # Fastest
        self._vs_speed_index = 0  # Fastest
        if use_em_amplifier:
            return (self._hs_speeds_em[0], self._vs_speeds[0])
        return (self._hs_speeds_conv[0], self._vs_speeds[0])

    def capture_single_frame(self, exposure: Optional[float] = None) -> np.ndarray:
        self.setup_single_scan()
        if exposure is not None:
            self.set_exposure_time(exposure)
        self.start_acquisition()
        self.wait_for_acquisition()
        return self.get_acquired_data()


def get_camera(simulate: bool = False, **kwargs) -> Union[AndorCamera, SimulatedCamera]:
    """
    Get camera instance, with automatic fallback to simulation.

    Parameters
    ----------
    simulate : bool
        If True, always return simulated camera.
    **kwargs
        Additional arguments passed to camera constructor.

    Returns
    -------
    camera : AndorCamera or SimulatedCamera
        Camera instance.
    """
    if simulate:
        return SimulatedCamera(**kwargs)

    try:
        num_cameras = get_available_cameras()
        print(num_cameras)

        if num_cameras > 0:
            return AndorCamera(**kwargs)
    except (AndorError, ImportError):
        pass

    print("No camera found, using simulated iXon Ultra 888")
    return SimulatedCamera(**kwargs)


if __name__ == "__main__":
    import time

    print("Andor SDK Python Bindings - iXon Ultra 888")
    print("=" * 50)

    # Use get_camera() for automatic fallback to simulation
    with get_camera() as cam:
        print(f"\nCamera: {cam.head_model}")
        print(f"Serial: {cam.serial_number}")
        print(f"Detector: {cam.width} x {cam.height}")
        print(f"Pixel size: {cam.pixel_size[0]:.1f} x {cam.pixel_size[1]:.1f} µm")

        # Get capabilities
        cam_type = cam.get_camera_type()
        print(f"Camera type: {cam_type.name}")

        # Temperature info
        temp_range = cam.get_temperature_range()
        print(f"\n--- Temperature ---")
        print(f"Range: {temp_range[0]}°C to {temp_range[1]}°C")
        temp, status = cam.get_temperature()
        print(f"Current: {temp:.1f}°C (status: {status})")

        # Horizontal speed info
        print(f"\n--- Horizontal Shift Speeds ---")
        for amp in range(cam.get_number_amplifiers()):
            amp_name = cam.get_amplifier_description(amp)
            num_hs = cam.get_number_hs_speeds(channel=0, output_amp=amp)
            print(f"  {amp_name}:")
            for i in range(num_hs):
                speed = cam.get_hs_speed(i, channel=0, output_amp=amp)
                print(f"    [{i}] {speed:.2f} MHz")

        # Vertical speed info
        print(f"\n--- Vertical Shift Speeds ---")
        num_vs = cam.get_number_vs_speeds()
        for i in range(num_vs):
            print(f"  [{i}] {cam.get_vs_speed(i):.2f} µs/row")

        # VS Amplitude info
        print(f"\n--- Vertical Clock Voltage ---")
        vs_amps = cam.get_vs_amplitudes()
        for i, amp in enumerate(vs_amps):
            print(f"  [{i}] {amp}")

        # EM Gain info (iXon specific)
        print(f"\n--- EM Gain ---")
        em_range = cam.get_em_gain_range()
        print(f"Range: {em_range[0]} - {em_range[1]}")
        current_gain = cam.get_emccd_gain()
        print(f"Current: {current_gain}")

        # Preamp gains
        print(f"\n--- Preamp Gains ---")
        num_pa = cam.get_number_preamp_gains()
        for i in range(num_pa):
            print(f"  [{i}] {cam.get_preamp_gain(i):.1f}x")

        # Filter settings
        print(f"\n--- CIC Filter ---")
        print(cam.get_filter_settings())

        print("\n" + "=" * 50)
        print("Taking a test image...")

        # Configure and capture
        cam.set_exposure_time(0.5)
        cam.set_emccd_gain(100)
        cam.optimize_readout_speed(use_em_amplifier=True)

        frame = cam.capture_single_frame()
        print(f"Frame shape: {frame.shape}")
        print(f"Frame dtype: {frame.dtype}")
        print(f"Min: {frame.min()}, Max: {frame.max()}, Mean: {frame.mean():.1f}")

        # Try to display if matplotlib available
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 10))
            plt.imshow(frame, cmap='gray', vmin=np.percentile(frame, 1), vmax=np.percentile(frame, 99))
            plt.colorbar(label='ADU')
            plt.title(f"{cam.head_model} - {cam._exposure_time if hasattr(cam, '_exposure_time') else 0.5}s exposure, EM gain {cam.get_emccd_gain()}")
            plt.savefig('/tmp/andor_test_frame.png', dpi=150)
            print("Saved test frame to /tmp/andor_test_frame.png")
            plt.close()
        except ImportError:
            print("(matplotlib not available for display)")

        print("\nDone!")
