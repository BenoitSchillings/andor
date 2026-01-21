"""
Session management for EMCCD data acquisition.

Handles:
- Directory structure (date/target/)
- Session logging (session.json)
- Disk space monitoring
"""

import os
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any


class SessionManager:
    """Manages acquisition sessions with automatic directory structure and logging."""

    # Disk space thresholds in GB
    WARNING_THRESHOLD_GB = 50
    CRITICAL_THRESHOLD_GB = 10

    def __init__(self, base_path: str = "."):
        """
        Initialize session manager.

        Parameters
        ----------
        base_path : str
            Base directory for all data (default: current directory)
        """
        self.base_path = Path(base_path).resolve()
        self.session_dir: Optional[Path] = None
        self.session_data: Dict[str, Any] = {}
        self.active = False

    def start_session(self, target: str, filter_name: Optional[str] = None,
                      camera_settings: Optional[Dict] = None) -> Path:
        """
        Start a new acquisition session.

        Creates directory structure: base_path/YYYY-MM-DD/target/

        Parameters
        ----------
        target : str
            Target name (e.g., "M42", "NGC7000")
        filter_name : str, optional
            Filter being used (e.g., "Ha", "OIII", "L")
        camera_settings : dict, optional
            Camera settings to log (exposure, gain, temp)

        Returns
        -------
        Path
            Path to session directory
        """
        if not target:
            target = "noname_target"

        # Sanitize target name for filesystem
        target_safe = self._sanitize_name(target)

        # Create date-based directory structure
        date_str = datetime.now().strftime("%Y-%m-%d")
        self.session_dir = self.base_path / date_str / target_safe
        self.session_dir.mkdir(parents=True, exist_ok=True)

        # Initialize session data
        self.session_data = {
            "session_id": f"{date_str}_{target_safe}",
            "target": target,
            "filter": filter_name,
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "camera": camera_settings or {},
            "captures": [],
            "total_frames": 0,
            "total_size_mb": 0,
            "notes": ""
        }

        self.active = True
        self._save_session()

        return self.session_dir

    def get_capture_path(self, prefix: str = "capture", extension: str = ".ser") -> Path:
        """
        Get path for next capture file.

        Parameters
        ----------
        prefix : str
            Filename prefix (default: "capture")
        extension : str
            File extension (default: ".ser")

        Returns
        -------
        Path
            Full path for the capture file
        """
        if self.session_dir is None:
            raise RuntimeError("No active session. Call start_session first.")

        # Build filename with filter and sequence number
        filter_part = f"_{self.session_data['filter']}" if self.session_data.get('filter') else ""

        # Find next sequence number
        existing = list(self.session_dir.glob(f"{prefix}{filter_part}_*{extension}"))
        seq = len(existing) + 1

        filename = f"{prefix}{filter_part}_{seq:03d}{extension}"
        return self.session_dir / filename

    def log_capture(self, filepath: Path, frames: int, size_bytes: int,
                    metadata: Optional[Dict] = None) -> None:
        """
        Log a completed capture.

        Parameters
        ----------
        filepath : Path
            Path to the captured file
        frames : int
            Number of frames captured
        size_bytes : int
            File size in bytes
        metadata : dict, optional
            Additional metadata to log
        """
        if not self.active:
            return

        capture_info = {
            "file": filepath.name,
            "frames": frames,
            "size_mb": round(size_bytes / (1024 * 1024), 2),
            "timestamp": datetime.now().isoformat()
        }
        if metadata:
            capture_info["metadata"] = metadata

        self.session_data["captures"].append(capture_info)
        self.session_data["total_frames"] += frames
        self.session_data["total_size_mb"] += capture_info["size_mb"]

        self._save_session()

    def end_session(self, notes: Optional[str] = None) -> Dict[str, Any]:
        """
        End the current session.

        Parameters
        ----------
        notes : str, optional
            Session notes to add

        Returns
        -------
        dict
            Final session data
        """
        if not self.active:
            return self.session_data

        self.session_data["end_time"] = datetime.now().isoformat()
        if notes:
            self.session_data["notes"] = notes

        self._save_session()
        self.active = False

        return self.session_data

    def add_notes(self, notes: str) -> None:
        """Add notes to current session."""
        if self.active:
            self.session_data["notes"] = notes
            self._save_session()

    def check_disk_space(self) -> Tuple[float, str]:
        """
        Check available disk space.

        Returns
        -------
        tuple
            (available_gb, status) where status is "ok", "warning", or "critical"
        """
        path = self.session_dir if self.session_dir else self.base_path

        try:
            usage = shutil.disk_usage(path)
            available_gb = usage.free / (1024 ** 3)

            if available_gb < self.CRITICAL_THRESHOLD_GB:
                status = "critical"
            elif available_gb < self.WARNING_THRESHOLD_GB:
                status = "warning"
            else:
                status = "ok"

            return (available_gb, status)
        except Exception:
            return (0.0, "unknown")

    def estimate_capture_size(self, frames: int, width: int = 1024,
                              height: int = 1024, bytes_per_pixel: int = 2) -> int:
        """
        Estimate capture file size in bytes.

        Parameters
        ----------
        frames : int
            Number of frames
        width : int
            Frame width (default: 1024)
        height : int
            Frame height (default: 1024)
        bytes_per_pixel : int
            Bytes per pixel (default: 2 for uint16)

        Returns
        -------
        int
            Estimated size in bytes
        """
        header_size = 178  # SER header
        frame_size = width * height * bytes_per_pixel
        return header_size + (frames * frame_size)

    def can_capture(self, frames: int, width: int = 1024, height: int = 1024) -> Tuple[bool, str]:
        """
        Check if there's enough space for a capture.

        Parameters
        ----------
        frames : int
            Number of frames to capture
        width : int
            Frame width
        height : int
            Frame height

        Returns
        -------
        tuple
            (can_proceed, message)
        """
        available_gb, status = self.check_disk_space()
        estimated_bytes = self.estimate_capture_size(frames, width, height)
        estimated_gb = estimated_bytes / (1024 ** 3)

        if status == "critical":
            return (False, f"Critical: only {available_gb:.1f} GB available")

        if estimated_gb > available_gb - self.CRITICAL_THRESHOLD_GB:
            return (False, f"Not enough space: need {estimated_gb:.1f} GB, have {available_gb:.1f} GB")

        if status == "warning":
            return (True, f"Warning: only {available_gb:.1f} GB available")

        return (True, f"OK: {available_gb:.1f} GB available")

    def _sanitize_name(self, name: str) -> str:
        """Sanitize a name for use as directory/filename."""
        # Replace problematic characters
        for char in ['/', '\\', ':', '*', '?', '"', '<', '>', '|', ' ']:
            name = name.replace(char, '_')
        return name

    def _save_session(self) -> None:
        """Save session data to JSON file."""
        if self.session_dir is None:
            return

        session_file = self.session_dir / "session.json"
        with open(session_file, 'w') as f:
            json.dump(self.session_data, f, indent=2)

    @property
    def session_path(self) -> Optional[Path]:
        """Get current session directory path."""
        return self.session_dir

    def get_session_summary(self) -> str:
        """Get a human-readable session summary."""
        if not self.session_data:
            return "No active session"

        lines = [
            f"Target: {self.session_data.get('target', 'unknown')}",
            f"Filter: {self.session_data.get('filter', 'none')}",
            f"Captures: {len(self.session_data.get('captures', []))}",
            f"Total frames: {self.session_data.get('total_frames', 0)}",
            f"Total size: {self.session_data.get('total_size_mb', 0):.1f} MB"
        ]
        return "\n".join(lines)


def get_disk_space(path: str = ".") -> Tuple[float, float, float]:
    """
    Get disk space information.

    Returns
    -------
    tuple
        (total_gb, used_gb, free_gb)
    """
    usage = shutil.disk_usage(path)
    return (
        usage.total / (1024 ** 3),
        usage.used / (1024 ** 3),
        usage.free / (1024 ** 3)
    )
