import sys
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QGroupBox, QLineEdit
)
from PyQt5.QtCore import Qt

# Attempt to import the focuser module, but handle failure gracefully.
try:
    from fli_focuser import focuser
    FOCUSER_AVAILABLE = True
except ImportError:
    FOCUSER_AVAILABLE = False
except Exception as e:
    # Catch other potential errors during import, like driver issues
    print(f"Could not import fli_focuser, running in simulation mode: {e}")
    FOCUSER_AVAILABLE = False


class FocuserControlWidget(QGroupBox):
    """
    A widget for controlling the telescope focuser with buttons for relative moves.
    """
    def __init__(self, parent=None):
        super().__init__("Focuser Control", parent)
        self.focuser = None
        self.is_simulated = True

        if FOCUSER_AVAILABLE:
            try:
                self.focuser = focuser()
                self.is_simulated = False
                print("Focuser connected.")
            except Exception as e:
                print(f"Failed to initialize focuser: {e}. Running in simulation mode.")
                self.is_simulated = True
        else:
            print("Focuser module not found. Running in simulation mode.")

        self._setup_ui()
        self.update_position()

    def _setup_ui(self):
        """Creates the UI elements for the widget."""
        main_layout = QVBoxLayout()

        # --- Position Display ---
        pos_layout = QHBoxLayout()
        pos_label = QLabel("Position:")
        self.pos_display = QLineEdit("N/A")
        self.pos_display.setReadOnly(True)
        self.pos_display.setAlignment(Qt.AlignCenter)
        self.pos_display.setStyleSheet("background-color: #17202A; font-weight: bold;")
        pos_layout.addWidget(pos_label)
        pos_layout.addWidget(self.pos_display)
        main_layout.addLayout(pos_layout)

        # --- Control Buttons ---
        btn_layout = QHBoxLayout()
        
        btn_large_in = QPushButton("<< (-100)")
        btn_small_in = QPushButton("< (-10)")
        btn_small_out = QPushButton("> (+10)")
        btn_large_out = QPushButton(">> (+100)")

        btn_large_in.clicked.connect(lambda: self.move_focus(-100))
        btn_small_in.clicked.connect(lambda: self.move_focus(-10))
        btn_small_out.clicked.connect(lambda: self.move_focus(10))
        btn_large_out.clicked.connect(lambda: self.move_focus(100))
        
        btn_layout.addWidget(btn_large_in)
        btn_layout.addWidget(btn_small_in)
        btn_layout.addWidget(btn_small_out)
        btn_layout.addWidget(btn_large_out)
        main_layout.addLayout(btn_layout)

        self.setLayout(main_layout)

    def move_focus(self, steps):
        """
        Commands the focuser to move by a relative number of steps.
        """
        if self.is_simulated:
            print(f"SIM: Moving focuser by {steps} steps.")
            # Simulate position change
            try:
                current_pos = int(self.pos_display.text())
                self.pos_display.setText(str(current_pos + steps))
            except (ValueError, TypeError):
                self.pos_display.setText("10000") # Default sim value
        else:
            try:
                print(f"REAL: Moving focuser by {steps} steps.")
                self.focuser.move_focus(steps)
                self.update_position()
            except Exception as e:
                print(f"Error moving focuser: {e}")

    def update_position(self):
        """
        Fetches the current position from focus_pos.txt and updates the display.
        """
        if self.is_simulated:
            # In sim mode, the position is just what's in the box.
            # If it's not a number, set a default.
            if not self.pos_display.text().isdigit():
                 self.pos_display.setText("10000")
        else:
            try:
                # Read position from the text file
                with open('focus_pos.txt', 'r') as f:
                    pos = f.read().strip()
                self.pos_display.setText(str(pos))
            except FileNotFoundError:
                print("Error: focus_pos.txt not found.")
                self.pos_display.setText("No File")
            except Exception as e:
                print(f"Error reading focuser position from file: {e}")
                self.pos_display.setText("Error")

if __name__ == '__main__':
    # This allows for testing the widget independently.
    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv)
    window = QWidget()
    layout = QVBoxLayout()
    focuser_widget = FocuserControlWidget()
    layout.addWidget(focuser_widget)
    window.setLayout(layout)
    window.setStyleSheet("""
        QWidget { background-color: #212F3D; color: #ECF0F1; }
        QPushButton { background-color: #3498DB; border: none; padding: 10px; border-radius: 5px; }
        QGroupBox { font-weight: bold; }
        QLineEdit { border-radius: 3px; }
    """)
    window.show()
    sys.exit(app.exec_())
