"""
Telescope control module for mount guiding via TheSkyX.
"""

import logging
from socket import socket, AF_INET, SOCK_STREAM, SHUT_RDWR, error

logger = logging.getLogger(__name__)


class ScopeConnectionError(Exception):
    """Exception for failures to connect to TheSkyX."""
    pass


class Scope:
    """Telescope control via TheSkyX."""

    def __init__(self, host="localhost", port=3040):
        self.host = host
        self.port = port
        self.connected = False

    def init(self):
        """Initialize connection to TheSkyX and connect to telescope."""
        try:
            # Test connection by checking if telescope is connected
            command = """
                var Out;
                sky6RASCOMTele.Connect();
                Out = sky6RASCOMTele.IsConnected;"""
            output = self._send(command).splitlines()
            if int(output[0]) == 1:
                self.connected = True
                logger.info("Scope: Connected to TheSkyX telescope")
                return True
            else:
                logger.warning("Scope: Telescope not connected in TheSkyX")
                return False
        except Exception as e:
            logger.warning(f"Scope: Failed to connect to TheSkyX: {e}")
            self.connected = False
            return False

    def _send(self, command):
        """Send a JavaScript command to TheSkyX and return the output."""
        try:
            sockobj = socket(AF_INET, SOCK_STREAM)
            sockobj.settimeout(5.0)
            sockobj.connect((self.host, self.port))
            sockobj.send(bytes(
                '/* Java Script */\n'
                '/* Socket Start Packet */\n' + command +
                '\n/* Socket End Packet */\n', 'utf8'))
            output = sockobj.recv(2048)
            output = output.decode('utf-8')
            sockobj.shutdown(SHUT_RDWR)
            sockobj.close()
            return output.split("|")[0]
        except error as msg:
            raise ScopeConnectionError(
                f"Connection to {self.host}:{self.port} failed: {msg}")

    def jog(self, dx, dy):
        """
        Jog the telescope by a small amount.

        Args:
            dx: East/West movement in arcseconds (positive=East, negative=West)
            dy: North/South movement in arcseconds (positive=North, negative=South)
        """
        if not self.connected:
            return

        # Clamp to ±9.9 arcsec
        dx = max(-9.9, min(9.9, dx))
        dy = max(-9.9, min(9.9, dy))

        # North/South jog
        if dy > 0.002:
            command = f'var Out = ""; sky6RASCOMTele.Jog({dy}, "N");'
            try:
                self._send(command)
            except ScopeConnectionError as e:
                logger.error(f"Scope jog N failed: {e}")
        elif dy < -0.002:
            command = f'var Out = ""; sky6RASCOMTele.Jog({-dy}, "S");'
            try:
                self._send(command)
            except ScopeConnectionError as e:
                logger.error(f"Scope jog S failed: {e}")

        # East/West jog
        if dx > 0.002:
            command = f'var Out = ""; sky6RASCOMTele.Jog({dx}, "E");'
            try:
                self._send(command)
            except ScopeConnectionError as e:
                logger.error(f"Scope jog E failed: {e}")
        elif dx < -0.002:
            command = f'var Out = ""; sky6RASCOMTele.Jog({-dx}, "W");'
            try:
                self._send(command)
            except ScopeConnectionError as e:
                logger.error(f"Scope jog W failed: {e}")

    def get_radec(self):
        """Get current RA and Dec from the telescope."""
        if not self.connected:
            return None, None
        try:
            command = """
                var Out;
                sky6RASCOMTele.GetRaDec();
                Out = String(sky6RASCOMTele.dRa) + " " + String(sky6RASCOMTele.dDec);
            """
            output = self._send(command).splitlines()[0].split()
            return float(output[0]), float(output[1])
        except Exception as e:
            logger.error(f"Scope get_radec failed: {e}")
            return None, None

    def is_connected(self):
        """Check if telescope is connected."""
        return self.connected


# Global scope instance
_scope = None


def init_scope(host="localhost", port=3040):
    """Initialize the global scope instance."""
    global _scope
    _scope = Scope(host, port)
    return _scope.init()


def get_scope():
    """Get the global scope instance."""
    return _scope
