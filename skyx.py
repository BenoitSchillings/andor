''' Module to handle connections to TheSkyX
The classes are defined to match the classes in Script TheSkyX. This isn't
really necessary as they all just send the javascript to TheSkyX via
SkyXConnection._send().
'''
from __future__ import print_function

import logging
import time
from socket import socket, AF_INET, SOCK_STREAM, SHUT_RDWR, error


#-------------------------------------------------------------------------


logger = logging.getLogger(__name__)

class Singleton(object):
    ''' Singleton class so we dont have to keep specifing host and port'''
    def __init__(self, klass):
        ''' Initiator '''
        self.klass = klass
        self.instance = None

    def __call__(self, *args, **kwds):
        ''' When called as a function return our singleton instance. '''
        if self.instance is None:
            self.instance = self.klass(*args, **kwds)
        return self.instance

#-------------------------------------------------------------------------

class SkyxObjectNotFoundError(Exception):
    ''' Exception for objects not found in SkyX.
    '''
    def __init__(self, value):
        ''' init'''
        super(SkyxObjectNotFoundError, self).__init__(value)
        self.value = value

    def __str__(self):
        ''' returns the error string '''
        return repr(self.value)

#-------------------------------------------------------------------------

class SkyxConnectionError(Exception):
    ''' Exception for Failures to Connect to SkyX
    '''
    def __init__(self, value):
        ''' init'''
        super(SkyxConnectionError, self).__init__(value)
        self.value = value

    def __str__(self):
        ''' returns the error string '''
        return repr(self.value)

#-------------------------------------------------------------------------


class SkyxTypeError(Exception):
    ''' Exception for Failures to Connect to SkyX
    '''
    def __init__(self, value):
        ''' init'''
        super(SkyxTypeError, self).__init__(value)
        self.value = value

    def __str__(self):
        ''' returns the error string '''
        return repr(self.value)

#-------------------------------------------------------------------------
    
@Singleton
class SkyXConnection(object):
    ''' Class to handle connections to TheSkyX
    '''
    def __init__(self, host="localhost", port=3040):
        ''' define host and port for TheSkyX.
        '''
        self.host = host
        self.port = port
        
    def reconfigure(self,host="localhost", port=3040):
        ''' If we need to chane ip we can do so this way'''
        self.host = host
        self.port = port
                
    def _send(self, command):
        ''' sends a js script to TheSkyX and returns the output.
        '''
        try:
            logger.debug(command)
            sockobj = socket(AF_INET, SOCK_STREAM)
            sockobj.connect((self.host, self.port))
            sockobj.send(bytes('/* Java Script */\n' +
                               '/* Socket Start Packet */\n' + command +
                               '\n/* Socket End Packet */\n', 'utf8'))
            output = sockobj.recv(2048)
            output = output.decode('utf-8')
            logger.debug(output)
            sockobj.shutdown(SHUT_RDWR)
            sockobj.close()

            return output.split("|")[0]
        except error as msg:
            raise SkyxConnectionError("Connection to " + self.host + ":" + \
                                      str(self.port) + " failed. :" + str(msg))


#-------------------------------------------------------------------------

    def find(self, target):
        ''' Find a target
            target can be a defined object or a decimal ra,dec
        '''
        output = self._send('sky6StarChart.Find("' + target + '")')
        if output == "undefined":
            return True
        else:
            raise SkyxObjectNotFoundError(target)
                                    
 
#-------------------------------------------------------------------------

 
class sky6RASCOMTele(object):
    ''' Class to implement the ccdsoftCamera script class
    '''
    def __init__(self, host="localhost", port=3040):
        ''' Define connection
        '''
        self.conn = SkyXConnection(host, port)
        
    def Connect(self):
        ''' Connect to the telescope
        '''
        command = """
                  var Out;
                  sky6RASCOMTele.Connect();
                  Out = sky6RASCOMTele.IsConnected"""

        output = self.conn._send(command).splitlines()

        if int(output[0]) != 1:
            raise SkyxTypeError("Telescope not connected. "+\
                                "sky6RASCOMTele.IsConnected=" + output[0])
        return True
        
    def Disconnect(self):
        ''' Disconnect the telescope
            Whatever this actually does...
        '''
        command = """
                  var Out;
                  sky6RASCOMTele.Disconnect();
                  Out = sky6RASCOMTele.IsConnected"""
        output = self.conn._send(command, 'utf8').splitlines()
        if int(output[0]) != 0:
            raise SkyxTypeError("Telescope still connected. " +\
                                "sky6RASCOMTele.IsConnected=" + output[0])
        return True

    def GetRaDec(self):
        ''' Get the current RA and Dec
        '''
        command = """
                  var Out;
                  sky6RASCOMTele.GetRaDec();
                  Out = String(sky6RASCOMTele.dRa) + " " + String(sky6RASCOMTele.dDec);
                  """
        output = self.conn._send(command).splitlines()[0].split()      
        return output

    def get_az_alt(self):
        ''' Get the current Azimuth and Altitude
        '''
        command = """
                  var Out;
                  sky6RASCOMTele.GetAzAlt();
                  Out = String(sky6RASCOMTele.dAz) + " " + String(sky6RASCOMTele.dAlt);
                  """
        output = self.conn._send(command).splitlines()[0].split()
        return [float(output[0]), float(output[1])]

    def is_connected(self):
        ''' Returns True if the telescope is connected, False otherwise.
        '''
        command = "var Out = sky6RASCOMTele.IsConnected;"
        output = self.conn._send(command).splitlines()[0]
        return int(output) == 1

    def is_parked(self):
        ''' Returns True if the telescope is parked, False otherwise.
        '''
        command = "var Out = sky6RASCOMTele.IsParked;"
        output = self.conn._send(command).splitlines()[0]
        return int(output) == 1

    def is_tracking(self):
        ''' Returns True if the telescope is tracking, False otherwise.
        '''
        command = "var Out = sky6RASCOMTele.IsTracking;"
        output = self.conn._send(command).splitlines()[0]
        return int(output) == 1
    
    def park(self):
        command = """
                sky6RASCOMTele.Park();
                """     
        output = self.conn._send(command).splitlines()
        return 0

    
    def unpark(self):
        command = """
                sky6RASCOMTele.Unpark();
                """     
        output = self.conn._send(command).splitlines()
        return 0

    
    def isSlewComplete(self):
        command = """
                Out = sky6RASCOMTele.IsSlewComplete;
                """     
        output = self.conn._send(command).splitlines()
        return output



    def Sync(self, pos):
        ''' Sync to a given pos [ra, dec]
            ra, dec should be Jnow coordinates
        '''
        command = """
                var Out = "";
                sky6RASCOMTele.Sync(""" + str(pos[0]) + "," + str(pos[1]) + """, "pyskyx");
                """
        output = self.conn._send(command).splitlines()
        print(output)
        time.sleep(1)
        print(self.GetRaDec())

    def goto(self, ra, dec):
        command = """
                sky6RASCOMTele.SlewToRaDec(""" + str(ra) + "," + str(dec) + """, "cxx");
                """
        print(command)
        output = self.conn._send(command).splitlines()
        print(output)
        time.sleep(0.4)
    
    def rate(self, d_ra, d_dec):
        command = """
                sky6RASCOMTele.SetTracking(1, 0, """ + str(d_ra) + "," + str(d_dec) + """);
                """
        print(command)
        output = self.conn._send(command).splitlines()
        print(output)
        time.sleep(0.1)

    def get_rate(self):
        ''' Get the current RA and Dec tracking rates
        '''
        command = """
                  var Out;
                  Out = String(sky6RASCOMTele.dRaRate) + " " + String(sky6RASCOMTele.dDecRate);
                  """
        output = self.conn._send(command).splitlines()[0].split()
        print(output) 
        return [float(output[0]), float(output[1])]

    def stop(self):
        # Stop tracking by reverting to default sidereal rate
        command = """
                var Out = "";
                sky6RASCOMTele.SetTracking(1, 1, 0, 0);
                """
        output = self.conn._send(command).splitlines()
        print(output)
        time.sleep(1)
        
    def bump(self, dx, dy):
        self.jog(dx, dy) 
        return 
        dx = dx * 1000
        dy = dy * 1000
        dx = max(dx, -2999)
        dx = min(dx, 2999)
        dy = max(dy, -2999)
        dy = min(dy, 2999)
        quote = '"'
        
        cmd = ""
        
        if (dy > 4):
            cmd = quote + ":Ms" + str(int(dy)) + "#" + quote
            
        if (dy < -4):
            cmd = quote + ":Mn" + str(int(-dy)) + "#" + quote
        
        if (not(cmd == "")):
            command = """
                var Out = "";
                sky6RASCOMTele.DoCommand(3, """
            command = command + cmd + ")\n"
            print(command)
            output = self.conn._send(command).splitlines()
            print(output)

        cmd1 = ""      
        if (dx > 4):
            cmd1 = quote + ":Me" + str(int(dx)) + "#" + quote
        if (dx < -4):
            cmd1 = quote + ":Mw" + str(int(-dx)) + "#" + quote
        
        if (not(cmd1 == "")):
            command = """
                var Out = "";
                sky6RASCOMTele.DoCommand(3, """
            command = command + cmd1 + ")\n"
            print(command)
            output = self.conn._send(command).splitlines()
            print(output)
       
    def jog(self, dx, dy):
        dx = max(dx, -9.9)
        dx = min(dx, 9.9)
        dy = max(dy, -9.9)
        dy = min(dy, 9.9)
        #print("jog ",dx,dy)
        quote = '"'
        
        cmd = None 
        if (dy > 0.002):
            cmd = "Jog(" + str(dy) + ',"N"' + ")"
            
        if (dy < -0.002):
            cmd = "Jog(" + str(-dy) + ',"S"' + ")"
        
        if (not(cmd is None)):
            command = """
                var Out = "";
                sky6RASCOMTele."""
            command = command + cmd + "\n"

            #print(command)
            output = self.conn._send(command).splitlines()
            #print(output)

        cmd1 = None 
        if (dx > 0.002):
            cmd1 = "Jog(" + str(dx) + ',"E"' + ")"
            
        if (dx < -0.002):
            cmd1 = "Jog(" + str(-dx) + ',"W"' + ")"
        
        if (not(cmd1 is None)):
            command = """
                var Out = "";
                sky6RASCOMTele."""
            command = command + cmd1 + "\n"
            print(command)
            output = self.conn._send(command).splitlines()
            print(output)


#-------------------------------------------------------------------------

class sky6FilterWheel(object):
    ''' Class to implement the sky6FilterWheel script class
    '''
    def __init__(self, host="localhost", port=3040):
        ''' Define connection
        '''
        self.conn = SkyXConnection(host, port)

    def connect(self):
        ''' Connect to the filter wheel
        '''
        command = "sky6FilterWheel.Connect();"
        self.conn._send(command)
        return True

    def disconnect(self):
        ''' Disconnect from the filter wheel
        '''
        command = "sky6FilterWheel.Disconnect();"
        self.conn._send(command)
        return True

    def goto(self, position):
        ''' Move to a specific filter position
        '''
        command = f"sky6FilterWheel.goto({position});"
        self.conn._send(command)
        return True

    def get_filter_name(self, position):
        ''' Get the name of the filter at a given position
        '''
        command = f"var Out = sky6FilterWheel.FilterName({position});"
        return self.conn._send(command).splitlines()[0]

    def get_count(self):
        ''' Get the total number of filter slots
        '''
        command = "var Out = sky6FilterWheel.Count;"
        return int(self.conn._send(command).splitlines()[0])

    def get_position(self):
        ''' Get the current filter position
        '''
        command = "var Out = sky6FilterWheel.Position;"
        return int(self.conn._send(command).splitlines()[0])




