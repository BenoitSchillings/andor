#!/usr/bin/env python3

import binascii
import struct
import usb1

# Constants
USB_WRITE_ENDPOINT = 0x02
USB_READ_ENDPOINT = 0x82

def validate_read(expected, actual, msg):
    """Validate the USB read operation."""
    if expected != actual:
        print(f'Failed {msg}')
        print(f'  Expected: {binascii.hexlify(expected)}')
        print(f'  Actual:   {binascii.hexlify(actual)}')

def bulk_read(dev, endpoint, length, timeout=1000):
    """Perform a bulk read operation."""
    return dev.bulkRead(endpoint, length, timeout=timeout)

def bulk_write(dev, endpoint, data, timeout=1000):
    """Perform a bulk write operation."""
    dev.bulkWrite(endpoint, data, timeout=timeout)

def write_to_usb(dev, command):
    """Write a command to the USB device."""
    bulk_write(dev, USB_WRITE_ENDPOINT, command)

def read_from_usb(dev):
    """Read data from the USB device."""
    return bulk_read(dev, USB_READ_ENDPOINT, 2)

def fli_getsteppos(dev):
    """Get stepper position for pre-Atlas hardware."""
    command_low = struct.pack('!H', 0x6000)
    write_to_usb(dev, command_low)
    response_low = read_from_usb(dev)
    if struct.unpack('!H', response_low)[0] & 0xf000 != 0x6000:
        return None

    command_high = struct.pack('!H', 0x6001)
    write_to_usb(dev, command_high)
    response_high = read_from_usb(dev)
    if struct.unpack('!H', response_high)[0] & 0xf000 != 0x6000:
        return None

    poslow = struct.unpack('!H', response_low)[0]
    poshigh = struct.unpack('!H', response_high)[0]
    if poshigh & 0x0080 > 0:
        pos = ((~poslow) & 0xff) + 1
        pos += (256 * ((~poshigh) & 0xff))
        pos = -pos
    else:
        pos = (poslow & 0xff) + 256 * (poshigh & 0xff)
    return pos

def move_stepper_motor(dev, n):
    """Move the stepper motor by n steps."""
    if n == 0:
        return True

    direction = 0x9000 if n > 0 else 0xA000
    absolute_steps = abs(n)
    command = struct.pack('!H', direction | (absolute_steps & 0x0FFF))
    write_to_usb(dev, command)
    return True

def fli_getstepperstatus(dev):
    """Get stepper status."""
    command = struct.pack('!H', 0xB000)
    write_to_usb(dev, command)
    response = read_from_usb(dev)
    status = struct.unpack('!H', response)[0]
    return status

def home_stepper_motor(dev):
    """Home the stepper motor."""
    command = struct.pack('!H', 0xF000)
    write_to_usb(dev, command)
    return True



def open_dev(vid_want, pid_want, usbcontext=None):
    """Open a USB device with the specified VID and PID."""
    if usbcontext is None:
        usbcontext = usb1.USBContext()

    print('Scanning for devices...')
    for udev in usbcontext.getDeviceList(skip_on_error=True):
        vid, pid = udev.getVendorID(), udev.getProductID()
        if (vid, pid) == (vid_want, pid_want):
            print(f'Found device\nBus {udev.getBusNumber():03} Device {udev.getDeviceAddress():03}: ID {vid:04x}:{pid:04x}')
            return udev.open()

    raise Exception('Failed to find a device')



class focuser:
    def __init__(self):
        vid_want, pid_want = 0x0F18, 0x0006


        self.usbcontext = usb1.USBContext()
        self.dev = open_dev(vid_want, pid_want, self.usbcontext)
        self.dev.claimInterface(0)
        self.dev.resetDevice()
        self.load()


    def save(self):
        with open('focus_pos.txt', 'w') as file:
            file.write(str(self.pos))

    def load(self):
        with open('focus_pos.txt', 'r') as file:
            self.pos = int(file.read())


    def move_focus(self, delta):
        move_stepper_motor(self.dev, delta) 
        self.pos = self.pos + delta
        self.save()


    def move_to(self, pos):
        delta = pos - self.pos
        self.move_focus(delta)
        self.save()
        
    def get_pos(self):
        return fli_getsteppos(self.dev) 

    def get_abs_pos(self):
        return self.pos  

    def home(self):
        home_stepper_motor(self.dev)
        self.pos = 0
        self.save()

    def status(self):
        return fli_getstepperstatus(self.dev)



if __name__ == '__main__':
    foc = focuser()

    foc.home()
    print(foc.status())
    foc.move_focus(20)
    print(foc.get_pos())
