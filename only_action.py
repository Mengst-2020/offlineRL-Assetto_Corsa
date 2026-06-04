import json
class DriverControls(dict):
    def __init__(self):
        self.set_defaults()
        self.scale_steer = 0.6 # an input of 1 will be scaled to this value
        self.local_controls = Controls()

    def set_defaults(self):
        # steer, acc and brake are in the range [-1, 1]
        self["steer"] = 0
        self["acc"] = -1
        self["brake"] = -1
        self["enable_clutch"] = 0
        self["clutch"] = -1
        self["enable_gear_shift"] = 0
        self["shift_up"] = 0
        self["shift_down"] = 0

    def set_controls(self, steer, acc, brake, enable_clutch=False, clutch=-1,
                     enable_gear_shift=False, shift_up=False, shift_down=False):
        self["steer"] = steer * self.scale_steer
        self["acc"] = acc
        self["brake"] = brake
        self["enable_clutch"] = int(enable_clutch)
        self["clutch"] = clutch
        self["enable_gear_shift"] = int(enable_gear_shift)
        self["shift_up"] = int(shift_up)
        self["shift_down"] = int(shift_down)

    def apply_local_controls(self):
        self.local_controls.set_controls(steer=self["steer"],
                                         acc=self["acc"],
                                         brake=self["brake"],
                                         enable_clutch=self["enable_clutch"],
                                         clutch=self["clutch"],
                                         enable_gear_shift=self["enable_gear_shift"],
                                         shift_up=self["shift_up"],
                                         shift_down=self["shift_down"])

    def export(self):
        return json.dumps(self)
    

import ctypes
import struct, time
import math

CONST_DLL_VJOY = "C:\\Program Files\\vJoy\\x64\\vJoyInterface.dll"
class vJoy(object):
    def __init__(self, reference=1):
        self.handle = None
        self.dll = ctypes.CDLL(CONST_DLL_VJOY)
        self.reference = reference
        self.acquired = False

    def open(self):
        if self.dll.AcquireVJD(self.reference):
            self.acquired = True
            return True
        return False

    def close(self):
        if self.dll.RelinquishVJD(self.reference):
            self.acquired = False
            return True
        return False

    def generateJoystickPosition(self,
                                 wThrottle=0, wRudder=0, wAileron=0,
                                 wAxisX=0, wAxisY=0, wAxisZ=0,
                                 wAxisXRot=0, wAxisYRot=0, wAxisZRot=0,
                                 wSlider=0, wDial=0, wWheel=0,
                                 wAxisVX=0, wAxisVY=0, wAxisVZ=0,
                                 wAxisVBRX=0, wAxisVBRY=0, wAxisVBRZ=0,
                                 lButtons=0, bHats=0, bHatsEx1=0, bHatsEx2=0, bHatsEx3=0):
        """
        typedef struct _JOYSTICK_POSITION
        {
            BYTE    bDevice; // Index of device. 1-based
            LONG    wThrottle;
            LONG    wRudder;
            LONG    wAileron;
            LONG    wAxisX;
            LONG    wAxisY;
            LONG    wAxisZ;
            LONG    wAxisXRot;
            LONG    wAxisYRot;
            LONG    wAxisZRot;
            LONG    wSlider;
            LONG    wDial;
            LONG    wWheel;
            LONG    wAxisVX;
            LONG    wAxisVY;
            LONG    wAxisVZ;
            LONG    wAxisVBRX;
            LONG    wAxisVBRY;
            LONG    wAxisVBRZ;
            LONG    lButtons;   // 32 buttons: 0x00000001 means button1 is pressed, 0x80000000 -> button32 is pressed
            DWORD   bHats;      // Lower 4 bits: HAT switch or 16-bit of continuous HAT switch
                        DWORD   bHatsEx1;   // 16-bit of continuous HAT switch
                        DWORD   bHatsEx2;   // 16-bit of continuous HAT switch
                        DWORD   bHatsEx3;   // 16-bit of continuous HAT switch
        } JOYSTICK_POSITION, *PJOYSTICK_POSITION;
        """
        joyPosFormat = "BlllllllllllllllllllIIII"
        pos = struct.pack(joyPosFormat, self.reference, wThrottle, wRudder,
                          wAileron, wAxisX, wAxisY, wAxisZ, wAxisXRot, wAxisYRot,
                          wAxisZRot, wSlider, wDial, wWheel, wAxisVX, wAxisVY, wAxisVZ,
                          wAxisVBRX, wAxisVBRY, wAxisVBRZ, lButtons, bHats, bHatsEx1, bHatsEx2, bHatsEx3)
        return pos

    def update(self, joystickPosition):
        if self.dll.UpdateVJD(self.reference, joystickPosition):
            return True
        return False

    # Not working, send buttons one by one
    def sendButtons(self, bState):
        joyPosition = self.generateJoystickPosition(lButtons=bState)
        return self.update(joyPosition)

    def setButton(self, index, state):
        if self.dll.SetBtn(state, self.reference, index):
            return True
        return False

# valueX between 0 and 2
# valueY, valueZ between 0 and 1
# scale between 0 and 16000
def setJoy(valueX, valueY, valueZ, onButtons, scale):
    xPos = int(valueX * scale)
    yPos = int(valueY * 2 * scale)
    zPos = int(valueZ * 2 * scale)
    #yPos = int(valueY * scale)
    #zPos = int(valueZ * scale)
    if onButtons != 0:
        joystickPosition = vj.generateJoystickPosition(wAxisX= xPos, wAxisY=yPos, wAxisZ=zPos, lButtons=onButtons)
        vj.update(joystickPosition)
        time.sleep(0.01)

    joystickPosition = vj.generateJoystickPosition(wAxisX= xPos, wAxisY=yPos, wAxisZ=zPos)
    vj.update(joystickPosition)


# Only for testing
def gearUp():
    #press
    setJoy(1 ,0.3, 0, 0x00000001, 16384)

    #release
    setJoy(1 ,0.3, 0, 0, 16384)



import logging
logger = logging.getLogger(__name__)

SCALE = 16384

class Controls(object):
    def __init__(self):
        self.onButtons = 0
        #self.vj = vj
        self.vj = vJoy()

        self.vj.open()

        # internal state
        self.steer = 1.0        # [0, 2]
        self.acc = 0.0          # [0, 1]
        self.brake = 0.0        # [0, 1]
        self.enable_clutch = 0
        self.clutch = 0.
        self.enable_gear_shift = 0.
        self.shift_up = 0.
        self.shift_down = 0.

        # commands
        self.steer_cmd = 0.0 # [-1,1]
        self.pedal_cmd = 0.0 # [-1,1]
        self.brake_cmd = 0.0 # [-1,1]

        self.ct_12_stop = False

    def close(self):
        self.vj.close()

    def trigger_emergency_stop(self):
        self.ct_12_stop = True
        self.steer= 1.0
        self.acc = 0.0
        self.brake = 0.5
        logger.info("CT12 triggered")
        self.update()

    def set_controls(self, steer, acc, brake, enable_clutch=False, clutch=-1, enable_gear_shift=False, shift_up=False, shift_down=False):
        self.steer_cmd = steer
        self.pedal_cmd = acc
        self.brake_cmd = brake

        if not self.ct_12_stop:
            self.steer = self.steer_cmd + 1
            if(self.steer < 0):
                self.steer = 0
            elif(self.steer > 2):
                self.steer = 2

            # Acc
            self.acc = (self.pedal_cmd + 1) / 2
            if(self.acc < 0):
                self.acc = 0
            elif(self.acc > 1):
                self.acc = 1

            # brake
            self.brake = (self.brake_cmd + 1) / 2
            if(self.brake < 0):
                self.brake = 0
            elif(self.brake > 1):
                self.brake = 1

            # # set gear
            # gear = info.physics.gear - 1
            # print("current gear: %d | required gear is %d " % (gear, data.data))
            # if gear > data.data:
            #     setJoy(self.steer, self.acc, self.brake, 0x00000002, SCALE)
            # elif gear < data.data:
            #     setJoy(self.steer, self.acc, self.brake, 0x00000001, SCALE)

        self.update()

    def update(self):
        self.setJoy(self.steer, self.acc, self.brake, self.onButtons, SCALE)

    def setJoy(self, valueX, valueY, valueZ, onButtons, scale):
        """
        valueX between 0 and 2
        valueY, valueZ between 0 and 1
        scale between 0 and 16000
        """
        xPos = int(valueX * scale)
        yPos = int(valueY * 2 * scale)
        zPos = int(valueZ * 2 * scale)
        if onButtons != 0:
            joystickPosition = self.vj.generateJoystickPosition(wAxisX= xPos, wAxisY=yPos, wAxisZ=zPos, lButtons=onButtons)
            self.vj.update(joystickPosition)

        joystickPosition = self.vj.generateJoystickPosition(wAxisX= xPos, wAxisY=yPos, wAxisZ=zPos)
        self.vj.update(joystickPosition)

    def gearUp(self):
        """ For testing """

        #press
        self.setJoy(1 ,0.3, 0, 0x00000001, 16384)

        #release
        self.setJoy(1 ,0.3, 0, 0, 16384)

if __name__ == "__main__":
    hello=DriverControls()
    for i in range(100):
        if i % 2 == 0:
            steer = .1
        else:
            steer = -.1
        hello.set_controls(steer=steer, acc=0.5, brake=-1.)
        hello.apply_local_controls