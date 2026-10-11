# -*- coding: utf-8 -*-
"""
Created on Wed May 21 15:27:07 2025

@author: XingPan
"""
import serial
import serial.tools.list_ports
import numpy as np
import time

def list_available_ports():
    """
    Lists available serial ports on the system.
    """
    ports = serial.tools.list_ports.comports()
    print("Available COM ports:")
    if not ports:
        print("  No COM ports found.")
        return []
    
    port_list = []
    for i, port_info in enumerate(ports):
        print(f"  {i}: {port_info.device} - {port_info.description} [{port_info.manufacturer}]")
        port_list.append(port_info.device)
    return port_list

class ER1000:
    def __init__(self,strComPort):

        # --- Serial Port Configuration ---
        baud_rate = 115200  # Common baud rate, adjust as needed for your device
        byte_size = serial.EIGHTBITS
        parity = serial.PARITY_NONE
        stop_bits = serial.STOPBITS_ONE
        timeout_read = 1  # Timeout for read operations in seconds
        timeout_write = 1 # Timeout for write operations in seconds
    
        # Initialize serial connection variable
        ser = None 
        # --- Open the Serial Port ---
        print(f"Configuring port {strComPort} with Baud Rate: {baud_rate}...")
        ser = serial.Serial(
            port=strComPort,
            baudrate=baud_rate,
            bytesize=byte_size,
            parity=parity,
            stopbits=stop_bits,
            timeout=timeout_read,       # Read timeout
            write_timeout=timeout_write # Write timeout
        )
        self.session=ser
        self.is_open=ser.is_open
        
        self.session.flushInput()
        self.session.write(b'E0\r\n') # echo off 
        self.session.flushInput()
        self.session.write(b'C1\r\n') # C1 capture mode <<<< important

    def getCommandList(self):
        
        self.session.flushInput()
        self.session.write(b'?')
        resp=self.session.readlines()
        return resp
    
    def getOperationMode(self):
        '''
        Returns
        -------
        curtent operation mode
        M0: ER mode
        M1: logging mode
        M2: power mode

        '''
        self.session.flushInput()
        self.session.write(b'M?\r\n')
        resp=self.session.readlines()
        return resp[1].decode('utf-8')
 
    def startMeasurement(self):
        self.session.flushInput()
        self.session.write(b'G1')
       
    def stopMeasurement(self):
        self.session.flushInput()
        self.session.write(b'G0')
                
    def measurePER(self,waitTime_s=5):
        self.session.flushInput()
        self.session.write(b'G1\r\n') # G1 start measurement
        time.sleep(waitTime_s)
        self.session.write(b'G0\r\n') # G0 stop measurement
        resp=self.session.readlines()
        rcds=[]
        for bAry in resp:
            if (b'dB' in bAry) and (b'dBm\r\n' in bAry):
                rcd=[]
                elms=bAry.split(b' ')
                for i, e in enumerate(elms):
                    if i!=1:
                        try:
                            rcd.append(float(e.split(b'dB')[0]))
                        except:
                            if e.split(b'dB')[0][0]==b'>':
                                rcd.append(float(e.split(b'dB')[0][1:]))
                    else:
                        rcd.append(float(e.split(b'\xf8')[0]))
                rcds.append(rcd)                                    
        
        return rcds
#%%
if __name__ == "__main__":
    list_available_ports()
    perMeter=ER1000('com3')
    #
    perMeter.session.get_settings() # works
    
    perMeter.session.flushInput()
    perMeter.session.write(b'E0\r\n') # echo off 

    perMeter.session.flushInput()
    perMeter.session.write(b'B?\r\n') 
    resp=perMeter.session.readlines()
    resp[1].decode('utf-8') # get baud rate
    
    perMeter.session.flushInput()
    perMeter.session.write(b'M0\r\n') # #M0: set to normal ER mode
    
    perMeter.session.flushInput()
    perMeter.session.write(b'M?\r\n') # #M0: set to normal ER mode
    
    resp=perMeter.session.readlines()
    resp[0].decode('utf-8') 
    
    perMeter.session.flushInput()
    perMeter.session.write(b'P?\r\n') # P? get current power
    
    resp=perMeter.session.readlines()
    resp[0].decode('utf-8') 
    
    perMeter.session.flushInput()
    perMeter.session.write(b'G?\r\n') # G?: get current ER measurement in normal ER mode
    
    resp=perMeter.session.readlines()
    print(resp)

    perMeter.session.flushInput()
    perMeter.session.write(b'C1\r\n') # C1 capture mode <<<< important

    perMeter.session.flushInput()
    perMeter.session.write(b'G1\r\n') # G1 start measurement
    time.sleep(5)
    perMeter.session.write(b'G0\r\n') # G0 stop measurement
    
    resp=perMeter.session.readlines()
    print(resp)
    
   #%%
    perMeter.session.flushInput()
    perMeter.session.write(b'?\r\n') # ? list of remote commands
   
    resp=perMeter.session.readlines()
    print(resp)
#%%    
    pauseTime=15
    rcds=[]
    t0=time.time()
    et_s=0
    while True:
        et_s=np.round(time.time()-t0,1)
        permeasurements=perMeter.measurePER()
        for mea in permeasurements:
            rcd=[et_s]+mea
            rcds.append(rcd)
            print(rcd)
        time.sleep(pauseTime)
#%%
    df=pd.DataFrame(rcds,columns=['et_s','PER_dB','theta_deg','power_dBm'])
    
    df.to_csv(dataDir+'\\'+'01-portA-ch0-PER vs time_5-24-2025.csv')
    
    #%%
    '''
    close session
    '''
    perMeter.session.close()