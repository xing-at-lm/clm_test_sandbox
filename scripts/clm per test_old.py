# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 11:29:58 2026

@author: MountainViewLabUser
"""
#%%
import os
import sys
from tqdm import tqdm
from scipy.signal import find_peaks
import copy
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import clm_tile_control_kurtz
import time
from datetime import datetime, timezone
import pyvisa as visa
import ozOptics_ER1000
import importlib
#import yokogawaAQ6380
#import ThorlabsPM100D
#import ThorlabsLK220
import yaml
import pickle
import random
import matplotlib.dates as mdates

import marlow_clm_control_api as mca

sys.path.append(os.path.dirname(__file__))
#import Temp_Controller as TC


sys.path.append(r'C:\Program Files (x86)\Thorlabs\PTC1\Sample\Thorlabs_PTC1_PythonSDK')
import PTC1_COMMAND_LIB as ptc1_module


#from pandasgui import show

sns.set_palette("bright")

#%%
target_grid_nm = np.array([
  1301.47, 1302.60, 1303.73, 1304.87,
  1306.01, 1307.14, 1308.28, 1309.43,
  1310.57, 1311.72, 1312.87, 1314.02,
  1315.17, 1316.33, 1317.48, 1318.64
  ])
target_grid_nm_A=target_grid_nm[0::2]
target_grid_nm_B=target_grid_nm[1::2]

rm=visa.ResourceManager()

dataDir=r'C:\Users\MountainViewLabUser\xingpan\testdata'

#%%
tb=mca.get_testboard(AARDVARK_SN_P0 = 2239867798,AARDVARK_SN_Io = 2239880394,AARDVARK_SN_P1 =2239789566,testBoardType='EVK')
#%%
idxTile=3
tile=tb.module.packages[0].tiles[idxTile]

#%%
mca.get_tile_fw_version(tb)
mca.read_fru_info(tile)
#%%
mca.initialize_tile(tb,tile=tile)
mca.set_clm_control_config(tile,dither_period_s=2,kp_tpic=0.3,kp_ld=0.3,kp_voa=0.3,amp_t_mux=3,ALPHA_I_LD=0.5)
time.sleep(0.2)
mca.optical_power_on(tile)
#%%
print(tile.read_lane_enables(mca.ClmTile.Bank.BANK_A))
print(tile.read_lane_enables(mca.ClmTile.Bank.BANK_B))

print(tile.read_aux_bias_current_monitors(mca.ClmTile.Bank.BANK_A))
print(tile.read_aux_bias_current_monitors(mca.ClmTile.Bank.BANK_B))

print(tile.read_bias_current_monitors(mca.ClmTile.Bank.BANK_A))
print(tile.read_bias_current_monitors(mca.ClmTile.Bank.BANK_B))

print(tile.read_optical_power_monitors(mca.ClmTile.Bank.BANK_A))
print(tile.read_optical_power_monitors(mca.ClmTile.Bank.BANK_B))
#%% 
'''
FW upgrade
'''
dir_fw=r'C:\Users\MountainViewLabUser\gitrepos\cmis-host-controller\bin_files'
filename_fw='clm_rel_marlow_tile_v1.7.bin'
mca.update_fw(tile,dir_fw,filename_fw)

#%%
'''
test clm power stability over # of channels enabled, random sample
'''
dataDir=r'C:\Users\MountainViewLabUser\xingpan\testdata'

sample_time=300
sample_int=2
pause_time=60

pow_tgts=[6,8,10,12]

dfs=[]
for pow_tgt in tqdm(pow_tgts,desc='power tgt'):
    mca.set_clm_output_targets(tile,[0]*16,[pow_tgt]*16)
    time.sleep(pause_time)
    for Nchs in tqdm(random.sample(range(1,17,2),8),desc='enable chs'):
        ch_mask=[0]*16
        picked_chs=random.sample(range(16),Nchs)
        for ch in picked_chs:
            ch_mask[ch]=1
        
        mca.clm_lane_enable(tile,chEnableMask=ch_mask)
        
        rcds,cols=mca.get_clm_record([tile], duration_s=sample_time,interval_s=sample_int)
        
        df=pd.DataFrame(rcds,columns=cols)
        df['pow_tgt']=pow_tgt
        for ch,chenb in zip(range(16),ch_mask):
            df[f'ch{ch}_enable']=chenb
        df['Nch_enabled']=Nchs
        
        dfs.append(df)
        
dft=pd.concat(dfs)

now = datetime.now()
timeStamp = now.strftime("%Y-%m-%d-%H-%M")

filename='clm stability over pow target and Nchs_sample duration-%i_sample interval-%i__%s.csv'\
    %(sample_time,sample_int,timeStamp)
dft.to_csv(dataDir+'\\'+filename)   
#%%
'''
test clm power stability over all chs to 1,
'''
dataDir=r'C:\Users\MountainViewLabUser\xingpan\testdata'

sample_time=300
sample_int=2
pause_time=60
dither_period_s=6
kp_tpic=0.3
kp_ld=0.3
kp_voa=0.3
ALPHA_I_LD=0.5
mca.set_clm_control_config(tile,dither_period_s=dither_period_s,kp_tpic=kp_tpic,kp_ld=kp_ld,kp_voa=kp_voa,amp_t_mux=3,ALPHA_I_LD=ALPHA_I_LD)

pow_tgts=[6,8,10,12]

dfs=[]
for pow_tgt in tqdm(pow_tgts,desc='power tgt'):
    mca.set_clm_output_targets(tile,[0]*16,[pow_tgt]*16)
    # 1st all chs on
    ch_mask=[1]*16
    mca.clm_lane_enable(tile,chEnableMask=ch_mask)    
    rcds,cols=mca.get_clm_record([tile], duration_s=sample_time,interval_s=sample_int)
    
    df=pd.DataFrame(rcds,columns=cols)
    df['pow_tgt']=pow_tgt
    for ch in range(16):
        df[f'ch{ch}_enable']=1
    df['Nch_enabled']=16
    dfs.append(df)
    
    #2nd 1 ch on
    ch_mask=[0]*16
    for ch in range(16):
        ch_mask[ch]=1
    
    mca.clm_lane_enable(tile,chEnableMask=ch_mask)    
    rcds,cols=mca.get_clm_record([tile], duration_s=sample_time,interval_s=sample_int)
    
    df=pd.DataFrame(rcds,columns=cols)
    df['pow_tgt']=pow_tgt
    for ch,chenb in zip(range(16),ch_mask):
        df[f'ch{ch}_enable']=chenb
    df['Nch_enabled']=1
    
    dfs.append(df)
        
dft=pd.concat(dfs)

now = datetime.now()
timeStamp = now.strftime("%Y-%m-%d-%H-%M")

filename='clm stability over pow target 16chs to 1ch_sample duration-%i_sample interval-%i_kTpic-%s_kLD-%s-kVOA-%s-alpha-%s_ditherPeriod-%i_%s.csv'\
    %(sample_time,sample_int,str(kp_tpic).replace('.','p'),str(kp_ld).replace('.','p'),str(kp_voa).replace('.','p'),str(ALPHA_I_LD).replace('.','p'),dither_period_s,timeStamp)
dft.to_csv(dataDir+'\\'+filename)   
#%%
df=dft
for bank in [0,1]:
    for ch in range(8):
        dfsub=df[(df[f'ch{8*bank+ch}_enable']==1) & (df.pow_tgt.isin([12,10,8]))]
        fig,ax=plt.subplots(figsize=(8,5))
        sns.lineplot(data=dfsub,x='et_s',y=f'p_mon{ch}',linestyle='-',marker='',hue='pow_tgt',style='Nch_enabled',palette='bright',ax=ax)
        ax.set_title(f'ch{ch}, bank{bank}')
        ax.grid(True)
        ax.set_ylim((6000,14000))
        plt.show()

        fig,ax=plt.subplots(figsize=(8,5))
        sns.lineplot(data=dfsub,x='et_s',y=f'I_VOA{ch}',linestyle='-',marker='',hue='pow_tgt',style='Nch_enabled',palette='bright',ax=ax)
        ax.set_title(f'ch{ch}, bank{bank}')
        ax.grid(True)
        # ax.set_ylim((5000,14000))
        plt.show()

        # fig,ax=plt.subplots(figsize=(5,5))
        # sns.lineplot(data=dfsub,x=f'I_VOA{ch}',y=f'I_LD{ch}',linestyle='-',marker='o',hue='pow_tgt',style='Nch_enabled',palette='bright',ax=ax)
        # ax.set_title(f'ch{ch}, bank{bank}')
        # ax.grid(True)
        # # ax.set_ylim((5000,14000))
        # plt.show()

#%%
'''
text procedure to expand clm power range
'''
dataDir=r'C:\Users\MountainViewLabUser\xingpan\testdata'

sample_time=300
sample_int=2
pause_time=60
dither_period_s=2
kp_tpic=0.3
kp_ld=0.3
kp_voa=0.3
ALPHA_I_LD=0.5
mca.set_clm_control_config(tile,dither_period_s=dither_period_s,kp_tpic=kp_tpic,kp_ld=kp_ld,kp_voa=kp_voa,amp_t_mux=3,ALPHA_I_LD=ALPHA_I_LD)

pow_tgts=[12,9,6,3]+[5,8,11,14]

bTweakTpic=False
dfs=[]
for i,pow_tgt in tqdm(enumerate(pow_tgts),desc='power tgt'):
    if i==0:
        ch_mask=[0]*16
        mca.clm_lane_enable(tile,chEnableMask=ch_mask) 
        mca.initialize_tile(tb,tile=tile)
        mca.optical_power_on(tile)       
        
    mca.set_clm_output_targets(tile,[0]*16,[pow_tgt]*16)
    # 1st all chs on
    ch_mask=[1]*16
    mca.clm_lane_enable(tile,chEnableMask=ch_mask)    
    rcds,cols=mca.get_clm_record([tile], duration_s=sample_time,interval_s=sample_int,bTweakTpic=bTweakTpic)
    
    df=pd.DataFrame(rcds,columns=cols)
    df['pow_tgt']=pow_tgt
    for ch in range(16):
        df[f'ch{ch}_enable']=1
    df['Nch_enabled']=16
    dfs.append(df)
    
    #2nd 1 ch on
    ch_mask=[0]*16
    for ch in range(16):
        ch_mask[ch]=1
    
    mca.clm_lane_enable(tile,chEnableMask=ch_mask)    
    rcds,cols=mca.get_clm_record([tile], duration_s=sample_time,interval_s=sample_int)
    
    df=pd.DataFrame(rcds,columns=cols)
    df['pow_tgt']=pow_tgt
    for ch,chenb in zip(range(16),ch_mask):
        df[f'ch{ch}_enable']=chenb
    df['Nch_enabled']=1
    
    dfs.append(df)
        
dft=pd.concat(dfs)

now = datetime.now()
timeStamp = now.strftime("%Y-%m-%d-%H-%M")

filename='clm stability over pow target 16chs to 1ch_sample duration-%i_sample interval-%i_kTpic-%s_kLD-%s-kVOA-%s-alpha-%s_ditherPeriod-%i_%s.csv'\
    %(sample_time,sample_int,str(kp_tpic).replace('.','p'),str(kp_ld).replace('.','p'),str(kp_voa).replace('.','p'),str(ALPHA_I_LD).replace('.','p'),dither_period_s,timeStamp)
dft.to_csv(dataDir+'\\'+filename)   

#%%
''' setup PTC temp controller'''
ptc1obj = ptc1_module.PTC1()

devs = ptc1_module.PTC1.list_devices()
print(devs)
if len(devs) <= 0:
    print('There is no devices connected')
    exit()
device_info = devs[0]
sn = device_info[0]
print("connect ", sn)

hdl = ptc1obj.open(sn, 115200, 3)
if hdl < 0:
    print("open ", sn, " failed, error code = ")
    exit()
elif ptc1obj.is_open(sn) == 0:
    print("ptc1 is Open failed, error code = ")
    ptc1obj.close()
    exit()
else:
    mode = [0]
    ptc1obj.get_mode(mode)
    if mode[0] != 0:
        print("Invalid mode. Please switch the mode to USB.\n")
    else:
        print('... nice ...')

result = ptc1obj.set_target_temperature(32000)
#%%
perMeter = ozOptics_ER1000.ER1000('com4')
#%%
DMM_VISA = "USB0::0x2A8D::0xB318::MY65070041::INSTR"
PSU_VISA = "USB0::0x2A8D::0x1202::MY61007909::INSTR"

# Set target to 25.0°C (calculated from your previous CSV data)
interval=15
pausetime=10
control = TC.ThermalControlThread(DMM_VISA, PSU_VISA, target_temp=32.0,interval=interval)
kp=0.1
ki=0.001
kd=0.
control.pid.Kp=kp
control.pid.Ki=ki
control.pid.Kd=kd

control.start()
#%%
rcds=[]
t0=time.time()
et=0
while True:
    data=control.get_latest_data()
    et=np.round(time.time()-t0,0)
    rcd=[time.ctime(),et,data['temp'],data['res'],np.round(data['volt'],4)]
    rcds.append(rcd)
    print(rcd[1:])
    time.sleep(10)
#%%
df=pd.DataFrame(rcds,columns=['time','et_s','T','R','V'])    

fig,(ax,ax1)=plt.subplots(2,1,figsize=(6,6))
axt=ax1.twinx()
df.plot('et_s','T',ax=ax)
df.plot('et_s','R',ax=ax1)
df.plot('et_s','V',color='r',ax=axt)
ax.grid(True);ax1.grid(True)
ax.set_ylabel('Temp (C)')
ax1.set_ylabel('R')
axt.set_ylabel('drive (V)', color='r')
plt.show()

fig,(ax,ax1)=plt.subplots(2,1,figsize=(6,6))
axt=ax1.twinx()
df.plot('time','T',ax=ax)
df.plot('time','R',ax=ax1)
df.plot('time','V',color='r',ax=axt)
ax.grid(True);ax1.grid(True)
plt.setp(ax.get_xticklabels(), visible=False)
ax.set_ylabel('Temp (C)')
ax1.set_ylabel('R')
axt.set_ylabel('drive (V)', color='r')
#ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
fig.autofmt_xdate() # Rotates labels for better visibilityplt.show()
plt.show()

#%%
'''
check TC
'''


dwell_time_s=600
sample_int=2

temp_tgts=[32,35,32]
t0=time.time()
rcds=[]
for temp_tgt in temp_tgts:
    _ = ptc1obj.set_target_temperature(int(1000*temp_tgt))
    control.pid.setpoint=temp_tgt
    
    t01=time.time()
    et_s_1=0
    
    while et_s_1<=dwell_time_s:
        
        stats = control.get_latest_data()
        
        if stats["temp"] is not None:
            
            rcd=[]
            t=time.ctime()
            et_s=np.round(time.time()-t0,1)
            et_s_1=np.round(time.time()-t01,1)

            res = stats['res']           
            rcd=[t,et_s,et_s_1,temp_tgt]+ [stats['temp'],stats['res'],stats['volt']]
            rcds.append(rcd)
            print(et_s,rcd[1:])
        time.sleep(sample_int)
        
control.stop()
control.join()

now = datetime.now()
timeStamp = now.strftime("%Y-%m-%d-%H-%M")

cols=['time','et_s','et_s_1','temp_tgt','temp','res','volt']
df=pd.DataFrame(rcds,columns=cols)
df['et_min']=df.et_s/60

filename='temp_controller use dV_behavior_P-%s_I-%s_D-%s_period-%i_%s.csv'\
    %(str(kp).replace('.','p'),str(ki).replace('.','p'),str(kd).replace('.','p'),interval,timeStamp)
df.to_csv(dataDir+'\\'+filename)

fig,ax=plt.subplots(figsize=(8,5))
axt=ax.twinx()
df.plot('et_min',['temp_tgt','temp'],ax=ax)
df.plot('et_min','volt',color='r',ax=axt)
ax.set_ylabel('temp(C)')
axt.set_ylabel('drive (V)', color='r')
ax.legend('center left')
axt.legend('center right')
ax.grid(True)
ax.set_title(f'Kp {kp:.3f}, Ki {ki:.3f}, Kd {kd:.3f}; update period {interval}')
plt.show()
#%%
'''
PER over temp and over ch 
'''

temp_tgts=np.concatenate((np.linspace(32,35,13),np.linspace(35-0.125,32-0.125,13)))

cols=['time','et_s','et_s_1','et_s_2','temp_tgt','temp','res','volt','ch','per_dB','theta','pow']

rcds=[]

interval=15
control = TC.ThermalControlThread(DMM_VISA, PSU_VISA, target_temp=32.0,interval=interval)
kp=0.1
ki=0.001
kd=0.
control.pid.Kp=kp
control.pid.Ki=ki
control.pid.Kd=kd

control.start()
#%%
time.sleep(3600) # to make sure the temp stabilizes before data collection

bank=1
chs=[0,1,2,3,4,5,6,7]
time_tc_s=600 # time for TC to stabilize
time_ch_s=30 # time for channel to stabilize
time_sample_s=120 # time to collect PER at each ch
int_sample_s=10 # time interval between PER 

t0=time.time()
for temp_tgt in tqdm(temp_tgts,desc='temp control'):
    _ = ptc1obj.set_target_temperature(int(1000*temp_tgt))
    control.pid.setpoint=temp_tgt
    
    t01=time.time()
    et_s_1=0
    
    while et_s_1<=time_tc_s:
        # wait a while for temp to stabilize
        t=time.ctime()
        et_s=np.round(time.time()-t0,1)
        et_s_1=np.round(time.time()-t01,1)
        time.sleep(1)
    
    for ch in tqdm(chs,desc='channel'):
        t02=time.time()
        et_s_2=0 
        
        chEnMask=[0]*16
        chEnMask[bank*8+ch]=1
        mca.clm_lane_enable(tile,chEnableMask=chEnMask)
        time.sleep(time_ch_s)
        
        while et_s_2<=time_sample_s:
    
            stats = control.get_latest_data()
            
            if stats["temp"] is not None:
                
                rcd=[]
                t=time.ctime()
                et_s=np.round(time.time()-t0,1)
                et_s_1=np.round(time.time()-t01,1)
                et_s_2=np.round(time.time()-t02,1)
            
                permeasurements=perMeter.measurePER()
                res = stats['res']
                
                for mea in permeasurements:
                    rcd=[t,et_s,et_s_1,et_s_2,temp_tgt]+ [stats['temp'],stats['res'],stats['volt']]+\
                        [ch]+mea
                    rcds.append(rcd)
                    print(et_s,rcd[5:])
            else:
                pass
        
            time.sleep(int_sample_s) 
        
control.stop()
control.join()

now = datetime.now()
timeStamp = now.strftime("%Y-%m-%d-%H-%M")

df=pd.DataFrame(rcds,columns=cols)
df['et_min']=df.et_s/60

filename='per over bank%i chs over temp_P-%s_I-%s_D-%s_period-%i_%s.csv'\
    %(bank,str(kp).replace('.','p'),str(ki).replace('.','p'),str(kd).replace('.','p'),interval,timeStamp)
df.to_csv(dataDir+'\\'+filename)   

fig,ax=plt.subplots(3,1,figsize=(8,10))
sns.lineplot(data=df,x='et_min',y='temp',hue='ch',palette='bright',ax=ax[0])
sns.lineplot(data=df,x='et_min',y='per_dB',linestyle='-',marker='',hue='ch',palette='bright',ax=ax[1])
sns.lineplot(data=df,x='et_min',y='theta',hue='ch',palette='bright',ax=ax[2])
ax[2].set_xlabel('et_min')
[ax[i].grid(True) for i in range(3)]
#ax[0].set_ylim((23,26.5))
plt.show()

fig,ax=plt.subplots(figsize=(6,6))
sns.scatterplot(data=df,x='theta',y='per_dB',hue='ch',linestyle='',marker='o',palette='bright',ax=ax)
ax.grid(True)
#ax[0].set_ylim((23,26.5))
plt.show()

#%%
''' per over chs over freq offset
    assume PM fiber is temp stabilized
'''
bank=0
BANK=mca.ClmTile.Bank.BANK_A

chs=[0,1,2,3,4,5,6,7]
frqoffsets=[-20,-15,-10,-5,0,5,10,15,20]

time_ch_s=30 # time for channel to stabilize
time_sample_s=120 # time to collect PER at each ch
int_sample_s=10 # time interval between PER 

cols=['time','et_s','et_s_1','temp_tgt','temp','res','volt','ch','freq_offset','per_dB','theta','pow']

rcds=[]

t0=time.time()
et_s=0

temp_tgt=control.pid.setpoint

for foffset in tqdm(frqoffsets,desc='freq_offset'):
    for ch in tqdm(chs,desc='ch'):
        chEnMask=[0]*16
        chEnMask[bank*8+ch]=1
        fos=[foffset]*8        
        tile.write_lane_frequency_setpoints(BANK,fos)
        mca.clm_lane_enable(tile,chEnableMask=chEnMask)
        
        time.sleep(time_ch_s)
        
        t01=time.time()
        et_s_1=0
        
        while et_s_1<=time_sample_s:
            # wait a while for temp to stabilize
            t=time.ctime()
            et_s=np.round(time.time()-t0,1)
            et_s_1=np.round(time.time()-t01,1)
        
            stats = control.get_latest_data()
            
            if stats["temp"] is not None:
                
                rcd=[]
                t=time.ctime()
                et_s=np.round(time.time()-t0,1)
                et_s_1=np.round(time.time()-t01,1)
            
                permeasurements=perMeter.measurePER()
                res = stats['res']
                
                for mea in permeasurements:
                    rcd=[t,et_s,et_s_1,temp_tgt]+ [stats['temp'],stats['res'],stats['volt']]+\
                        [ch,foffset]+mea
                    rcds.append(rcd)
                    print(et_s,rcd[4:])
            else:
                pass
        
            time.sleep(int_sample_s) 
        
now = datetime.now()
timeStamp = now.strftime("%Y-%m-%d-%H-%M")

df=pd.DataFrame(rcds,columns=cols)
df['et_min']=df.et_s/60

filename='per over freq offset over chs__%s.csv'\
    %(timeStamp)
df.to_csv(dataDir+'\\'+filename)   

fig,ax=plt.subplots(2,1,figsize=(8,8))
sns.lineplot(data=df,x='freq_offset',y='per_dB',linestyle='-',marker='o',hue='ch',palette='bright',ax=ax[0])
sns.lineplot(data=df,x='freq_offset',y='theta',hue='ch',marker='o',palette='bright',ax=ax[1])
ax[1].set_xlabel('freq_offset')
[ax[i].grid(True) for i in range(2)]
#ax[0].set_ylim((23,26.5))
plt.show()

fig,ax=plt.subplots(figsize=(6,6))
sns.scatterplot(data=df,x='theta',y='per_dB',hue='ch',linestyle='',marker='o',palette='bright',ax=ax)
ax.grid(True)
#ax[0].set_ylim((23,26.5))
ax.set_title('(PER, theta) over [-20,20] GHz frequency change')
plt.show()

df['et_h']=df.et_s/3600

fig,ax=plt.subplots(figsize=(6,6))
sns.scatterplot(data=df,x='et_h',y='temp',hue='ch',linestyle=':',marker='o',palette='bright',ax=ax)
ax.grid(True)
ax.set_ylim((31,33))
ax.set_title('PM fiber temp over time')
plt.show()

#%%
''' passive per measurement till interrupt'''

cols=['time','et_s','per_dB','theta','pow','res','temp']
rcds=[]

try:
    t0=time.time()
    while True:
        # Instead of talking to the DMM, we ask the thread for its latest cached data
        stats = control.get_latest_data()
        
        if stats["temp"] is not None:
            
            rcd=[]
            t=time.ctime()
            et_s=np.round(time.time()-t0,1)
        
            permeasurements=perMeter.measurePER()
            res = stats['res']
            
            for mea in permeasurements:
                rcd=[t,et_s]+mea + [stats['res'],stats['temp']]
                rcds.append(rcd)
                print(et_s,rcd[1:])
        else:
            pass
        
        # Main program does other things here
        time.sleep(pausetime) 
        
except KeyboardInterrupt:
    control.stop()  
    control.join()
#%%     
#df=pd.DataFrame(rcds,columns=cols)

dataDir=r'C:\Users\MountainViewLabUser\xingpan\testdata'

#df.to_csv(dataDir+'\\ch5 overnight per and temp_with fiber and clm case TC at 32C_4-30-2026.csv')

df=pd.read_csv(dataDir+'\\ch5 overnight per and temp_with fiber and clm case TC at 32C_4-30-2026.csv')
df['et_h']=df.et_s/3600
#df=pd.read_csv(dataDir+'\\ch5 overnight per and temp_with foam insulation_4-27-2026.csv')
#%%
fig,ax=plt.subplots(3,1,figsize=(5,5))
df.plot('et_h','temp',marker='',ax=ax[0])
df.plot('et_h','per_dB',marker='',ax=ax[1])
df.plot('et_h','theta',marker='',ax=ax[2])
ax[2].set_xlabel('et_s')
[ax[i].grid(True) for i in range(3)]
#ax[0].set_ylim((23,26.5))
plt.show()
#%%
fig,ax=plt.subplots(2,1,figsize=(5,5))
df.plot('temp','per_dB',linestyle=':',marker='.',ax=ax[0])
df.plot('temp','theta',linestyle=':',marker='.',ax=ax[1])
[ax[i].grid(True) for i in range(2)]
plt.show()
#%%
df=df[df.et_s>3600]
fig,ax=plt.subplots(figsize=(5,5))
df.plot('theta','per_dB',linestyle='',marker='.',ax=ax)
ax.grid(True)
ax.set_xlim((-6,8))
ax.set_ylim((15,35))
plt.show()
#%%
dt=np.zeros(len(df))
dt[1:]=np.array(np.diff(df.et_s))
dT=np.zeros(len(df))
dT[1:]=np.array(np.diff(df.temp))
df['dT_C']=dT
df['dt']=dt
df=df[df.dt>0]

plt.hist(df.dT_C, bins=30, color='skyblue', edgecolor='black')
plt.title('fiber n temp sensor in foam, temp change over 15s')
plt.xlabel('dT (C)')
plt.ylabel('Frequency')
plt.grid(True)
plt.xlim((-0.2,0.2))
plt.show()

      