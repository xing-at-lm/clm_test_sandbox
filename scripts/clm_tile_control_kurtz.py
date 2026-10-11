# -*- coding: utf-8 -*-
"""
Created on Sat May 31 13:22:37 2025

@author: XingPan

"""
from pathlib import Path
import sys
import numpy as np
import time
from datetime import datetime
import matplotlib.pyplot as plt
import copy

from cmis_host.platform.clm import ClmTile
from cmis_host.protocol.oc103 import TargetDac
# Optional alias if you still want the old name:
ClmBank = ClmTile.Bank

#from cmis_lm import Lt8722Module,CmisCommand,Lt8722Register
sys.path.append(r'C:\Users\MountainViewLabUser\gitrepos\cmis-host-controller\src\cmis_host\protocol')
from oc103 import Lt8722Module,Lt8722Register


wavelength_grid_nm = np.array([
  1301.47, 1302.60, 1303.73, 1304.87,
  1306.01, 1307.14, 1308.28, 1309.43,
  1310.57, 1311.72, 1312.87, 1314.02,
  1315.17, 1316.33, 1317.48, 1318.64
  ])

wavelength_grid_nm_A=wavelength_grid_nm[0::2]
wavelength_grid_nm_B=wavelength_grid_nm[1::2]

PID_PERIOD_MS=10
OUTPUT_MIN=-1.5
OUTPUT_MAX=1.5

clip_I_LD_mA=6
clip_I_VOA_mA=4

dictConfig_default={'temp_mux': 50.0,
 'temp_pic': 46.57,
 'bias_currents': np.array([150]*16),
 'voa_currents': np.array([10]*16),
 'mux_mpd_tap_ratios': np.array([0.05]*16),
 'pic_mpd_tap_ratios': np.array([0.04]*16),
 'ditherAmp':3,
 'ditherElement':'T_MUX',
 'pout_mon_weight':0.5}

def bathtub_derivative(x, x_min, x_max, delta_x, p_left=3.0, p_right=3.0):
    """
    Computes the continuous derivative of an asymmetric bathtub function.
    p_left is increased to 3.0 to ensure a smooth, rounded transition into the zero-region.
    """
    x = np.asarray(x, dtype=float)
    dy = np.zeros_like(x)
    
    left_flat_start = x_min + delta_x
    right_flat_end = x_max - delta_x
    
    # 1. Left Side: Smooth quadratic drop (matches the red curve)
    left_mask = (x < left_flat_start)
    dy[left_mask] = -(p_left / delta_x) * ((left_flat_start - x[left_mask]) / delta_x) ** (p_left - 1)
    
    # 2. Right Transition: Positive superlinear curve
    right_mask = (x > right_flat_end) # & (x <= x_max)
    dy[right_mask] = (p_right / delta_x) * ((x[right_mask] - right_flat_end) / delta_x) ** (p_right - 1)
    return dy

# class clmTile_control:
#     def __init__(self,tileObj,tileSN='-01',sample_period=6,freqOffset_GHz_tgt=[0]*16,pout_mW_tgt=[5]*16,freqOffset_BW=3,
#                  pout_ratio_BW=0.3,actMask=np.ones(33),costMask=np.ones(32),P_of_PID=np.ones(33)*0.5,dictConfig=None):
#         self.clmTile=tileObj
#         self.sn=tileSN
#         if dictConfig==None:
#             dictConfig=copy.deepcopy(dictConfig_default)
            
#         self.T_PIC=dictConfig['temp_pic']
#         self.T_MUX=dictConfig['temp_mux']
#         self.T_PIC_mon=9999
#         self.T_MUX_mon=9999
#         self.I_LD_mA_Ary=dictConfig['bias_currents']
#         self.I_VOA_mA_Ary=dictConfig['voa_currents']
#         self.attn_VOA_dB_Ary=self.getVOA_attenuation_dB(self.I_VOA_mA_Ary)
#         self.MPD_PIC_uA_Ary=np.zeros(16)
#         self.MPD_MUX_uA_Ary=np.zeros(16)
#         self.connected=False
#         self.FW_version='na'
#         self.currentOpticalState=np.zeros(32)
#         self.currentControlVector=np.zeros(33)
        
#         self.tap_ratio_pic=dictConfig['pic_mpd_tap_ratios']
#         self.tap_ratio_mux=dictConfig['mux_mpd_tap_ratios']      
#         self.ditherAmp=dictConfig['ditherAmp'] 
#         self.ditherElement=dictConfig['ditherElement'] # dither T_MUX, T_PIC or LD bias 
#         self.pout_mon_weight=dictConfig['pout_mon_weight']
#         self.norm_PIC_MUX=self.tap_ratio_pic/self.tap_ratio_mux
#         if self.ditherElement=='T_MUX':
#             self.dFreq_dRdT=1/(-0.000571) # 
#         if self.ditherElement=='T_PIC':
#             self.dFreq_dRdT=244 #             
#         self.dRdT=np.zeros(16)
#         self.freq_offset_GHz=np.zeros(16)
#         self.Pout_mW=np.zeros(16)
#         self.sample_period=sample_period
        
#         self.freqOffset_GHz_tgt=freqOffset_GHz_tgt # target frequency offset, relative to MUX peak frequency
#         self.pout_mW_tgt=pout_mW_tgt # target optical power target at output fiber 
#         self.freqOffset_BW=freqOffset_BW # weight of frequency error in feedback control
#         self.pout_ratio_BW=pout_ratio_BW # weight of power error in feedback control
#         self.errorVector_norm=np.zeros(32)
#         self.costFunc=None
#         self.dCFdx=np.zeros(33) # 0:16: LD current; 16:32: VOA current; 32: T_PIC derivation of the cost function w respect to control variables/vector
#         self.actMask=np.array(actMask) # actMask controls which 'knob' are controlled. [0:16]: LD bias; [16:32]: VOAs, [32]: T_PIC
#         '''
#         # costMask: control what metric are included in the cost function calculation. (The intention is that) it has no impact on control but
#             # rather make the cost function numerically easy to interpret. for example:
#             # if we are interested in power equilization by using VOAs, we can set costMask[0:16]=0, so frequency errors will not contribute to cost function
#             # this we, when the channel is power equalized, the cost function should give us a numerical value close to 0 ('easy to interpret')
#             # converse, if we are only interested in channel frequency 'MUX-lock', we can set costMask[16:32]=0, to ignore contributions from power error
#      '''
#         self.costMask=costMask 
#         self.controlVector=np.zeros(33)
#         self.controlVector[0:16]=self.I_LD_mA_Ary
#         self.controlVector[16:32]=self.I_VOA_mA_Ary
#         self.controlVector[32]=self.T_PIC
#         self.P_of_PID=P_of_PID # as the name suggests, 'P' term of the PID loop
        
    
#     def configure_pid(self,TEC_SELECT=Lt8722Module.PIC,SETPOINT=40,KP=-1,KI=-0.15,KD=0):
#         """
#         Applies the specified configuration to the PID under test
#         """
#         self.clmTile.pid_enable(TEC_SELECT,False)
#         self.clmTile.pid_config_constants(TEC_SELECT, KP, KI, KD, PID_PERIOD_MS)
#         self.clmTile.pid_config_limits(TEC_SELECT, OUTPUT_MIN, OUTPUT_MAX)
#         self.clmTile.pid_config_setpoint(TEC_SELECT, SETPOINT)
#         self.clmTile.pid_enable(TEC_SELECT,True)
#         return 0
        
#     def setTECTemps(self,T_PIC,T_MUX):
#         T_PIC=np.clip(T_PIC,30,60) # to be safe
#         T_MUX=np.clip(T_MUX,15,65) # to be safe
        
#         self.clmTile.pid_config_setpoint(Lt8722Module.PIC, T_PIC)
#         time.sleep(0.3)
#         self.clmTile.pid_config_setpoint(Lt8722Module.MUX, T_MUX)
        
#         self.T_PIC=T_PIC
#         self.T_MUX=T_MUX
        
#         return 0

#     def readTECTempSersors(self):
#         T_PIC_mon,T_MUX_mon=self.clmTile.temp_read()[0:2]
#         self.T_PIC_mon=np.round(T_PIC_mon,2)
#         self.T_MUX_mon=np.round(T_MUX_mon,2)
#         return self.T_PIC_mon,self.T_MUX_mon
    
#     def set_LD_current(self,ch,I_mA):
#         I_mA=np.clip(I_mA,0,185)        
#         if ch<8:         
#             self.clmTile.laser_set_lane(ClmBank.BANK_A, ch,I_mA)
#         else:
#             self.clmTile.laser_set_lane(ClmBank.BANK_B, ch-8,I_mA)
#         self.I_LD_mA_Ary[ch]=I_mA

#     def set_VOA_current(self,ch,I_mA):
#         I_mA=np.clip(I_mA,0,25)        
#         if ch<8:         
#             self.clmTile.voa_set_lane(ClmBank.BANK_A, ch,I_mA)
#         else:
#             self.clmTile.voa_set_lane(ClmBank.BANK_B, ch-8,I_mA)
#         self.I_VOA_mA_Ary[ch]=I_mA
#         self.attn_VOA_dB_Ary[ch]=self.getVOA_attenuation_dB(I_mA)
            
#     def readMPDcurrent(self):
#         mpd_pic=np.zeros(16)
#         mpd_mux=np.zeros(16)
#         mpd_pic[0:8]=self.clmTile.mpd_read_pic(ClmBank.BANK_A)
#         mpd_pic[8:]=self.clmTile.mpd_read_pic(ClmBank.BANK_B)
        
#         mpd_mux[0:8]=self.clmTile.mpd_read_mux(ClmBank.BANK_A)
#         mpd_mux[8:]=self.clmTile.mpd_read_mux(ClmBank.BANK_B)
        
#         self.MPD_PIC_uA_Ary= mpd_pic       
#         self.MPD_MUX_uA_Ary= mpd_mux     
#         return mpd_pic,mpd_mux
    
#     def getVOA_attenuation_dB(self,I_mA):
#         '''
#         emperical formula for VOA
#         '''
#         return np.round(10*np.log10(1+0.061*I_mA),2)
    
#     def updateCurrentOpticalState(self):
#         '''
#         Returns
#         clm opticalState [0:32]: [0:16] optical frequency in GHz (relative to MUX peak frequency); [16:32] optical power in mW
#         -------
#         TYPE
#             DESCRIPTION.

#         '''
#         if self.ditherElement=='LD':
#             print('dithering %s not implemented yet.'%self.ditherElement)
#             exit
#         elif self.ditherElement=='T_PIC':
#             '''
#                 XP note: 8/9/2025: instead of dither, we sweep T_PIC to average out MPD ripple effect
#             '''
#             Nsteps=11
#             T_PIC_lo=self.T_PIC - self.ditherAmp
#             T_PIC_hi=self.T_PIC + self.ditherAmp
#             T_PICs=np.linspace(T_PIC_lo,T_PIC_hi,Nsteps)

#             mpd_pics=np.zeros((16,Nsteps))
#             mpd_muxs=np.zeros((16,Nsteps))
#             rs= np.zeros((16,Nsteps))
            
#             for j, t_pic in enumerate(T_PICs):
#                 self.clmTile.pid_config_setpoint(Lt8722Module.PIC,t_pic)
#                 if j==0:
#                     time.sleep(self.sample_period*0.5)
#                 else:
#                     time.sleep(self.sample_period/(Nsteps))
                    
#                 mpd_pics[:,j],mpd_muxs[:,j]=self.readMPDcurrent()
#                 rs[:,j]=np.clip(mpd_muxs[:,j],1,1e6)/np.clip(mpd_pics[:,j],1,1e6)
           
#             mpd_pic=mpd_pics.mean(axis=1)# take mean
#             mpd_mux=mpd_muxs.mean(axis=1) # take mean
            
#             self.MPD_PIC_uA_Ary=mpd_pic
#             self.MPD_MUX_uA_Ary=mpd_mux
            
#             dRdT_PICs=np.zeros(16)
#             for i in range(16):
#                 x=T_PICs
#                 y=rs[i,:]
#                 pFit=np.polyfit(x,y,1)
#                 dRdT_PICs[i]=pFit[0]            
                                      
#             # fig, ax=plt.subplots(1,2,figsize=(8,4))
#             # ax[0].plot(T_PICs,rs[0:8,:])
#             # ax[1].plot(T_PICs,rs[8:16,:])
#             # ax[0].grid(True);ax[1].grid(True)
#             # plt.show()
            
#             dRdT=dRdT_PICs*self.norm_PIC_MUX # dRdT is the derivative with respect to temperature. normalize such that it is independent of dither amp, or when we have a bright or dark MUX MPD
#             self.dRdT=dRdT
#             self.freq_offset_GHz=dRdT*self.dFreq_dRdT # frequency offset estimate 
#             self.Pout_mW=(self.pout_mon_weight*mpd_pic/self.tap_ratio_pic+\
#                         (1 - self.pout_mon_weight)*mpd_mux/self.tap_ratio_mux)*1e-3 # self.pout_mon_weight: do we trust mpd_PIC or mpd_mux? ideally, they should produce the same answer
                        
#             self.currentOpticalState[0:16]=np.round(self.freq_offset_GHz,2)
#             self.currentOpticalState[16:]=np.round(self.Pout_mW,2)
                        
#             # set temp back
#             self.clmTile.pid_config_setpoint(Lt8722Module.PIC, self.T_PIC)
            
#             return self.currentOpticalState
            
#         elif self.ditherElement=='T_MUX':
            
#             T_MUX_lo=self.T_MUX - self.ditherAmp
#             T_MUX_hi=self.T_MUX + self.ditherAmp
            
#             self.clmTile.pid_config_setpoint(Lt8722Module.MUX,T_MUX_lo)
#             time.sleep(self.sample_period*0.5)
#             mpd_pic_lo,mpd_mux_lo=self.readMPDcurrent()

#             self.clmTile.pid_config_setpoint(Lt8722Module.MUX,T_MUX_hi)
#             time.sleep(self.sample_period*0.5)
#             mpd_pic_hi,mpd_mux_hi=self.readMPDcurrent()
            
#             mpd_pic=0.5*(mpd_pic_hi+mpd_pic_lo) # take mean
#             mpd_mux=0.5*(mpd_mux_lo+mpd_mux_hi) # take mean
            
#             self.MPD_PIC_uA_Ary=mpd_pic
#             self.MPD_MUX_uA_Ary=mpd_mux
            
#             # try to avoid exceptions, which can happen when a channel has low/no power
#             dR=mpd_mux_hi/np.clip(mpd_pic_hi,1,1000000) - mpd_mux_lo/np.clip(mpd_pic_lo,1,1000000)
#             dR=np.where((mpd_pic_hi!=0) &(mpd_pic_lo!=0),dR,np.zeros(len(dR)))
            
#             dRdT=dR/self.ditherAmp/2*self.norm_PIC_MUX # dRdT is the derivative with respect to temperature. normalize such that it is independent of dither amp, or when we have a bright or dark MUX MPD
#             self.dRdT=dRdT
#             self.freq_offset_GHz=dRdT*self.dFreq_dRdT # frequency offset estimate 
#             self.Pout_mW=(self.pout_mon_weight*mpd_pic/self.tap_ratio_pic+\
#                         (1 - self.pout_mon_weight)*mpd_mux/self.tap_ratio_mux)*1e-3 # self.pout_mon_weight: do we trust mpd_PIC or mpd_mux? ideally, they should produce the same answer
                        
#             self.currentOpticalState[0:16]=np.round(self.freq_offset_GHz,2)
#             self.currentOpticalState[16:]=np.round(self.Pout_mW,2)
                        
#             # set temp back
#             self.T_MUX=0.5*(T_MUX_lo+T_MUX_hi) # 
#             self.clmTile.pid_config_setpoint(Lt8722Module.MUX, self.T_MUX)
            
#             return self.currentOpticalState

#     def updateCostFunction(self):
#         state_optical=self.updateCurrentOpticalState()
#         for i,ic in enumerate(self.costMask):
#             if i<16:
#                 if ic==np.True_:
#                     self.errorVector_norm[i]=(state_optical[i]-self.freqOffset_GHz_tgt[i])/self.freqOffset_BW
#                 else:
#                     self.errorVector_norm[i]=0
#             else:
#                 if ic==np.True_:
#                     self.errorVector_norm[i]=10*np.log10(state_optical[i]/self.pout_mW_tgt[i-16])/self.pout_ratio_BW
#                 else:
#                     self.errorVector_norm[i]=0
#         self.errorVector_norm=np.round(self.errorVector_norm,4)*self.costMask
#         self.errorVector_norm=np.nan_to_num(self.errorVector_norm,nan=0)
#         cFunc=np.sum((self.errorVector_norm)**2)
#         self.costFunc=np.round(cFunc,3)
        
#         self.dCFdx[32]=-16*2*np.sum(self.errorVector_norm[0:16]) # x=T_PIC, impact only freqeuncy, so assumed 
#         self.dCFdx[0:16]=-16/20*2*self.errorVector_norm[0:16]+2*self.errorVector_norm[16:16+16]*0.1/self.pout_mW_tgt # x= LD bias. X-talk ignored, slope effeciency 0f 0.1 mW/mA assumed
#         self.dCFdx[16:32]=2*self.errorVector_norm[16:16+16]*(-0.061/(1+0.061*self.I_LD_mA_Ary)**2) # x= VOA bias
    
#     def computeControlAdjustment(self):
#         '''
#         Note: this order is 'influenced' by clm tile test flow  
#         1. compute T_PIC adjustment
#         2. LD bias
#         3 VOA bias
#         '''
#         # since we know T_PIC impact all channels the same, we use it to adjust mean frequency error
#         if np.sum(self.costMask[0:16])>0:
#             meanFreqErr_norm=np.sum(self.errorVector_norm[0:16])/np.sum(self.costMask[0:16])
#         else:
#             meanFreqErr_norm=0 # if the user do not care frequency error at the moment, set it to zero for cost function calculation
            
#         deltaT_PIC=-meanFreqErr_norm*self.freqOffset_BW/(-16)*self.actMask[32]*self.P_of_PID[32]
#         deltaT_PIC=np.clip(deltaT_PIC,-0.3,0.3) # clip T_PIC step size. 
        
#         # 7/31/2025 XP: (-16)*deltaT_PIC term remove the frequency error that WILL be removed by T_PIC adjustment.
#         deltaI_LD=-(self.errorVector_norm[0:16]*self.freqOffset_BW - (-16)*deltaT_PIC)/(-16/20)*self.P_of_PID[0:16]*self.actMask[0:16]

#         powerratioErr=10**(self.errorVector_norm[16:32]*self.pout_ratio_BW/10)
        
#         #9/9 added: also use LD current to compensate for power, assume 0.15 slope effeciency
#         alpha_LD=0.5 # portion of power correction using LD
#         deltaI_LD=deltaI_LD-alpha_LD*(powerratioErr-1)*self.pout_mW_tgt/0.15*self.P_of_PID[0:16]*self.actMask[0:16]
#         '''
#         the above line can also simply written as:
#             delta_I_LD=deltaI_LD -(self.pout_mW/self.pout_mW_tgt -1)*self.pout_mW_tgt/0.15*self.P_of_PID[0:16]*self.actMask[0:16]
#         '''
#         deltaI_LD=np.clip(deltaI_LD,-2.5,2.5) # clip LD bias step size.
        
#         deltaI_VOA=(1-alpha_LD)*(powerratioErr-1)*(1+0.061*self.I_VOA_mA_Ary)/0.061*self.actMask[16:32]*self.P_of_PID[16:32]
#         deltaI_VOA=np.clip(deltaI_VOA,-1,1)  # clip LD bias step size.
        
#         deltaControlVector=np.zeros(33)
#         deltaControlVector[0:16]=deltaI_LD
#         deltaControlVector[16:32]=deltaI_VOA
#         deltaControlVector[32]=deltaT_PIC
#         deltaControlVector=np.round(deltaControlVector,3)
#         deltaControlVector=deltaControlVector*np.array(self.actMask)
        
#         deltaControlVector=np.nan_to_num(deltaControlVector,nan=0)
        
#         self.controlVector=self.controlVector+deltaControlVector
#         self.controlVector[0:16]=np.clip(self.controlVector[0:16],0,185)
#         self.controlVector[16:32]=np.clip(self.controlVector[16:32],0,25)
#         self.controlVector[32]=np.clip(self.controlVector[32],30,65)
#         return deltaControlVector
    
#     def applyControlVector(self):
#         self.clmTile.pid_config_setpoint(Lt8722Module.PIC,self.controlVector[32])
#         for ch in range(16):
#             self.set_LD_current(ch,self.controlVector[ch])
#             time.sleep(0.05)
#             self.set_VOA_current(ch,self.controlVector[ch+16])
        
#         self.I_LD_mA_Ary=self.controlVector[0:16]
#         self.I_VOA_mA_Ary=self.controlVector[16:32]
#         self.T_PIC=self.controlVector[32]
   

class clmTile_control_new:
    def __init__(self,tileObj,tileSN='-01',sample_period=6,freqOffset_GHz_tgt=[0]*16,pout_mW_tgt=[5]*16,freqOffset_BW=3,
                 pout_ratio_BW=0.3,actMask=np.ones(33),costMask=np.ones(32+33),P_of_PID=np.ones(33)*0.5,I_LD_corner_BW=5,I_VOA_corner_BW=2,
                 T_PIC_corner_BW=2,dictConfig=None):
        self.clmTile=tileObj
        self.sn=tileSN
        if dictConfig==None:
            dictConfig=copy.deepcopy(dictConfig_default)
            
        self.T_PIC=dictConfig['temp_pic']
        self.T_MUX=dictConfig['temp_mux']
        self.T_PIC_mon=9999
        self.T_MUX_mon=9999
        self.I_LD_mA_Ary=dictConfig['bias_currents']
        self.I_VOA_mA_Ary=dictConfig['voa_currents']
        self.attn_VOA_dB_Ary=self.getVOA_attenuation_dB(self.I_VOA_mA_Ary)
        self.MPD_PIC_uA_Ary=np.zeros(16)
        self.MPD_MUX_uA_Ary=np.zeros(16)
        self.connected=False
        self.FW_version='na'
        self.currentState=np.zeros(32+33)
        self.currentControlVector=np.zeros(33)
        
        self.tap_ratio_pic=dictConfig['pic_mpd_tap_ratios']
        self.tap_ratio_mux=dictConfig['mux_mpd_tap_ratios']      
        self.ditherAmp=dictConfig['ditherAmp'] 
        self.ditherElement=dictConfig['ditherElement'] # dither T_MUX, T_PIC or LD bias 
        self.pout_mon_weight=dictConfig['pout_mon_weight']
        self.norm_PIC_MUX=self.tap_ratio_pic/self.tap_ratio_mux
        if self.ditherElement=='T_MUX':
            self.dFreq_dRdT=1/(-0.000571) # 
        if self.ditherElement=='T_PIC':
            self.dFreq_dRdT=244 #             
        self.dRdT=np.zeros(16)
        self.freq_offset_GHz=np.zeros(16)
        self.Pout_mW=np.zeros(16)
        self.sample_period=sample_period
        
        self.freqOffset_GHz_tgt=freqOffset_GHz_tgt # target frequency offset, relative to MUX peak frequency
        self.pout_mW_tgt=pout_mW_tgt # target optical power target at output fiber 
        self.freqOffset_BW=freqOffset_BW # weight of frequency error in feedback control
        self.pout_ratio_BW=pout_ratio_BW # weight of power error in feedback control
        self.errorVector_norm=np.zeros(32+33)
        self.costFunc=None
        self.dCFdx=np.zeros(33) # 0:16: LD current; 16:32: VOA current; 32: T_PIC derivation of the cost function w respect to control variables/vector
        self.actMask=np.array(actMask) # actMask controls which 'knob' are controlled. [0:16]: LD bias; [16:32]: VOAs, [32]: T_PIC
        '''
        # costMask: control what metric are included in the cost function calculation. (The intention is that) it has no impact on control but
        rather make the cost function numerically easy to interpret. for example:
            if we are interested in power equilization by using VOAs, we can set costMask[0:16]=0, so frequency errors will not contribute to cost function
            this we, when the channel is power equalized, the cost function should give us a numerical value close to 0 ('easy to interpret')
            converse, if we are only interested in channel frequency 'MUX-lock', we can set costMask[16:32]=0, to ignore contributions from power error
        '''
        self.costMask=costMask 
        self.controlVector=np.zeros(33)
        self.controlVector[0:16]=self.I_LD_mA_Ary
        self.controlVector[16:32]=self.I_VOA_mA_Ary
        self.controlVector[32]=self.T_PIC
        self.P_of_PID=P_of_PID # as the name suggests, 'P' term of the PID loop
        self.I_LD_corner_BW=I_LD_corner_BW
        self.I_VOA_corner_BW=I_VOA_corner_BW
        self.T_PIC_corner_BW=T_PIC_corner_BW    
    
    def configure_pid(self,TEC_SELECT=Lt8722Module.PIC,SETPOINT=40,KP=-1,KI=-0.15,KD=0):
        """
        Applies the specified configuration to the PID under test
        """
        self.clmTile.pid_enable(TEC_SELECT,False)
        self.clmTile.pid_config_constants(TEC_SELECT, KP, KI, KD, PID_PERIOD_MS)
        self.clmTile.pid_config_limits(TEC_SELECT, OUTPUT_MIN, OUTPUT_MAX)
        self.clmTile.pid_config_setpoint(TEC_SELECT, SETPOINT)
        self.clmTile.pid_enable(TEC_SELECT,True)
        return 0
        
    def setTECTemps(self,T_PIC,T_MUX):
        T_PIC=np.clip(T_PIC,15,60) # to be safe
        T_MUX=np.clip(T_MUX,15,65) # to be safe
        
        self.clmTile.pid_config_setpoint(Lt8722Module.PIC, T_PIC)
        time.sleep(0.3)
        self.clmTile.pid_config_setpoint(Lt8722Module.MUX, T_MUX)
        
        self.T_PIC=T_PIC
        self.T_MUX=T_MUX
        
        return 0

    def readTECTempSersors(self):
        T_PIC_mon,T_MUX_mon=self.clmTile.temp_read()[0:2]
        self.T_PIC_mon=np.round(T_PIC_mon,2)
        self.T_MUX_mon=np.round(T_MUX_mon,2)
        return self.T_PIC_mon,self.T_MUX_mon
    
    def set_LD_current(self,ch,I_mA):
        I_mA=np.clip(I_mA,0,185)        
        if ch<8:         
            self.clmTile.laser_set_lane(ClmBank.BANK_A, ch,I_mA)
        else:
            self.clmTile.laser_set_lane(ClmBank.BANK_B, ch-8,I_mA)
        self.I_LD_mA_Ary[ch]=I_mA

    def set_VOA_current(self,ch,I_mA):
        I_mA=np.clip(I_mA,0,25)        
        if ch<8:         
            self.clmTile.voa_set_lane(ClmBank.BANK_A, ch,I_mA)
        else:
            self.clmTile.voa_set_lane(ClmBank.BANK_B, ch-8,I_mA)
        self.I_VOA_mA_Ary[ch]=I_mA
        self.attn_VOA_dB_Ary[ch]=self.getVOA_attenuation_dB(I_mA)
            
    def readMPDcurrent(self):
        mpd_pic=np.zeros(16)
        mpd_mux=np.zeros(16)
        mpd_pic[0:8]=self.clmTile.mpd_read_pic(ClmBank.BANK_A)
        mpd_pic[8:]=self.clmTile.mpd_read_pic(ClmBank.BANK_B)
        
        mpd_mux[0:8]=self.clmTile.mpd_read_mux(ClmBank.BANK_A)
        mpd_mux[8:]=self.clmTile.mpd_read_mux(ClmBank.BANK_B)
        
        self.MPD_PIC_uA_Ary= mpd_pic       
        self.MPD_MUX_uA_Ary= mpd_mux     
        return mpd_pic,mpd_mux
    
    def getVOA_attenuation_dB(self,I_mA):
        '''
        emperical formula for VOA
        '''
        return np.round(10*np.log10(1+0.061*I_mA),2)
    
    def getVirtualErrorSignal(self):
        N_order_bath=3
        ve=np.zeros(33)
        ve[0:16]=bathtub_derivative(self.I_LD_mA_Ary, 80, 185, self.I_LD_corner_BW, p_left=N_order_bath, p_right=N_order_bath)
        ve[16:32]=bathtub_derivative(self.I_VOA_mA_Ary, 0, 25, self.I_VOA_corner_BW, p_left=N_order_bath, p_right=N_order_bath)
        ve[32]=bathtub_derivative(self.T_PIC, 30, 60, self.T_PIC_corner_BW, p_left=N_order_bath, p_right=N_order_bath)
        return ve
        
    def updateCurrentState(self):
        '''
        Returns
        clm opticalState [0:32]: [0:16] optical frequency in GHz (relative to MUX peak frequency); [16:32] optical power in mW
        -------
        TYPE
            DESCRIPTION.

        '''
        self.currentControlVector[0:16]=self.I_LD_mA_Ary
        self.currentControlVector[16:32]=self.I_VOA_mA_Ary
        self.currentControlVector[32]=self.T_PIC
        
        if self.ditherElement=='LD':
            print('dithering %s not implemented yet.'%self.ditherElement)
            exit
        elif self.ditherElement=='T_PIC':
            '''
                XP note: 8/9/2025: instead of dither, we sweep T_PIC to average out MPD ripple effect
            '''
            Nsteps=11
            T_PIC_lo=self.T_PIC - self.ditherAmp
            T_PIC_hi=self.T_PIC + self.ditherAmp
            T_PICs=np.linspace(T_PIC_lo,T_PIC_hi,Nsteps)

            mpd_pics=np.zeros((16,Nsteps))
            mpd_muxs=np.zeros((16,Nsteps))
            rs= np.zeros((16,Nsteps))
            
            for j, t_pic in enumerate(T_PICs):
                self.clmTile.pid_config_setpoint(Lt8722Module.PIC,t_pic)
                if j==0:
                    time.sleep(self.sample_period*0.5)
                else:
                    time.sleep(self.sample_period/(Nsteps))
                    
                mpd_pics[:,j],mpd_muxs[:,j]=self.readMPDcurrent()
                rs[:,j]=np.clip(mpd_muxs[:,j],1,1e6)/np.clip(mpd_pics[:,j],1,1e6)
           
            mpd_pic=mpd_pics.mean(axis=1)# take mean
            mpd_mux=mpd_muxs.mean(axis=1) # take mean
            
            self.MPD_PIC_uA_Ary=mpd_pic
            self.MPD_MUX_uA_Ary=mpd_mux
            
            dRdT_PICs=np.zeros(16)
            for i in range(16):
                x=T_PICs
                y=rs[i,:]
                pFit=np.polyfit(x,y,1)
                dRdT_PICs[i]=pFit[0]            
                                      
            # fig, ax=plt.subplots(1,2,figsize=(8,4))
            # ax[0].plot(T_PICs,rs[0:8,:])
            # ax[1].plot(T_PICs,rs[8:16,:])
            # ax[0].grid(True);ax[1].grid(True)
            # plt.show()
            
            dRdT=dRdT_PICs*self.norm_PIC_MUX # dRdT is the derivative with respect to temperature. normalize such that it is independent of dither amp, or when we have a bright or dark MUX MPD
            self.dRdT=dRdT
            self.freq_offset_GHz=dRdT*self.dFreq_dRdT # frequency offset estimate 
            self.Pout_mW=(self.pout_mon_weight*mpd_pic/self.tap_ratio_pic+\
                        (1 - self.pout_mon_weight)*mpd_mux/self.tap_ratio_mux)*1e-3 # self.pout_mon_weight: do we trust mpd_PIC or mpd_mux? ideally, they should produce the same answer
                        
            self.currentState[0:16]=np.round(self.freq_offset_GHz,2)
            self.currentState[16:32]=np.round(self.Pout_mW,2)
            self.currentState[32:]=self.getVirtualErrorSignal()
                        
            # set temp back
            self.clmTile.pid_config_setpoint(Lt8722Module.PIC, self.T_PIC)
            
            return self.currentState
            
        elif self.ditherElement=='T_MUX':
            
            T_MUX_lo=self.T_MUX - self.ditherAmp
            T_MUX_hi=self.T_MUX + self.ditherAmp
            
            self.clmTile.pid_config_setpoint(Lt8722Module.MUX,T_MUX_lo)
            time.sleep(self.sample_period*0.5)
            mpd_pic_lo,mpd_mux_lo=self.readMPDcurrent()

            self.clmTile.pid_config_setpoint(Lt8722Module.MUX,T_MUX_hi)
            time.sleep(self.sample_period*0.5)
            mpd_pic_hi,mpd_mux_hi=self.readMPDcurrent()
            
            mpd_pic=0.5*(mpd_pic_hi+mpd_pic_lo) # take mean
            mpd_mux=0.5*(mpd_mux_lo+mpd_mux_hi) # take mean
            
            self.MPD_PIC_uA_Ary=mpd_pic
            self.MPD_MUX_uA_Ary=mpd_mux
            
            # try to avoid exceptions, which can happen when a channel has low/no power
            dR=mpd_mux_hi/np.clip(mpd_pic_hi,1,1000000) - mpd_mux_lo/np.clip(mpd_pic_lo,1,1000000)
            dR=np.where((mpd_pic_hi!=0) &(mpd_pic_lo!=0),dR,np.zeros(len(dR)))
            
            dRdT=dR/self.ditherAmp/2*self.norm_PIC_MUX # dRdT is the derivative with respect to temperature. normalize such that it is independent of dither amp, or when we have a bright or dark MUX MPD
            self.dRdT=dRdT
            self.freq_offset_GHz=dRdT*self.dFreq_dRdT # frequency offset estimate 
            self.Pout_mW=(self.pout_mon_weight*mpd_pic/self.tap_ratio_pic+\
                        (1 - self.pout_mon_weight)*mpd_mux/self.tap_ratio_mux)*1e-3 # self.pout_mon_weight: do we trust mpd_PIC or mpd_mux? ideally, they should produce the same answer
                        
            self.currentState[0:16]=np.round(self.freq_offset_GHz,2)
            self.currentState[16:32]=np.round(self.Pout_mW,2)
            self.currentState[32:]=self.getVirtualErrorSignal()
                        
            # set temp back
            self.T_MUX=0.5*(T_MUX_lo+T_MUX_hi) # 
            self.clmTile.pid_config_setpoint(Lt8722Module.MUX, self.T_MUX)
            
            return self.currentState

    def updateCostFunction(self):
        state=self.updateCurrentState()
        for i,ic in enumerate(self.costMask):
            if i<16:
                if ic==np.True_:
                    self.errorVector_norm[i]=(state[i]-self.freqOffset_GHz_tgt[i])/self.freqOffset_BW
                else:
                    self.errorVector_norm[i]=0
            elif i<32:
                if ic==np.True_:
                    self.errorVector_norm[i]=(state[i]-self.pout_mW_tgt[i-16])/self.pout_ratio_BW
                else:
                    self.errorVector_norm[i]=0
            else:
                # derivative of bathtub functions
                pass
        self.errorVector_norm=np.round(self.errorVector_norm,4)*self.costMask
        self.errorVector_norm=np.nan_to_num(self.errorVector_norm,nan=0)
        cFunc=np.sum((self.errorVector_norm)**2)
        self.costFunc=np.round(cFunc,3)
        
        self.dCFdx[32]=-16*2*np.sum(self.errorVector_norm[0:16])/self.freqOffset_BW+2*self.errorVector_norm[32+16+16] # x=T_PIC, impact only freqeuncy, so assumed 
        self.dCFdx[0:16]=-16/20*2*self.errorVector_norm[0:16]/self.freqOffset_BW+2*self.errorVector_norm[16:16+16]*0.1/self.pout_mW_tgt +\
        2*self.errorVector_norm[32:32+16]# x= LD bias. X-talk ignored, slope effeciency 0f 0.1 mW/mA assumed
        self.dCFdx[16:32]=2*self.errorVector_norm[16:16+16]*(-0.061/(1+0.061*self.I_LD_mA_Ary)**2) +\
        2*self.errorVector_norm[32+16:32+16+16]    # x= VOA bias
    
    def computeControlAdjustment(self):
#         '''
#         Note: this order is 'influenced' by clm tile test flow  
#         1. compute T_PIC adjustment
#         2. LD bias
#         3 VOA bias
#         '''
#         # since we know T_PIC impact all channels the same, we use it to adjust mean frequency error
        if np.sum(self.costMask[0:16])>0:
            meanFreqErr_norm=np.sum(self.errorVector_norm[0:16])/np.sum(self.costMask[0:16])
        else:
            meanFreqErr_norm=0 # if the user do not care frequency error at the moment, set it to zero for cost function calculation
            
        deltaT_PIC=-(meanFreqErr_norm*self.freqOffset_BW/(-16)+self.errorVector_norm[-1])*self.actMask[32]*self.P_of_PID[32]
        deltaT_PIC=np.clip(deltaT_PIC,-0.5,0.5) # clip T_PIC step size. 
        
        # 7/31/2025 XP: (-16)*deltaT_PIC term remove the frequency error that WILL be removed by T_PIC adjustment.
        deltaI_LD=-((self.errorVector_norm[0:16]*self.freqOffset_BW - (-16)*deltaT_PIC)/(-16/20)+self.errorVector_norm[32:32+16])*self.P_of_PID[0:16]*self.actMask[0:16]

        powerratioErr=self.errorVector_norm[16:32]*self.pout_ratio_BW/self.pout_mW_tgt
        
        #9/9 added: also use LD current to compensate for power, assume 0.15 slope effeciency
        alpha_LD=0.5 # portion of power correction using LD
        deltaI_LD=deltaI_LD-alpha_LD*(powerratioErr)*self.pout_mW_tgt/0.15*self.P_of_PID[0:16]*self.actMask[0:16]
        '''
        the above line can also simply written as:
            delta_I_LD=deltaI_LD -(self.pout_mW/self.pout_mW_tgt -1)*self.pout_mW_tgt/0.15*self.P_of_PID[0:16]*self.actMask[0:16]
        '''
        deltaI_LD=np.clip(deltaI_LD,-10,10) # clip LD bias step size.
        
        deltaI_VOA=((1-alpha_LD)*(powerratioErr)*(1+0.061*self.I_VOA_mA_Ary)/0.061+self.errorVector_norm[32+16:32+16+16])*self.actMask[16:32]*self.P_of_PID[16:32]
        deltaI_VOA=np.clip(deltaI_VOA,-1,1)  # clip LD bias step size.
        
        deltaControlVector=np.zeros(33)
        deltaControlVector[0:16]=deltaI_LD
        deltaControlVector[16:32]=deltaI_VOA
        deltaControlVector[32]=deltaT_PIC
        deltaControlVector=np.round(deltaControlVector,3)
        deltaControlVector=deltaControlVector*np.array(self.actMask)
        
        deltaControlVector=np.nan_to_num(deltaControlVector,nan=0)
        
        self.controlVector=self.controlVector+deltaControlVector
        # apply some intelegence to shift T_PIC
        ivoa_min=self.controlVector[16:32].min()
        ivoa_max=self.controlVector[16:32].max()
        if (ivoa_min>0.5) and (ivoa_max<24.5):
            pass
        elif (ivoa_min<=0.5):
            self.controlVector[32]+=-0.1
            self.controlVector[0:16]=self.controlVector[0:16]+np.ones(16)*2*self.costMask[16:32]
        elif ivoa_max>=24.5:
            self.controlVector[32]+=0.1
            self.controlVector[0:16]=self.controlVector[0:16]-np.ones(16)*2*self.costMask[16:32]
            
        self.controlVector[0:16]=np.clip(self.controlVector[0:16],0,185)
        self.controlVector[16:32]=np.clip(self.controlVector[16:32],0,25)
        self.controlVector[32]=np.clip(self.controlVector[32],30,65)
        return deltaControlVector
        
        '''
        Note:  
        compute update using dCfdx
        '''
            
        # deltaT_PIC=-self.dCFdx[32]*self.freqOffset_BW/(16)*self.actMask[32]*self.P_of_PID[32]/max([1,np.sum(self.costMask[0:16])])
        # deltaT_PIC=np.clip(deltaT_PIC,-0.3,0.3) # clip T_PIC step size. 
        
        # deltaI_LD=-(self.dCFdx[0:16]*self.freqOffset_BW)/(16/20)*self.P_of_PID[0:16]*self.actMask[0:16]
        
        # deltaI_LD=np.clip(deltaI_LD,-2.5,2.5) # clip LD bias step size.
        
        # deltaI_VOA=-self.dCFdx[16:32]*(1+0.061*self.I_VOA_mA_Ary)/0.061*self.actMask[16:32]*self.P_of_PID[16:32]
        # deltaI_VOA=np.clip(deltaI_VOA,-1,1)  # clip LD bias step size.
        
        # deltaControlVector=np.zeros(33)
        # deltaControlVector[0:16]=deltaI_LD
        # deltaControlVector[16:32]=deltaI_VOA
        # deltaControlVector[32]=deltaT_PIC
        # deltaControlVector=np.round(deltaControlVector,3)
        # deltaControlVector=deltaControlVector*np.array(self.actMask)
        
        # deltaControlVector=np.nan_to_num(deltaControlVector,nan=0)
        
        # self.controlVector=self.controlVector+deltaControlVector
        # self.controlVector[0:16]=np.clip(self.controlVector[0:16],0,185)
        # self.controlVector[16:32]=np.clip(self.controlVector[16:32],0,25)
        # self.controlVector[32]=np.clip(self.controlVector[32],30,65)
        # return deltaControlVector
    
    def applyControlVector(self):
        self.clmTile.pid_config_setpoint(Lt8722Module.PIC,self.controlVector[32])
        for ch in range(16):
            self.set_LD_current(ch,self.controlVector[ch])
            time.sleep(0.05)
            self.set_VOA_current(ch,self.controlVector[ch+16])
        
        self.I_LD_mA_Ary=self.controlVector[0:16]
        self.I_VOA_mA_Ary=self.controlVector[16:32]
        self.T_PIC=self.controlVector[32]
        
        
            
            

