#!/usr/bin/env python
# coding: utf-8

# ### Dependencies

import time
from pathlib import Path

import pandas as pd
import numpy as np
#import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from cmis_host.adapters import AardvarkI2cMci
#from cmis_host.platform.clm.test_board import KlmModuleTestBoard

from cmis_host.platform.clm import ClmTile, ClmTileMarlow, CntCarrierBoard,ClmEvkBoard, KlmModuleTestBoard,ClmModuleTestBoard

from cmis_host.adapters.aardvark_lm import Aardvark
import cmis_host.adapters.aardvark_lm as aardvark_lm
import yaml
import copy
#print(aardvark_lm.__file__)

# ### Configuration of default values

# In[12]:

cLight=299792458
target_grid_nm = np.array([
  1301.47, 1302.60, 1303.73, 1304.87,
  1306.01, 1307.14, 1308.28, 1309.43,
  1310.57, 1311.72, 1312.87, 1314.02,
  1315.17, 1316.33, 1317.48, 1318.64
  ])

target_grid_GHz=np.round(cLight/target_grid_nm,0)

target_grid_nm_A=target_grid_nm[0::2]
target_grid_nm_B=target_grid_nm[1::2]

target_grid_GHz_A=target_grid_GHz[0::2]
target_grid_GHz_B=target_grid_GHz[1::2]

DEFAULT_RUNTIME_S        = 0  # 0 == Off
DEFAULT_POLLING_PERIOD_S = 1

DEFAULT_MAX_TEMP = 75  # Datasheet: -40°C < operational temperature < 125°C
CAP_MAX_TEMP     = 125

DEFAULT_POWER_SETTING_MW = 8
CAP_MIN_POWER_SETTING_MW = 0.1
CAP_MAX_POWER_SETTING_MW = 14.5

DEFAULT_FREQUENCY_OFFSET_SETTING_GHZ = [0.] * ClmTile.NUM_LANES

DEFAULT_LASERS_FIRING                = [3,4]

# MUX temperature dither amplitude
DEFAULT_STABILIZATION_AMPLITUDE_C = 3
# MUX temperature dither period
DEFAULT_STABILIZATION_PERIOD_S    = 2


runtime_s      = DEFAULT_RUNTIME_S
poll_period_s  = DEFAULT_POLLING_PERIOD_S
max_temp       = DEFAULT_MAX_TEMP
pwr_setpoint   = DEFAULT_POWER_SETTING_MW
freq_setpoint  = DEFAULT_FREQUENCY_OFFSET_SETTING_GHZ
lasers_firing  = DEFAULT_LASERS_FIRING
stab_amplitude = DEFAULT_STABILIZATION_AMPLITUDE_C
stab_period    = DEFAULT_STABILIZATION_PERIOD_S


# #### Input sanitization

if max_temp > CAP_MAX_TEMP:
    max_temp = CAP_MAX_TEMP
    print(f"INFO: Capped {max_temp=}")
if pwr_setpoint > CAP_MAX_POWER_SETTING_MW:
    pwr_setpoint = CAP_MAX_POWER_SETTING_MW
    print(f"INFO: Capped {pwr_setpoint=}")
elif pwr_setpoint < CAP_MIN_POWER_SETTING_MW:
    pwr_setpoint = CAP_MIN_POWER_SETTING_MW
    print(f"INFO: Capped {pwr_setpoint=}")


# #### Setup board


def get_testboard(AARDVARK_SN_P0 = 2239879015,AARDVARK_SN_Io =  2239831227,AARDVARK_SN_P1 =2239789566,testBoardType='CLM-M'):
    '''
    testBoardType='EVK' #'CNT', 'EVK', 'KLM', 'CLM-M': clm module test board 
    '''
    try:
        # Close existing Aardvarks if we did not shut down properly
        tb.close()
    except Exception:
        pass
    if testBoardType=='EVK':
        ''' verified to work '''
        # aardvark_io = AardvarkI2cMci(serial_number=AARDVARK_SN_Io)
        # aardvark_clm = AardvarkI2cMci(serial_number=AARDVARK_SN_P0)
        tb = ClmEvkBoard.from_auto_scan()    
        time.sleep(0.5)
        tb.reset_tiles()
    if testBoardType=='CNT':
        ''' did not work on one CNT board. same board run CNT cli did work'''
        tb = CntCarrierBoard.from_auto_scan()
    if testBoardType=='CLM-M':
        ''' did not work on one CNT board. same board run CNT cli did work'''
        aardvark_io = AardvarkI2cMci(serial_number=AARDVARK_SN_Io)
        aardvark_clm = AardvarkI2cMci(serial_number=AARDVARK_SN_P0)
        tb = ClmModuleTestBoard(aardvark_io,
                                        aardvark_clm)                    
        time.sleep(0.5)
        tb.reset_tiles()
    return tb

def shut_down_test_board_nicely(tb):
    for _, tile_idx, tile in tb.module.iter_tiles(indices=True):
        try:
            tile.write_low_pwr_req_s(1)
        except Exception as e:
            print(f"WARNING: Failed to shut down Tile {tile_idx}: {e}")
    time.sleep(0.1)  # Arbitrary small delay to allow CLM state transition
    
    tb.close()
    return 0


def get_tile_fw_version(tb):
    dict_fw_info={}
    for tile in tb.module.iter_tiles():
        if tile:
            print(f'{tile.dev_addr} present, FW version: {tile.read_fw_version()}')
            dict_fw_info.update({tile.dev_addr:tile.read_fw_version()})
    return dict_fw_info

def reset_tiles(tb):
    tb.reset_tiles()
    return 0

def read_fru_info(tile):
    v_fru={}
    x=tile.flash_read_fru_mfg_id()
    v_fru.update({'part_number':x.part_number.split(b'\x00')[0].decode('utf-8'),
                  'revision':x.revision.split(b'\x00')[0].decode('utf-8'),
                  'serial_number':x.serial_number.split(b'\x00')[0].decode('utf-8'),
                  'version':x.version})
    return v_fru

def get_fru_full(tile):
    
    v_fru={}
    x=tile.flash_read_fru_mfg_id()
    v_fru.update({'part_number':x.part_number.split(b'\x00')[0].decode('utf-8'),
                  'revision':x.revision.split(b'\x00')[0].decode('utf-8'),
                  'serial_number':x.serial_number.split(b'\x00')[0].decode('utf-8'),
                  'version':x.version})
    
    x=tile.flash_read_fru_app_settings()
    v_fru.update({'power_targets':x.power_targets,
                  'freq_error_targets':x.freq_error_targets,
                  'voa_currents':x.voa_currents})
    
    x=tile.flash_read_fru_setpoints()
    v_fru.update({'bias_currents':x.bias_currents,
                  'temp_pic':x.temp_pic,
                  'temp_mux':x.temp_mux,
                  'pic_mpd_currents': list(x.pic_mpd_currents),
                  'mux_mpd_currents': list(x.mux_mpd_currents),
                  'mux_mpd_gain_settings': list(x.mux_mpd_gain_settings)})
    
    
    x=tile.flash_read_fru_optical_measurements()
    
    v_fru.update({'pic_mpd_tap_ratios':x.pic_mpd_tap_ratios,
                  'mux_mpd_tap_ratios':x.mux_mpd_tap_ratios,
                  'mux_freq_errors':x.mux_freq_errors})
    return v_fru


def update_fw(tile,dir_fw,filename_fw):
    tile.bootloader.reset_to_application()
    time.sleep(1)
    tile.reopen()
    time.sleep(1)
    tile.firmware_update(dir_fw+'\\'+filename_fw)
    return 0
#%%
from cmis_host.platform.clm import ClmTile
def reinitialize_bootloader(tb,dir_fw,filename_fw,idxTiles=[3,4]):
    
    PATH_TO_FIRMWARE_BINARY=dir_fw+'\\'+filename_fw
    from cmis_host.platform.clm.bootloader import ClmBootloader
    from cmis_host.platform.clm.test_board import ClmTileTestBoard

    tb = ClmTileTestBoard.from_auto_scan(allow_bl=True)
    tb.firmware_update(PATH_TO_FIRMWARE_BINARY)

    for tile_idx in idxTiles:
        tb.hold_tiles_in_reset()
        tb.reset_tiles(1 << tile_idx)
        tile = tb.module.packages[0].tiles[tile_idx]
        if isinstance(tile, ClmBootloader):
            tile.firmware_update(PATH_TO_FIRMWARE_BINARY)
            print('Success!')
            break
        else:
            continue
    else:
        print('No tile found in bootloader - must be a different problem.')



def target_power_mw_from_spec():
    """Channel power setpoint from clm_test_spec.yaml, in mW.

    Capped to the same 0.1–14.5 mW range as the old default setpoint.
    """
    path = Path(__file__).resolve().parent / "clm_test_spec.yaml"
    with open(path, encoding="utf-8") as handle:
        spec = yaml.safe_load(handle) or {}
    section = spec.get("clm_test_spec") or {}
    if "target_power" not in section:
        raise ValueError(f"clm_test_spec.target_power is missing from {path}")
    power = float(section["target_power"])
    if power > CAP_MAX_POWER_SETTING_MW:
        print(f"INFO: Capped target_power {power:g} mW to {CAP_MAX_POWER_SETTING_MW:g} mW")
        power = CAP_MAX_POWER_SETTING_MW
    elif power < CAP_MIN_POWER_SETTING_MW:
        print(f"INFO: Capped target_power {power:g} mW to {CAP_MIN_POWER_SETTING_MW:g} mW")
        power = CAP_MIN_POWER_SETTING_MW
    return power


def _configure_tile_low_power(tile_i, tile_idx="?"):
    """Fault masks, dither defaults, low-power, and power/freq setpoints (no software_reset)."""
    global pwr_setpoint
    pwr_setpoint = target_power_mw_from_spec()
    print(f"Optical power setpoint {pwr_setpoint:g} mW from clm_test_spec.yaml")
    tile_i.write_mask_optical_fault_passive(1)
    tile_i.write_mask_optical_fault_active(1)
    tile_i.write_self_test_s(0)
    tile_i.write_stabilization_amplitude(stab_amplitude)
    tile_i.write_stabilization_period(stab_period)
    tile_i.write_low_pwr_req_s(1)
    for b, bank in enumerate(ClmTile.Bank):
        try:
            tile_i.write_optical_power_setpoints(bank, [pwr_setpoint] * ClmTile.NUM_LANES_PER_BANK)
        except Exception as e:
            print(f"ERROR: Failed to set power setpoint for Tile {tile_idx}.{bank}: {e}")
        try:
            bank_setpoints = freq_setpoint[(b * ClmTile.NUM_LANES_PER_BANK):((b + 1) * ClmTile.NUM_LANES_PER_BANK)]
            tile_i.write_lane_frequency_setpoints(bank, bank_setpoints)
        except Exception as e:
            print(f"ERROR: Failed to set frequency setpoint for Tile {tile_idx}.{bank}: {e}")


def initialize_tile(tb, tile=None, software_reset=True):
    """Bring tile(s) to low-power with default setpoints.

    ``software_reset=False`` skips CMIS SoftwareReset (use after a board
    hardware reset + tile.reopen() when SoftwareReset hits I2C -103).
    """
    if tile is None:
        for _, tile_idx, tile_i in tb.module.iter_tiles(indices=True):
            print(tile_idx)
            print(tile_i)
            if not isinstance(tile_i, ClmTileMarlow):
                raise TypeError(
                    f"Expected tile of type {ClmTileMarlow.__name__} but encountered {type(tile_i).__name__}"
                )
            if software_reset:
                tile_i.software_reset()
            _configure_tile_low_power(tile_i, tile_idx=tile_idx)
    else:
        tile_i = tile
        if not isinstance(tile_i, ClmTileMarlow):
            raise TypeError(
                f"Expected tile of type {ClmTileMarlow.__name__} but encountered {type(tile).__name__}"
            )
        if software_reset:
            tile_i.software_reset()
        _configure_tile_low_power(tile_i, tile_idx="?")
    return 0

def optical_power_on(tile):        
    # Continue SM transitions to Mission if requested for given tile
    print("INFO: Firing up Tile ... wait ...")
    tile.write_autoreduction_s(1)
    tile.write_low_pwr_req_s(0)
    while not tile.read_temp_lock_complete():
        time.sleep(0.1)

    # Transition to Mission
    tile.write_autoreduction_s(0)
    print("\n------------------------------\nWARNING: TILE IS Optical ON \n------------------------------\n")
    return 0

def set_clm_control_config(tile,dither_period_s=6,kp_tpic=0.3,kp_ld=0.3,kp_voa=0.3,amp_t_mux=3,ALPHA_I_LD=0.5):
    tile.write_stabilization_period(dither_period_s)
    time.sleep(0.05)
    ALPHA_I_LD=ALPHA_I_LD
    tile.write_stabilization_pid_constants(kp_ld, kp_tpic, kp_voa, ALPHA_I_LD)
    time.sleep(0.05)
    tile.write_stabilization_amplitude(amp_t_mux)
    return 0

def set_clm_output_targets(tile,freq_offset_targets=[0]*16,power_targets=[7]*16):
    for b, bank in enumerate(ClmTile.Bank):
        try:
            tile.write_optical_power_setpoints(bank, power_targets[int(8*b):8+int(b*8)])
        except Exception as e:
            print(f"ERROR: Failed to set power setpoint for {bank}: {e}")
        try:
            tile.write_lane_frequency_setpoints(bank, freq_offset_targets[int(8*b):8+int(b*8)])
        except Exception as e:
            print(f"ERROR: Failed to set frequency setpoint for {bank}: {e}")
    return 0

def clm_lane_enable(tile,chEnableMask=[1]*16):
    for b, bank in enumerate(ClmTile.Bank):
        try:
            for ch in range(8):
                tile.write_lane_enable(bank, ch,chEnableMask[ch+8*b])
        except Exception as e:
            print(f"ERROR: Failed to set lane enable bit for {bank}, ch {ch}: {e}")
    return 0

def get_faults(tile):
    return {'fault': tile.read_fault().name,'custom_fault':tile.read_custom_fault().name}
    

def get_clm_record(tiles, duration_s=30, interval_s=2, tweakT_PIC=False):
    cols=['time','et_s','tile','bank','T_PMIC','T_PIC','T_MUX']+\
         [f'I_LD{i}' for i in range(8)]+\
         [f'I_VOA{i}' for i in range(8)]+\
         [f'p_mon{i}' for i in range(8)]+\
         [f'p_set{i}' for i in range(8)]+\
         [f'freq_mon{i}' for i in range(8)]+\
         [f'fOffset_set{i}' for i in range(8)]+\
         [f'fErr{i}' for i in range(8)]

    def get_bank_record(tile,bank):
        rcd=[]
        time.sleep(delta_t_bwt_read)
        rcd.append(np.round(tile.read_temp_pmic(),2))
        time.sleep(delta_t_bwt_read)
        rcd.append(np.round(tile.read_temp_pic(),2))
        time.sleep(delta_t_bwt_read)
        rcd.append(np.round(tile.read_temp_mux(),2))

        time.sleep(delta_t_bwt_read)
        monitors=tile.read_bias_current_monitors(bank)
        time.sleep(delta_t_bwt_read)
        for mvar in monitors:
            rcd.append(mvar)
        monitors=tile.read_aux_bias_current_monitors(bank)
        time.sleep(delta_t_bwt_read)
        for mvar in monitors:
            rcd.append(mvar)

        monitors = tile.read_optical_power_monitors(bank)
        time.sleep(delta_t_bwt_read)
        for mvar in monitors:
            rcd.append(mvar)
        setpoints = tile.read_optical_power_setpoints(bank)
        time.sleep(delta_t_bwt_read)
        for mset in setpoints:
            rcd.append(mset)

        monitors = tile.read_lane_frequencies(bank)
        time.sleep(delta_t_bwt_read)
        for mvar in monitors:
            rcd.append(mvar)
        setpoints=tile.read_lane_frequency_setpoints(bank)    
        time.sleep(delta_t_bwt_read)
        for mset in setpoints:
            rcd.append(mset)
            
        if bank==ClmTile.Bank.BANK_A:    
            ferr=np.array(monitors)-(np.array(setpoints)+target_grid_GHz_A)
        else:
            ferr=np.array(monitors)-(np.array(setpoints)+target_grid_GHz_B)
        for e in ferr:
            rcd.append(e)
            
        return rcd   

    t0=time.time()
    rcds=[]
    et_s=0
    delta_t_bwt_read=0.1
    while et_s<=duration_s:
         for i, tile in enumerate(tiles):
            i_ld = []
            i_voa = []
            for b, bank in enumerate(ClmTile.Bank):
                t=time.time()
                et_s=t-t0
                rcd=[]
                rcd.append(time.ctime())
                rcd.append(et_s)
                rcd.append(i)

                rcd.append(b)
                rcd_bank=get_bank_record(tile,bank)
                rcds.append(rcd+rcd_bank)

                i_ld.extend(rcd_bank[3:3+8])
                i_voa.extend(rcd_bank[11:11+8])
            if tweakT_PIC:
                i_ld = np.asarray(i_ld, dtype=float)
                i_voa = np.asarray(i_voa, dtype=float)
                score = np.zeros(16)
                for ch in range(16):
                    if i_ld[ch] >= 184:
                        score[ch] += 1
                    if i_voa[ch] <= 0.1:
                        score[ch] += -1
                    if i_ld[ch] <= 80:
                        score[ch] += -1
                    if i_voa[ch] >= 19.9:
                        score[ch] += 1
                t_pic_step = 0.08 * np.sign(np.sum(np.unique(score)))
                if t_pic_step != 0:
                    t_pic = tile.read_pic_tec_setpoint() + t_pic_step
                    t_pic = float(np.clip(t_pic, 30, 65))
                    tile.write_pic_tec_setpoint(t_pic)
                    print(
                        f"\nT_PIC setpoint -> {t_pic:.2f} C "
                        f"(score {np.sum(np.unique(score)):.0f})",
                        flush=True,
                    )
         time.sleep(interval_s)
         print(f'{et_s/duration_s*100:.2f}%,', end='', flush=True)
         #print(f'et_s, duration: ,{et_s:.1f},{duration_s}')
    return rcds, cols   

def load_config_data_from_yaml(tile,yamlDir,yamlFile,powerTragets,freqOffsetTargets,initVOAOffsets,T_PIC_offset=0):
    yaml_file_path=yamlDir+'\\'+yamlFile
    with open(yaml_file_path, 'r') as file:
            data_prev = yaml.safe_load(file)
            print(data_prev)#
    POWER_TARGETS = powerTragets
    FREQ_TARGETS=freqOffsetTargets
    
    mux_freq_errors=[40]*16
    
    """write setpoints"""
    pic_tap_ratios = data_prev["responsivity"]["pic"]
    mux_tap_ratios = data_prev["responsivity"]["mux"]
    temp_pic = data_prev["temp"]["pic"]+T_PIC_offset
    temp_mux = data_prev["temp"]["mux"]
    bias_currents = np.clip(np.array(data_prev["current"]["laser"]),0,180)
    voa_current_old = copy.deepcopy(data_prev['current']['voa'])
    initVOAcurrents=list(np.array(voa_current_old)+np.array(initVOAOffsets))
    #voa_current_old = [15]*16
    existing = get_fru_full(tile)
    tile.flash_write_fru_app_settings(voa_currents=initVOAcurrents, power_targets=POWER_TARGETS, freq_error_targets=FREQ_TARGETS)
    tile.flash_write_fru_optical_measurements(pic_tap_ratios, mux_tap_ratios, mux_freq_errors)
    tile.flash_write_fru_setpoints(temp_pic,
                                  temp_mux,
                                  bias_currents.tolist(),
                                  list(existing.get("pic_mpd_currents") or [0]*16),
                                  list(existing.get("mux_mpd_currents") or [0]*16),
                                  list(existing.get("mux_mpd_gain_settings") or [0]*16))
    tile.software_reset() 

def write_new_fru_data(tile,dict_fru_revised):    
    pic_tap_ratios = dict_fru_revised['pic_mpd_tap_ratios']
    mux_tap_ratios = dict_fru_revised["mux_mpd_tap_ratios"]
    temp_pic = dict_fru_revised["temp_pic"]
    temp_mux = dict_fru_revised["temp_mux"]
    bias_currents = np.clip(np.array(dict_fru_revised["bias_currents"]),0,185)
    initVOAcurrents=dict_fru_revised['voa_currents']
    POWER_TARGETS=dict_fru_revised['power_targets']
    FREQ_TARGETS=dict_fru_revised['freq_error_targets']
    mux_freq_errors=[40]*16
    pic_mpd_currents = list(dict_fru_revised.get("pic_mpd_currents") or [0]*16)
    mux_mpd_currents = list(dict_fru_revised.get("mux_mpd_currents") or [0]*16)
    mux_mpd_gain_settings = list(dict_fru_revised.get("mux_mpd_gain_settings") or [0]*16)
    #voa_current_old = [15]*16
    tile.flash_write_fru_app_settings(voa_currents=initVOAcurrents, power_targets=POWER_TARGETS, freq_error_targets=FREQ_TARGETS)
    tile.flash_write_fru_optical_measurements(pic_tap_ratios, mux_tap_ratios, mux_freq_errors)
    tile.flash_write_fru_setpoints(temp_pic,
                                  temp_mux,
                                  bias_currents.tolist(),
                                  pic_mpd_currents,
                                  mux_mpd_currents,
                                  mux_mpd_gain_settings)
    tile.software_reset() 

if __name__=='__main__':
    pass




