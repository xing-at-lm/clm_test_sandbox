# -*- coding: utf-8 -*-
"""CLM bench session.

Run one cell at a time. The first cell only loads configuration. Later cells
open hardware and start a test. Pass note= to record why the run was made;
that text is stored in the note column of the results CSV.
The previous script is clm per test_old.py.
"""
# %% imports
import sys
import time
from pathlib import Path


def _script_dir():
    try:
        return Path(__file__).resolve().parent
    except NameError:
        cwd = Path.cwd()
        for candidate in (cwd, cwd / "scripts"):
            if (candidate / "clm_experiments.py").exists():
                return candidate
        return cwd


SCRIPT_DIR = _script_dir()
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import marlow_clm_control_api as mca # Marlow control API
import clm_tile_control_kurtz as kca # Kurtz control API
import clm_experiments as clm_exp
import clm_plots as plots

cfg = clm_exp.load_config(SCRIPT_DIR / "lab_config.yaml")

sn = "Y2534000160"

cfg["data_dir"] = Path(cfg["data_dir"]) / sn
DATA_DIR = cfg["data_dir"]

# %% connect testboard
tb, tile = clm_exp.connect_testboard(cfg)

# %% firmware and FRU
print(mca.get_tile_fw_version(tb))
print(mca.read_fru_info(tile))

# %% initialize tile
mca.initialize_tile(tb, tile=tile)
clm_exp.apply_control_config(tile, **cfg["control"])
time.sleep(0.2)
mca.optical_power_on(tile)

# %% lane monitors
for bank in (mca.ClmTile.Bank.BANK_A, mca.ClmTile.Bank.BANK_B):
    print(tile.read_lane_enables(bank))
    print(tile.read_aux_bias_current_monitors(bank))
    print(tile.read_bias_current_monitors(bank))
    print(tile.read_optical_power_monitors(bank))

# %% firmware update
# Uncomment the next line only when you intend to flash the tile.
# mca.update_fw(tile, cfg["firmware"]["dir"], cfg["firmware"]["filename"])

# %% power stability versus channel count
df = clm_exp.power_stability_vs_channel_count(
    tile,
    DATA_DIR,
    power_targets=[6, 8, 10, 12],
    sample_s=300,
    interval_s=2,
    settle_s=60,
    seed=None,
    tile_index=cfg["tile_index"],
    note="",
)

# %% plot power stability versus channel count
plots.plot_power_stability(df)

# %% power stability, 16 lanes then 1 lane
# single_channel is the only lane enabled in the second half of each target.
# settle_s waits after each new power target. The previous script set
# pause_time = 60 here but did not sleep.
df = clm_exp.power_stability_16_to_1(
    tile,
    DATA_DIR,
    power_targets=[6, 8, 10, 12],
    single_channel=0,
    sample_s=300,
    interval_s=2,
    settle_s=60,
    dither_period_s=6,
    kp_tpic=0.3,
    kp_ld=0.3,
    kp_voa=0.3,
    alpha_i_ld=0.5,
    tile_index=cfg["tile_index"],
    note="",
)

# %% plot 16 lanes then 1 lane
plots.plot_power_stability(df)

# %% power range, 16 lanes then 1 lane
df = clm_exp.power_range_sweep(
    tb,
    tile,
    DATA_DIR,
    power_targets=[12, 9, 6, 3, 5, 8, 11, 14],
    single_channel=0,
    sample_s=300,
    interval_s=2,
    settle_s=60,
    dither_period_s=2,
    kp_tpic=0.3,
    kp_ld=0.3,
    kp_voa=0.3,
    alpha_i_ld=0.5,
    tile_index=cfg["tile_index"],
    note="",
)

# %% plot power range
plots.plot_power_stability(df, p_ylim=None)

# %% open Thorlabs TEC
tec_thorlab = clm_exp.open_ptc(cfg)

# %% open PER meter
per_meter = clm_exp.open_per_meter(cfg)

# %% open ER-100 PER meter
per_meter_er100 = clm_exp.open_per_meter_er100(cfg)

# %% PER meter calibration
# Ramp the Thorlabs TEC temp0 -> temp1 -> temp0 while recording the ER-100.
df = clm_exp.per_meter_cal(
    tec_thorlab,
    per_meter_er100,
    DATA_DIR,
    temp0=25,
    temp1=45,
    ramp_time_s=1200,
    note="",
)
plots.plot_per_meter_cal(df)

# %% open Arroyo TEC
tec_arroyo = clm_exp.open_arroyo(cfg)

# %% open OSA
osa = clm_exp.open_osa(cfg)

#%%
clm_exp.apply_control_config(
    tile,
    dither_period_s=2,
    kp_tpic=0.2,
    kp_ld=0.2,
    kp_voa=0.2,
    alpha_i_ld=0.5,
    amp_t_mux=3)
# %% CLM to OSA loss calibration
dict_results = clm_exp.clm_to_OSA_loss_cal(tile, osa)

# %% CLM power and wavelength stability
df = clm_exp.clm_power_n_wavelength_stability(
    tile,
    osa,
    dict_results["clm_osa_loss_A"],
    dict_results["clm_osa_loss_B"],
    test_duration=60 * 15,
    sample_interval=10,
    trace="TRA",
    idx_tile=cfg["tile_index"],
)

# %% CLM power and wavelength stability report
# Uses the stability df above. Min PER is left out; this df has no PER samples.
import importlib
from datetime import datetime
import generate_clm_test_report as clm_report
importlib.reload(clm_report)

stability_spec = {
    name: clm_report.spec_config_dict[name]
    for name in (
        "wavelength error",
        "power error",
        "15 min power stability",
    )
}
stability_results = clm_exp.summarize_clm_stability(df, power_tgt_mw=8)
dict_test = {
    "test date": datetime.now().strftime("%m/%d/%Y"),
    "tester": "xing pan",
    "target power": "8 mW/ch",
}
dut = {"Serial Number": sn, "Part Number": "2120000153"}
report_paths = clm_report.generate_clm_test_report(
    dict_test,
    dut,
    stability_spec,
    results=stability_results,
    output_dir=DATA_DIR,
)

# %% wait for the case to settle
# time.sleep(3600)
#%%

# %% PER versus temperature
df = clm_exp.per_vs_temperature(
    tile,
    tec_thorlab,
    tec_arroyo,
    per_meter,
    DATA_DIR,
    bank=0,
    channels=[3, 4, 5, 6], # [0, 1, 2, 3, 4, 5, 6, 7]
    stabilize_s=300,
    channel_settle_s=30,
    sample_s=120,
    sample_interval_s=10,
    tile_index=cfg["tile_index"],
    update_period_s=cfg["thermal"]["interval_s"],
    note="",
)
plots.plot_per_vs_temperature(df)

# %% per_over_temp_2
# Continuous PER while the TEC profile steps. per_meter must be oz_er1000.ER1000.
df = clm_exp.per_over_temp_2(
    tile,
    tec_thorlab,
    tec_arroyo,
    per_meter,
    DATA_DIR,
    bank=0,
    channels=[3, 4, 5, 6],
    channel_settle_s=15,
    channel_s=10,
    tile_index=cfg["tile_index"],
    note="",
)
plots.plot_per_vs_temperature(df)

# %% per_over_temp_3
# Enable one channel, wait, then ramp both TECs temp0 -> temp1 -> temp0
# across channel_s while recording PER. per_meter must be oz_er1000.ER1000.
for t_mux in[40,30,25]:
    tile.write_mux_tec_setpoint(t_mux)
    time.sleep(20)
print('mux temp: ',tile.read_mux_tec_setpoint())
#%%
df = clm_exp.per_over_temp_3(
    tile,
    tec_thorlab,
    tec_arroyo,
    per_meter,
    DATA_DIR,
    bank=0,
    channels=[3, 4, 5, 6],
    temp0=30,
    temp1=36,
    channel_settle_s=20,
    channel_s=600,
    tile_index=cfg["tile_index"],
    note="",
)
plots.plot_per_vs_temperature(df)

# %% CLM port A and port B PER test
# Disables every lane, prompts for port A, runs per_over_temp_3, then port B.
import importlib
from datetime import datetime
importlib.reload(clm_exp)
df_per = clm_exp.clm_per_test(
    tile,
    tec_thorlab,
    tec_arroyo,
    per_meter,
    DATA_DIR,
    channels=range(8),
    temp0=30,
    temp1=36,
    channel_settle_s=20,
    channel_s=600,
    tile_index=cfg["tile_index"],
    note="",
)
per_summary = clm_exp.channel_min_per(df_per)

import generate_clm_test_report as clm_report
importlib.reload(clm_report)
if "stability_results" in globals():
    report_results = stability_results.merge(
        per_summary, on=["tile", "bank", "channel"], how="outer"
    )
else:
    report_results = per_summary

report_spec = {
    name: clm_report.spec_config_dict[name]
    for name in (
        "wavelength error",
        "power error",
        "15 min power stability",
        "min PER",
    )
    if name in report_results.columns
}

dict_test = {
    "test date": datetime.now().strftime("%m/%d/%Y"),
    "tester": "xing pan",
    "target power": "8 mW/ch",
}
dut = {"Serial Number": sn, "Part Number": "2120000153"}
report_paths = clm_report.generate_clm_test_report(
    dict_test,
    dut,
    report_spec,
    results=report_results,
    output_dir=DATA_DIR,
)

# %% PER versus frequency offset
df = clm_exp.per_vs_frequency_offset(
    tile,
    tec_thorlab,
    tec_arroyo,
    per_meter,
    DATA_DIR,
    bank=0,
    channels=[0, 1, 2, 3, 4, 5, 6, 7],
    freq_offsets=[-20, -15, -10, -5, 0, 5, 10, 15, 20],
    channel_settle_s=30,
    sample_s=120,
    sample_interval_s=10,
    tile_index=cfg["tile_index"],
    note="",
)
plots.plot_per_vs_frequency(df)

# %% PER until KeyboardInterrupt
df = clm_exp.per_passive(per_meter, tec_thorlab, tec_arroyo, DATA_DIR, period_s=10, note="")

# %% plot a saved overnight PER file
plots.plot_overnight(cfg["overnight_csv"])

# %% close Arroyo TEC
# tec_arroyo.close()
