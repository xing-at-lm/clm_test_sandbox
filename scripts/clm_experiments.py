"""CLM bench experiments.

Hardware is only touched when a function here is called. Importing this
module does not open the testboard, the PTC, or the PER meter.
"""

import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

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

import marlow_clm_control_api as mca

N_LANES = 16
LANES_PER_BANK = 8
PER_FIELDS = ("per_dB", "theta", "pow")


def load_config(path=None):
    """Load lab_config.yaml from scripts/, or from path."""
    if path is None:
        path = Path(__file__).resolve().parent / "lab_config.yaml"
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _scripts_on_path():
    scripts_dir = str(Path(__file__).resolve().parent)
    if scripts_dir not in sys.path:
        sys.path.append(scripts_dir)


def connect_testboard(cfg):
    """Open the testboard named in cfg and return (testboard, tile)."""
    tb = mca.get_testboard(
        AARDVARK_SN_P0=cfg["aardvark"]["p0"],
        AARDVARK_SN_Io=cfg["aardvark"]["io"],
        AARDVARK_SN_P1=cfg["aardvark"]["p1"],
        testBoardType=cfg["test_board_type"],
    )
    tile = tb.module.packages[0].tiles[cfg["tile_index"]]
    return tb, tile


def apply_control_config(
    tile,
    dither_period_s,
    kp_tpic,
    kp_ld,
    kp_voa,
    alpha_i_ld,
    amp_t_mux=3,
):
    """Write the dither period and stabilization gains."""
    return mca.set_clm_control_config(
        tile,
        dither_period_s=dither_period_s,
        kp_tpic=kp_tpic,
        kp_ld=kp_ld,
        kp_voa=kp_voa,
        amp_t_mux=amp_t_mux,
        ALPHA_I_LD=alpha_i_ld,
    )


def lane_mask(enabled_channels):
    """Return a length-16 mask with the given lane indexes set to 1."""
    mask = [0] * N_LANES
    for channel in enabled_channels:
        mask[int(channel)] = 1
    return mask


def bank_for(bank_index):
    banks = {
        0: mca.ClmTile.Bank.BANK_A,
        1: mca.ClmTile.Bank.BANK_B,
    }
    return banks[bank_index]


def record_condition(tile, enable_mask, sample_s, interval_s, **tags):
    """Enable lanes, record, and attach the mask plus any extra columns."""
    mca.clm_lane_enable(tile, chEnableMask=list(enable_mask))
    records, columns = mca.get_clm_record(
        [tile], duration_s=sample_s, interval_s=interval_s
    )
    frame = pd.DataFrame(records, columns=columns)
    for key, value in tags.items():
        frame[key] = value
    for channel, enabled in enumerate(enable_mask):
        frame[f"ch{channel}_enable"] = enabled
    frame["Nch_enabled"] = int(sum(enable_mask))
    return frame


def save_run(frame, data_dir, stem, params=None):
    """Write frame to data_dir and return the CSV path."""
    frame = frame.copy()
    if params:
        for key, value in params.items():
            frame[key] = value
    stamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    path = Path(data_dir) / f"{stem}_{stamp}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    print(path)
    return path


def _finish_frames(frames, data_dir, stem, params):
    if not frames:
        print("No samples collected.")
        return pd.DataFrame()
    frame = pd.concat(frames, ignore_index=True)
    save_run(frame, data_dir, stem, params)
    return frame


def _control_params(tile_index, sample_s, interval_s, control):
    params = {
        "sample_s": sample_s,
        "interval_s": interval_s,
    }
    if tile_index is not None:
        params["tile_index"] = tile_index
    if control:
        params.update(control)
    return params


def power_stability_vs_channel_count(
    tile,
    data_dir,
    power_targets=(6, 8, 10, 12),
    channel_counts=None,
    sample_s=300,
    interval_s=2,
    settle_s=60,
    seed=None,
    tile_index=None,
    note="",
):
    """Record each power target at several enabled-lane counts.

    channel_counts defaults to the odd counts from 1 through 15, in random
    order. seed makes that order and the lane picks repeatable.
    """
    if channel_counts is None:
        channel_counts = list(range(1, N_LANES + 1, 2))
    else:
        channel_counts = list(channel_counts)
    rng = random.Random(seed)

    stem = (
        "clm stability over pow target and Nchs"
        f"_sample duration-{sample_s}_sample interval-{interval_s}"
    )
    params = _control_params(tile_index, sample_s, interval_s, None)
    params["note"] = note
    frames = []
    try:
        for power_target in tqdm(power_targets, desc="power tgt"):
            mca.set_clm_output_targets(tile, [0] * N_LANES, [power_target] * N_LANES)
            time.sleep(settle_s)
            count_order = list(channel_counts)
            rng.shuffle(count_order)
            for n_channels in tqdm(count_order, desc="enable chs"):
                picked = rng.sample(range(N_LANES), n_channels)
                frames.append(
                    record_condition(
                        tile,
                        lane_mask(picked),
                        sample_s,
                        interval_s,
                        pow_tgt=power_target,
                    )
                )
    except KeyboardInterrupt:
        print("Interrupted. Saving samples collected so far.")
        return _finish_frames(frames, data_dir, stem + "_partial", params)
    except Exception:
        print("Saving samples collected before the error.")
        _finish_frames(frames, data_dir, stem + "_partial", params)
        raise
    return _finish_frames(frames, data_dir, stem, params)


def power_stability_16_to_1(
    tile,
    data_dir,
    power_targets=(6, 8, 10, 12),
    single_channel=0,
    sample_s=300,
    interval_s=2,
    settle_s=60,
    dither_period_s=6,
    kp_tpic=0.3,
    kp_ld=0.3,
    kp_voa=0.3,
    alpha_i_ld=0.5,
    amp_t_mux=3,
    tile_index=None,
    note="",
):
    """At each power target, record all 16 lanes, then one lane.

    single_channel is the only lane enabled in the second record.
    """
    control = dict(
        dither_period_s=dither_period_s,
        kp_tpic=kp_tpic,
        kp_ld=kp_ld,
        kp_voa=kp_voa,
        alpha_i_ld=alpha_i_ld,
        amp_t_mux=amp_t_mux,
    )
    apply_control_config(tile, **control)
    return _power_pair_sweep(
        tile,
        data_dir,
        power_targets,
        single_channel,
        sample_s,
        interval_s,
        settle_s,
        control,
        tile_index,
        stem_prefix="clm stability over pow target 16chs to 1ch",
        note=note,
    )


def power_range_sweep(
    tb,
    tile,
    data_dir,
    power_targets=(12, 9, 6, 3, 5, 8, 11, 14),
    single_channel=0,
    sample_s=300,
    interval_s=2,
    settle_s=60,
    dither_period_s=2,
    kp_tpic=0.3,
    kp_ld=0.3,
    kp_voa=0.3,
    alpha_i_ld=0.5,
    amp_t_mux=3,
    tile_index=None,
    note="",
):
    """Same 16-lane then 1-lane record, over a wider power list.

    The tile is re-initialized before the first power target.
    """
    control = dict(
        dither_period_s=dither_period_s,
        kp_tpic=kp_tpic,
        kp_ld=kp_ld,
        kp_voa=kp_voa,
        alpha_i_ld=alpha_i_ld,
        amp_t_mux=amp_t_mux,
    )
    apply_control_config(tile, **control)
    return _power_pair_sweep(
        tile,
        data_dir,
        power_targets,
        single_channel,
        sample_s,
        interval_s,
        settle_s,
        control,
        tile_index,
        stem_prefix="clm stability over pow target 16chs to 1ch",
        reinitialize=lambda: _reinitialize_tile(tb, tile),
        note=note,
    )


def _reinitialize_tile(tb, tile):
    mca.clm_lane_enable(tile, chEnableMask=[0] * N_LANES)
    mca.initialize_tile(tb, tile=tile)
    mca.optical_power_on(tile)


def _gain_tag(control):
    def dotted(value):
        return str(value).replace(".", "p")

    return (
        f"kTpic-{dotted(control['kp_tpic'])}"
        f"_kLD-{dotted(control['kp_ld'])}"
        f"_kVOA-{dotted(control['kp_voa'])}"
        f"_alpha-{dotted(control['alpha_i_ld'])}"
        f"_ditherPeriod-{control['dither_period_s']}"
    )


def _power_pair_sweep(
    tile,
    data_dir,
    power_targets,
    single_channel,
    sample_s,
    interval_s,
    settle_s,
    control,
    tile_index,
    stem_prefix,
    reinitialize=None,
    note="",
):
    power_targets = list(power_targets)
    one_lane = lane_mask([single_channel])
    all_lanes = [1] * N_LANES
    stem = (
        f"{stem_prefix}_sample duration-{sample_s}_sample interval-{interval_s}"
        f"_{_gain_tag(control)}"
    )
    params = _control_params(tile_index, sample_s, interval_s, control)
    params["single_channel"] = single_channel
    params["note"] = note
    frames = []
    try:
        for index, power_target in tqdm(
            enumerate(power_targets), desc="power tgt", total=len(power_targets)
        ):
            if index == 0 and reinitialize is not None:
                reinitialize()
            mca.set_clm_output_targets(tile, [0] * N_LANES, [power_target] * N_LANES)
            time.sleep(settle_s)
            frames.append(
                record_condition(
                    tile, all_lanes, sample_s, interval_s, pow_tgt=power_target
                )
            )
            frames.append(
                record_condition(
                    tile, one_lane, sample_s, interval_s, pow_tgt=power_target
                )
            )
    except KeyboardInterrupt:
        print("Interrupted. Saving samples collected so far.")
        return _finish_frames(frames, data_dir, stem + "_partial", params)
    except Exception:
        print("Saving samples collected before the error.")
        _finish_frames(frames, data_dir, stem + "_partial", params)
        raise
    return _finish_frames(frames, data_dir, stem, params)


def open_ptc(cfg):
    """Open the first PTC1 in USB mode and set its target temperature."""
    sdk = Path(cfg["thorlabs_sdk"])
    sdk_dir = str(sdk)
    if sdk_dir not in sys.path:
        sys.path.append(sdk_dir)
    os.add_dll_directory(sdk_dir)
    import PTC1_COMMAND_LIB as ptc1_module

    # Thorlabs loads ./PTC_COMMAND_LIB_x64.dll from the working directory.
    # Load the SDK copy by full path first so that relative lookup is skipped.
    dll_name = "PTC_COMMAND_LIB_x64.dll" if sys.maxsize > 2**32 else "PTC_COMMAND_LIB_x86.dll"
    dll_path = str(sdk / dll_name)
    if not ptc1_module.PTC1.isLoad:
        ptc1_module.PTC1.load_library(dll_path)
    tec_thorlab = ptc1_module.PTC1()
    devices = ptc1_module.PTC1.list_devices()
    print(devices)
    if len(devices) <= 0:
        raise RuntimeError("No PTC1 devices connected")
    serial_number = devices[0][0]
    print("connect", serial_number)
    handle = tec_thorlab.open(serial_number, cfg["tec_thorlab"]["baud"], cfg["tec_thorlab"]["timeout"])
    if handle < 0 or tec_thorlab.is_open(serial_number) == 0:
        tec_thorlab.close()
        raise RuntimeError(f"Failed to open PTC1 {serial_number}")
    mode = [0]
    tec_thorlab.get_mode(mode)
    if mode[0] != 0:
        raise RuntimeError("PTC1 is not in USB mode")
    print("... nice ...")
    tec_thorlab.set_target_temperature(int(cfg["tec_thorlab"]["target_mC"]))
    return tec_thorlab


def open_per_meter(cfg):
    _scripts_on_path()
    import oz_er1000

    return oz_er1000.ER1000(cfg["per_meter_port"])


def open_arroyo(cfg):
    """Open the Arroyo TEC on the configured COM port and print its status.

    The setpoint and the output are left as they are.
    """
    from instrument_drivers.devices.arroyo_temp_controller import ArroyoTempController

    tec_arroyo = ArroyoTempController(cfg["arroyo_port"])
    print(
        f"mode {tec_arroyo.get_mode()}, "
        f"temp {tec_arroyo.get_temp()} C, "
        f"setpoint {tec_arroyo.get_set_temp()} C, "
        f"current {tec_arroyo.get_curr()} A, "
        f"output {tec_arroyo.get_output()}"
    )
    return tec_arroyo


def set_temperature(tec_thorlab, tec_arroyo, temp_c):
    """Set the same target, in degrees C, on both TECs."""
    tec_thorlab.set_target_temperature(int(round(1000 * temp_c)))
    tec_arroyo.set_temp(float(temp_c))


def read_tec_state(tec_thorlab, tec_arroyo):
    """Read both TECs. Thorlabs temperature is returned in degrees C."""
    temp_mC = [0]
    curr_thorlab = [0]
    volt_thorlab = [0]
    tec_thorlab.get_actual_temperature(temp_mC)
    tec_thorlab.get_actual_current(curr_thorlab)
    tec_thorlab.get_actual_voltage(volt_thorlab)
    temp_thorlab = temp_mC[0] / 1000.0
    return {
        "temp": temp_thorlab,
        "temp_thorlab": temp_thorlab,
        "curr_thorlab": curr_thorlab[0]/1000,
        "volt_thorlab": volt_thorlab[0]/1000,
        "temp_arroyo": tec_arroyo.get_temp(),
        "curr_arroyo": tec_arroyo.get_curr(),
        "volt_arroyo": tec_arroyo.get_volt(),
    }


def _now():
    return datetime.now().isoformat(timespec="seconds")


class _RunWriter:
    """Append rows to a CSV, creating the header on the first write.

    constants, such as note, are written on every row.
    """

    def __init__(self, path, columns, constants=None):
        self.path = Path(path)
        self.constants = dict(constants or {})
        self.columns = list(columns)
        for key in self.constants:
            if key not in self.columns:
                self.columns.append(key)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._header_written = False

    def append(self, rows):
        if not rows:
            return
        if self.constants:
            for row in rows:
                row.update(self.constants)
        frame = pd.DataFrame(rows).reindex(columns=self.columns)
        frame.to_csv(
            self.path,
            mode="a",
            header=not self._header_written,
            index=False,
        )
        self._header_written = True


def _new_csv_path(data_dir, stem):
    stamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    return Path(data_dir) / f"{stem}_{stamp}.csv"


def _load_saved(path, extra_columns=None):
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    frame = pd.read_csv(path)
    if extra_columns:
        for name, value in extra_columns.items():
            if name not in frame.columns:
                frame[name] = value
    if "et_s" in frame.columns and "et_min" not in frame.columns:
        frame["et_min"] = frame["et_s"] / 60
    return frame


def default_per_temperature_targets():
    """32 C to 35 C, then back down, offset by 0.125 C so 35 C is not repeated."""
    going_up = np.linspace(30, 33, 13)
    going_down = np.linspace(33 - 0.125, 30 - 0.125, 13)
    return np.concatenate((going_up, going_down))


def per_readings(per_meter, duration_s=5.0):
    """Return PER measurements as dicts. Skip a reading that is not 3 numbers."""
    if hasattr(per_meter, "measure_per"):
        measurements = per_meter.measure_per(duration_s)
    else:
        measurements = per_meter.measurePER()
    rows = []
    for measurement in measurements:
        if len(measurement) != len(PER_FIELDS):
            print(f"Skipping PER reading with {len(measurement)} fields: {measurement}")
            continue
        rows.append(dict(zip(PER_FIELDS, measurement)))
    return rows


def per_vs_temperature(
    tile,
    tec_thorlab,
    tec_arroyo,
    per_meter,
    data_dir,
    bank=0,
    channels=range(LANES_PER_BANK),
    temp_targets=None,
    stabilize_s=300,
    channel_settle_s=30,
    sample_s=120,
    sample_interval_s=10,
    tile_index=None,
    update_period_s=None,
    note="",
):
    """Dwell at each temperature, then record PER on each channel of one bank.

    Both TECs are set to the same target. Samples are appended to the CSV
    as they are taken.
    """
    if temp_targets is None:
        temp_targets = default_per_temperature_targets()
    channels = list(channels)
    columns = [
        "time", "et_s", "et_s_1", "et_s_2", "temp_tgt",
        "temp", "temp_thorlab", "temp_arroyo",
        "curr_thorlab", "volt_thorlab", "curr_arroyo", "volt_arroyo",
        "bank", "ch", "tile_index", "per_dB", "theta", "pow",
    ]
    stem = (f"per over bank{bank} chs over temp")

    path = _new_csv_path(data_dir, stem)
    writer = _RunWriter(path, columns, {"note": note})
    print(f"Writing {path}")
    t0 = time.time()
    try:
        for temp_target in tqdm(temp_targets, desc="temp ramp"):
            set_temperature(tec_thorlab, tec_arroyo, temp_target)
            t_temp = time.time()
            time.sleep(stabilize_s)
            for channel in tqdm(channels, desc="channel"):
                mca.clm_lane_enable(tile, chEnableMask=lane_mask([bank * LANES_PER_BANK + channel]))
                time.sleep(channel_settle_s)
                t_channel = time.time()
                elapsed_channel = 0
                while elapsed_channel <= sample_s:
                    tec_state = read_tec_state(tec_thorlab, tec_arroyo)
                    elapsed = round(time.time() - t0, 1)
                    elapsed_temp = round(time.time() - t_temp, 1)
                    elapsed_channel = round(time.time() - t_channel, 1)
                    for reading in per_readings(per_meter):
                        row = {
                            "time": _now(),
                            "et_s": elapsed,
                            "et_s_1": elapsed_temp,
                            "et_s_2": elapsed_channel,
                            "temp_tgt": temp_target,
                            "ch": channel,
                            "bank": bank,
                            "tile_index": tile_index,
                        }
                        row.update(tec_state)
                        row.update(reading)
                        writer.append([row])
                        print(elapsed, channel, reading)
                    time.sleep(sample_interval_s)
    except KeyboardInterrupt:
        print(f"Interrupted. Partial data: {path}")
    except Exception:
        print(f"Partial data: {path}")
        raise
    return _load_saved(path)


def per_over_temp_2(
    tile,
    tec_thorlab,
    tec_arroyo,
    per_meter,
    data_dir,
    bank=0,
    channels=range(LANES_PER_BANK),
    temp_targets=None,
    channel_s=120,
    channel_settle_s=30,
    tile_index=None,
    note="",
):
    """Step both TECs through a temperature profile and record PER the whole time.

    The ER-1000 from oz_er1000 is started once and left running. There is no
    wait for the TEC reading to settle. After a lane is enabled, the test
    waits channel_settle_s seconds, drops meter lines from that wait, then
    records for channel_s seconds. et_0 is time since the test started,
    et_tecAdj since the setpoint write, and et_chEnb since that lane was
    enabled, so the first saved sample is about channel_settle_s after enable.
    """
    if temp_targets is None:
        temp_targets = default_per_temperature_targets()
    channels = list(channels)
    if channel_s <= 0:
        raise ValueError("channel_s must be positive")
    if channel_settle_s < 0:
        raise ValueError("channel_settle_s must be zero or positive")
    columns = [
        "time", "et_0", "et_chEnb", "et_tecAdj", "temp_tgt",
        "temp", "temp_thorlab", "temp_arroyo",
        "curr_thorlab", "volt_thorlab", "curr_arroyo", "volt_arroyo",
        "bank", "ch", "tile_index", "per_dB", "theta", "pow",
    ]
    path = _new_csv_path(data_dir, f"per over temp 2 bank{bank}")
    writer = _RunWriter(path, columns, {"note": note})
    print(f"Writing {path}")
    t0 = time.time()
    started = False
    try:
        per_meter.stop(flush=True)
        per_meter.set_mode("er")
        per_meter.set_capture(True)
        for temp_target in tqdm(list(temp_targets), desc="temp ramp"):
            set_temperature(tec_thorlab, tec_arroyo, temp_target)
            t_tec = time.time()
            for channel in tqdm(channels, desc="channel"):
                mca.clm_lane_enable(
                    tile, chEnableMask=lane_mask([bank * LANES_PER_BANK + channel])
                )
                t_ch = time.time()
                settle_deadline = time.monotonic() + channel_settle_s
                while time.monotonic() < settle_deadline:
                    time.sleep(min(0.5, settle_deadline - time.monotonic()))
                    if started:
                        per_meter._discard_input()
                if not started:
                    per_meter.start()
                    started = True
                for reading in per_meter.iter_er(channel_s):
                    now = time.time()
                    tec_state = read_tec_state(tec_thorlab, tec_arroyo)
                    row = {
                        "time": _now(),
                        "et_0": round(now - t0, 1),
                        "et_chEnb": round(now - t_ch, 1),
                        "et_tecAdj": round(now - t_tec, 1),
                        "temp_tgt": temp_target,
                        "ch": channel,
                        "bank": bank,
                        "tile_index": tile_index,
                        "per_dB": reading.per_db,
                        "theta": reading.angle_deg,
                        "pow": reading.power_dbm,
                    }
                    row.update(tec_state)
                    writer.append([row])
                    print(row["et_0"], channel, temp_target, reading.per_db)
    except KeyboardInterrupt:
        print(f"Interrupted. Partial data: {path}")
    except Exception:
        print(f"Partial data: {path}")
        raise
    finally:
        if started:
            per_meter.stop(flush=True)
    frame = _load_saved(path)
    if len(frame) and "et_0" in frame.columns:
        frame["et_s"] = frame["et_0"]
        frame["et_min"] = frame["et_0"] / 60
    return frame


def triangle_setpoint(elapsed_s, duration_s, temp0, temp1):
    """Return the ramp command: temp0 to temp1, then back to temp0.

    The first half of duration_s is the rise. The second half is the fall.
    """
    half = duration_s / 2.0
    if elapsed_s <= 0 or elapsed_s >= duration_s:
        return float(temp0)
    if elapsed_s <= half:
        return float(temp0 + (temp1 - temp0) * (elapsed_s / half))
    return float(temp1 + (temp0 - temp1) * ((elapsed_s - half) / half))


def per_over_temp_3(
    tile,
    tec_thorlab,
    tec_arroyo,
    per_meter,
    data_dir,
    bank=0,
    channels=range(LANES_PER_BANK),
    temp0=30,
    temp1=35,
    channel_s=120,
    channel_settle_s=30,
    ramp_step_s=1.0,
    tile_index=None,
    note="",
):
    """Enable one channel at a time and ramp both TECs while recording PER.

    The TECs are commanded to temp0 before the channels start. After a lane
    is enabled, the test waits channel_settle_s seconds and does not save that
    wait. It then ramps both setpoints together from temp0 to temp1 over the
    first half of channel_s and back to temp0 over the second half. The
    setpoint is rewritten every ramp_step_s seconds from the ramp clock, and
    PER plus the measured TEC temperatures are recorded between those writes.
    et_0 is time since the test started, et_chEnb since the lane was enabled,
    and et_tecAdj since this channel's ramp started.
    """
    channels = list(channels)
    if channel_s <= 0:
        raise ValueError("channel_s must be positive")
    if channel_settle_s < 0:
        raise ValueError("channel_settle_s must be zero or positive")
    if ramp_step_s <= 0 or ramp_step_s > channel_s / 2:
        raise ValueError("ramp_step_s must be positive and no more than channel_s / 2")
    columns = [
        "time", "et_0", "et_chEnb", "et_tecAdj", "temp_tgt",
        "temp", "temp_thorlab", "temp_arroyo",
        "curr_thorlab", "volt_thorlab", "curr_arroyo", "volt_arroyo",
        "bank", "ch", "tile_index", "per_dB", "theta", "pow",
    ]
    path = _new_csv_path(data_dir, f"per over temp 3 bank{bank}")
    writer = _RunWriter(path, columns, {"note": note})
    print(f"Writing {path}")
    t0 = time.time()
    started = False
    try:
        per_meter.stop(flush=True)
        per_meter.set_mode("er")
        per_meter.set_capture(True)
        set_temperature(tec_thorlab, tec_arroyo, temp0)
        for channel in tqdm(channels, desc="channel"):
            mca.clm_lane_enable(
                tile, chEnableMask=lane_mask([bank * LANES_PER_BANK + channel])
            )
            t_ch = time.time()
            settle_deadline = time.monotonic() + channel_settle_s
            while time.monotonic() < settle_deadline:
                time.sleep(min(0.5, settle_deadline - time.monotonic()))
                if started:
                    per_meter._discard_input()
            if not started:
                per_meter.start()
                started = True
            ramp_mark = time.monotonic()
            t_tec = time.time()
            next_write = ramp_mark
            commanded = float(temp0)
            while True:
                elapsed = time.monotonic() - ramp_mark
                if elapsed >= channel_s:
                    break
                if time.monotonic() >= next_write:
                    commanded = triangle_setpoint(elapsed, channel_s, temp0, temp1)
                    set_temperature(tec_thorlab, tec_arroyo, commanded)
                    next_write = time.monotonic() + ramp_step_s
                remain = channel_s - (time.monotonic() - ramp_mark)
                slice_s = min(next_write - time.monotonic(), remain)
                if slice_s <= 0:
                    continue
                for reading in per_meter.iter_er(slice_s):
                    now = time.time()
                    tec_state = read_tec_state(tec_thorlab, tec_arroyo)
                    row = {
                        "time": _now(),
                        "et_0": round(now - t0, 1),
                        "et_chEnb": round(now - t_ch, 1),
                        "et_tecAdj": round(now - t_tec, 1),
                        "temp_tgt": commanded,
                        "ch": channel,
                        "bank": bank,
                        "tile_index": tile_index,
                        "per_dB": reading.per_db,
                        "theta": reading.angle_deg,
                        "pow": reading.power_dbm,
                    }
                    row.update(tec_state)
                    writer.append([row])
                    print(
                        row["et_0"],
                        channel,
                        round(commanded, 3),
                        round(tec_state["temp_thorlab"], 3),
                        round(tec_state["temp_arroyo"], 3),
                        reading.per_db,
                    )
            set_temperature(tec_thorlab, tec_arroyo, temp0)
    except KeyboardInterrupt:
        print(f"Interrupted. Partial data: {path}")
    except Exception:
        print(f"Partial data: {path}")
        raise
    finally:
        if started:
            per_meter.stop(flush=True)
    frame = _load_saved(path)
    if len(frame) and "et_0" in frame.columns:
        frame["et_s"] = frame["et_0"]
        frame["et_min"] = frame["et_0"] / 60
    return frame


def per_vs_frequency_offset(
    tile,
    tec_thorlab,
    tec_arroyo,
    per_meter,
    data_dir,
    bank=0,
    channels=range(LANES_PER_BANK),
    freq_offsets=(-20, -15, -10, -5, 0, 5, 10, 15, 20),
    channel_settle_s=30,
    sample_s=120,
    sample_interval_s=10,
    tile_index=None,
    note="",
):
    """Record PER versus laser frequency offset at the current temperature."""
    channels = list(channels)
    columns = [
        "time", "et_s", "et_s_1", "temp_tgt",
        "temp", "temp_thorlab", "temp_arroyo",
        "curr_thorlab", "volt_thorlab", "curr_arroyo", "volt_arroyo",
        "bank", "ch", "freq_offset", "tile_index", "per_dB", "theta", "pow",
    ]
    path = _new_csv_path(data_dir, "per over freq offset over chs")
    writer = _RunWriter(path, columns, {"note": note})
    print(f"Writing {path}")
    target_mC = [0]
    tec_thorlab.get_target_temperature(target_mC)
    temp_target = target_mC[0] / 1000.0
    bank_enum = bank_for(bank)
    t0 = time.time()
    try:
        for offset in tqdm(freq_offsets, desc="freq_offset"):
            tile.write_lane_frequency_setpoints(bank_enum, [offset] * LANES_PER_BANK)
            for channel in tqdm(channels, desc="ch"):
                mca.clm_lane_enable(
                    tile, chEnableMask=lane_mask([bank * LANES_PER_BANK + channel])
                )
                time.sleep(channel_settle_s)
                t_channel = time.time()
                elapsed_channel = 0
                while elapsed_channel <= sample_s:
                    tec_state = read_tec_state(tec_thorlab, tec_arroyo)
                    elapsed = round(time.time() - t0, 1)
                    elapsed_channel = round(time.time() - t_channel, 1)
                    for reading in per_readings(per_meter):
                        row = {
                            "time": _now(),
                            "et_s": elapsed,
                            "et_s_1": elapsed_channel,
                            "temp_tgt": temp_target,
                            "ch": channel,
                            "freq_offset": offset,
                            "bank": bank,
                            "tile_index": tile_index,
                        }
                        row.update(tec_state)
                        row.update(reading)
                        writer.append([row])
                        print(elapsed, channel, offset, reading)
                    time.sleep(sample_interval_s)
    except KeyboardInterrupt:
        print(f"Interrupted. Partial data: {path}")
    except Exception:
        print(f"Partial data: {path}")
        raise
    return _load_saved(path)


def per_passive(per_meter, tec_thorlab, tec_arroyo, data_dir, period_s=10, note=""):
    """Record PER and both TEC temperatures until KeyboardInterrupt."""
    columns = [
        "time", "et_s", "per_dB", "theta", "pow",
        "temp", "temp_thorlab", "temp_arroyo",
        "curr_thorlab", "volt_thorlab", "curr_arroyo", "volt_arroyo",
    ]
    path = _new_csv_path(data_dir, "per_passive")
    writer = _RunWriter(path, columns, {"note": note})
    print(f"Writing {path}")
    t0 = time.time()
    try:
        while True:
            tec_state = read_tec_state(tec_thorlab, tec_arroyo)
            elapsed = round(time.time() - t0, 1)
            for reading in per_readings(per_meter):
                row = {"time": _now(), "et_s": elapsed}
                row.update(reading)
                row.update(tec_state)
                writer.append([row])
                print(elapsed, reading, tec_state["temp_thorlab"], tec_state["temp_arroyo"])
            time.sleep(period_s)
    except KeyboardInterrupt:
        print(f"Interrupted. Saved {path}")
    except Exception:
        print(f"Partial data: {path}")
        raise
    return _load_saved(path)


def _dot(value):
    return str(value).replace(".", "p")
