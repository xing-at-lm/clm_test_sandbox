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
from three_dB_coupler_T191134 import coupler_insertion_loss_db as c3db_loss

N_LANES = 16
LANES_PER_BANK = 8
PER_FIELDS = ("per_dB", "theta", "pow")

# Interleaved CLM grid from clm_tile_control_kurtz.py. Even entries are bank A.
WAVELENGTH_GRID_NM = np.array([
    1301.47, 1302.60, 1303.73, 1304.87,
    1306.01, 1307.14, 1308.28, 1309.43,
    1310.57, 1311.72, 1312.87, 1314.02,
    1315.17, 1316.33, 1317.48, 1318.64,
])

WAVELENGTH_GRID_NM_A=WAVELENGTH_GRID_NM[0::2]
WAVELENGTH_GRID_NM_B=WAVELENGTH_GRID_NM[1::2]

def load_config(path=None):
    """Load lab_config.yaml from scripts/, or from path."""
    if path is None:
        path = Path(__file__).resolve().parent / "lab_config.yaml"
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_test_spec(path=None):
    """Load clm_test_spec.yaml from scripts/, or from path."""
    if path is None:
        path = Path(__file__).resolve().parent / "clm_test_spec.yaml"
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def clm_fw_update(tile, spec=None, note=""):
    """Flash firmware named in clm_test_spec.yaml onto one tile.

    Uses ``tile.firmware_update`` (per-tile, not board-wide).
    ``spec`` may be the loaded YAML dict, or None to load clm_test_spec.yaml.
    Returns the absolute path of the binary that was programmed.
    """
    if tile is None:
        raise ValueError("tile handle is required for firmware update")
    if spec is None:
        spec = load_test_spec()
    firmware = spec.get("firmware") if isinstance(spec, dict) else None
    if not firmware or "dir" not in firmware or "filename" not in firmware:
        raise ValueError("clm_test_spec.yaml must define firmware.dir and firmware.filename")
    fw_dir = Path(firmware["dir"])
    fw_name = firmware["filename"]
    fw_path = fw_dir / fw_name
    if not fw_path.is_file():
        raise FileNotFoundError(f"Firmware binary not found: {fw_path}")
    print(f"Updating tile firmware from {fw_path}" + (f" ({note})" if note else ""))
    tile.firmware_update(str(fw_path))
    print(f"Firmware update finished: {fw_name}")
    return fw_path


def _scripts_on_path():
    scripts_dir = str(Path(__file__).resolve().parent)
    if scripts_dir not in sys.path:
        sys.path.append(scripts_dir)


def _tile_inventory(tb):
    """Return list of (tile_index, tile) for non-None package-0 tiles."""
    package = tb.module.packages[0]
    return [
        (idx, tile)
        for idx, tile in enumerate(package.tiles)
        if tile is not None
    ]


def reopen_package_tiles(tb, log=print):
    """Reopen package MCI then every present tile (required after reset / bus errors)."""
    package = tb.module.packages[0]
    log("Reopening package MCI + all present tiles...")
    package.reopen()
    time.sleep(0.3)


def probe_tile_link(tile, label="tile", log=print):
    """True if tile answers read_fw_version."""
    try:
        version = tile.read_fw_version()
        log(f"{label} reachable, FW version: {version}")
        return True
    except Exception as exc:
        log(f"{label} probe failed: {exc}")
        return False


def ensure_tile_ready(tb, tile_index, log=print):
    """Return a live tile handle for tile_index, or raise with inventory details."""
    package = tb.module.packages[0]
    present = _tile_inventory(tb)
    log(
        "Present tiles at connect: "
        + (
            ", ".join(
                f"{idx}(addr={getattr(tile, 'dev_addr', None)})"
                for idx, tile in present
            )
            or "(none)"
        )
    )
    if tile_index is None:
        raise ValueError("tile_index is required")
    tile_index = int(tile_index)
    if tile_index < 0 or tile_index >= len(package.tiles):
        raise IndexError(f"tile_index {tile_index} out of range for package.tiles")

    tile = package.tiles[tile_index]
    if tile is None:
        raise RuntimeError(
            f"tiles[{tile_index}] is empty (no CLM at that address slot). "
            f"Present indices: {[idx for idx, _ in present]}"
        )

    reopen_package_tiles(tb, log=log)
    tile = package.tiles[tile_index]
    if probe_tile_link(tile, label=f"tile {tile_index}", log=log):
        return tile

    log(f"tile {tile_index} not reachable; hardware-reset all tiles and reopen...")
    tb.reset_tiles(0xFF)
    time.sleep(0.5)
    reopen_package_tiles(tb, log=log)
    tile = package.tiles[tile_index]
    if tile is not None and probe_tile_link(tile, label=f"tile {tile_index}", log=log):
        return tile

    raise RuntimeError(
        f"tile {tile_index} not responding after package reopen / full reset. "
        f"Present slots: {[idx for idx, _ in _tile_inventory(tb)]}. "
        "Confirm DUT is seated, EVK regulators are on, and only one GUI holds the Aardvark."
    )


def connect_testboard(cfg, log=print):
    """Open the testboard named in cfg and return (testboard, tile)."""
    tb = mca.get_testboard(
        AARDVARK_SN_P0=cfg["aardvark"]["p0"],
        AARDVARK_SN_Io=cfg["aardvark"]["io"],
        AARDVARK_SN_P1=cfg["aardvark"]["p1"],
        testBoardType=cfg["test_board_type"],
    )
    tile_index = int(cfg.get("tile_index", 3))
    tile = ensure_tile_ready(tb, tile_index, log=log)
    return tb, tile


def _open_station_testboard(cfg):
    log = cfg.get("_station_log", print)
    tb, tile = connect_testboard(cfg, log=log)
    return {"tb": tb, "tile": tile}


def _close_station_testboard(handle):
    tb = handle.get("tb") if isinstance(handle, dict) else None
    if tb is not None:
        mca.shut_down_test_board_nicely(tb)


def _close_station_instrument(obj):
    if obj is None:
        return
    close = getattr(obj, "close", None)
    if callable(close):
        close()


def _station_handlers():
    """Open/close pairs for lab_config.test_station.instruments.

    Add a new instrument here, then list its name under test_station in YAML.
    """
    return {
        "testboard": (_open_station_testboard, _close_station_testboard),
        "tec_thorlab": (open_ptc, _close_station_instrument),
        "tec_arroyo": (open_arroyo, _close_station_instrument),
        "per_meter": (open_per_meter, _close_station_instrument),
        "per_meter_er100": (open_per_meter_er100, _close_station_instrument),
        "osa": (open_osa, _close_station_instrument),
        "opm": (open_opm, _close_opm),
    }


def clm_start_test_station(cfg, log=print):
    """Open every instrument listed under cfg['test_station']['instruments'].

    Returns a dict name -> open handle for instruments that opened successfully.
    A failure on one instrument is logged and skipped so the others stay usable.
    Raises only when the instrument list is missing, or when every open fails.
    """
    handlers = _station_handlers()
    names = list(cfg.get("test_station", {}).get("instruments", []))
    if not names:
        raise ValueError("lab_config.yaml has no test_station.instruments list")
    log(f"Starting test station: {', '.join(names)}")
    # So connect_testboard / ensure_tile_ready messages appear in the GUI log.
    cfg = dict(cfg)
    cfg["_station_log"] = log
    instruments = {}
    failures = {}
    for name in names:
        if name not in handlers:
            failures[name] = KeyError(
                f"Unknown instrument {name!r}. Register it in _station_handlers()."
            )
            log(f"  {name} failed: {failures[name]}")
            continue
        open_fn, _ = handlers[name]
        log(f"  opening {name} ...")
        try:
            instruments[name] = open_fn(cfg)
            log(f"  {name} ready")
        except Exception as exc:
            failures[name] = exc
            log(f"  {name} failed: {exc}")
    if not instruments:
        detail = "; ".join(f"{name}: {exc}" for name, exc in failures.items())
        raise RuntimeError(f"No test-station instruments opened. {detail}")
    if failures:
        log(
            f"Test station partial start. Ready: {', '.join(instruments)}. "
            f"Failed: {', '.join(failures)}."
        )
    else:
        log("Test station started.")
    return instruments


def clm_close_test_station(instruments, log=print):
    """Close every instrument in the dict, newest first.

    ``instruments`` is the dict returned by clm_start_test_station. Entries
    are removed from the dict as they are closed.
    """
    if not instruments:
        log("Test station is already closed.")
        return
    handlers = _station_handlers()
    names = list(reversed(list(instruments.keys())))
    log(f"Closing test station: {', '.join(names)}")
    for name in names:
        handle = instruments.pop(name, None)
        if handle is None:
            continue
        _, close_fn = handlers.get(name, (None, _close_station_instrument))
        try:
            close_fn(handle)
            log(f"  closed {name}")
        except Exception as exc:
            log(f"  warning closing {name}: {exc}")
    log("Test station closed.")


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


def record_condition(tile, enable_mask, sample_s, interval_s, tweakT_PIC=False, **tags):
    """Enable lanes, record, and attach the mask plus any extra columns.

    tweakT_PIC is forwarded to get_clm_record. It is not a tag column.
    """
    mca.clm_lane_enable(tile, chEnableMask=list(enable_mask))
    records, columns = mca.get_clm_record(
        [tile],
        duration_s=sample_s,
        interval_s=interval_s,
        tweakT_PIC=tweakT_PIC,
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


def power_level_sweep(
    tile,
    data_dir,
    power_targets=(6, 8, 10, 12),
    duration_s=300,
    interval_s=2,
    tweakT_PIC=False,
    tile_index=None,
    note="",
):
    """Hold all 16 lanes at each power target for duration_s.

    Each dwell is sampled by get_clm_record at interval_s. tweakT_PIC is
    passed through to that call. et_run_s is elapsed time across the whole
    sweep; et_s restarts inside each dwell.
    """
    power_targets = list(power_targets)
    all_lanes = [1] * N_LANES
    stem = (
        f"clm power level sweep_duration-{duration_s}_interval-{interval_s}"
        f"_tweakTpic-{int(bool(tweakT_PIC))}"
    )
    params = {
        "duration_s": duration_s,
        "interval_s": interval_s,
        "tweakT_PIC": bool(tweakT_PIC),
        "note": note,
    }
    if tile_index is not None:
        params["tile_index"] = tile_index
    frames = []
    offset = 0.0
    try:
        for power_target in tqdm(power_targets, desc="power tgt"):
            mca.set_clm_output_targets(tile, [0] * N_LANES, [power_target] * N_LANES)
            frame = record_condition(
                tile,
                all_lanes,
                duration_s,
                interval_s,
                tweakT_PIC=tweakT_PIC,
                pow_tgt=power_target,
            )
            frame["et_run_s"] = frame["et_s"] + offset
            if len(frame):
                offset = float(frame["et_run_s"].max())
            frames.append(frame)
    except KeyboardInterrupt:
        print("Interrupted. Saving samples collected so far.")
        return _finish_frames(frames, data_dir, stem + "_partial", params)
    except Exception:
        print("Saving samples collected before the error.")
        _finish_frames(frames, data_dir, stem + "_partial", params)
        raise
    return _finish_frames(frames, data_dir, stem, params)


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


def open_per_meter_er100(cfg):
    """Open the OZ Optics ER-100 at the configured COM port."""
    _scripts_on_path()
    import oz_er100

    return oz_er100.ER100(cfg["per_meter_er100_port"])


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


def open_osa(cfg):
    """Open the Yokogawa OSA at the configured VISA address."""
    from instrument_drivers.devices.yokogawa_osa import YokogawaOSA

    return YokogawaOSA(cfg["osa_visa"])


def open_opm(cfg):
    """Open the Keysight N7745C optical power meter at cfg['opm_visa']."""
    from instrument_drivers.devices.keysight_N7745c_opm import KeysightN7745C

    opm = KeysightN7745C(cfg["opm_visa"])
    print(opm.connect().strip())
    return opm


def _close_opm(opm):
    if opm is None:
        return
    disconnect = getattr(opm, "disconnect", None)
    if callable(disconnect):
        disconnect()
        return
    _close_station_instrument(opm)


def set_temperature(tec_thorlab, tec_arroyo, temp_c):
    """Set the same target, in degrees C, on both TECs."""
    tec_thorlab.set_target_temperature(int(round(1000 * temp_c)))
    tec_arroyo.set_temp(float(temp_c))


def set_temperature_arroyo(tec_arroyo, temp_c):
    """Set Arroyo TEC target only (degrees C)."""
    tec_arroyo.set_temp(float(temp_c))


def _arroyo_flush_input(tec_arroyo):
    """Drop unread serial bytes so the next query is not a stale reply.

    The Arroyo port is non-blocking. ``TEC:OUT 1`` can leave an echo or the
    previous ``0`` in the buffer, and the driver's immediate readback then
    reports the output as still off.
    """
    com = getattr(tec_arroyo, "com", None)
    if com is None:
        return
    try:
        com.reset_input_buffer()
    except Exception:
        pass


def _arroyo_write(tec_arroyo, command):
    """Send one command terminated by CR/LF, with no trailing space.

    ``ArroyoInstrument.write`` appends `` \\r\\n``. That extra space makes an
    integer argument such as ``TEC:OUT 1`` fail with E-202 Invalid data type.
    """
    payload = str(command).strip() + "\r\n"
    _arroyo_flush_input(tec_arroyo)
    tec_arroyo.com.write(payload.encode("ascii"))
    time.sleep(0.15)


def _arroyo_query_text(tec_arroyo, command):
    _arroyo_flush_input(tec_arroyo)
    return str(tec_arroyo.query(command)).strip()


def _arroyo_last_int(text):
    """Last integer in a reply. An echoed command must not hide the value."""
    numbers = []
    for piece in str(text).replace(",", " ").split():
        token = piece.strip().strip('"')
        try:
            numbers.append(int(float(token)))
        except ValueError:
            continue
    if not numbers:
        raise ValueError(f"no integer in Arroyo response {text!r}")
    return numbers[-1]


def _arroyo_output_state(tec_arroyo):
    text = _arroyo_query_text(tec_arroyo, "TEC:OUT?")
    return _arroyo_last_int(text), text


def _arroyo_error_text(tec_arroyo):
    try:
        text = _arroyo_query_text(tec_arroyo, "ERRSTR?")
    except Exception as exc:
        return f"ERRSTR? failed: {exc}"
    if not text or text.strip() in ("0", "0,0"):
        return ""
    return text.strip()


def ensure_arroyo_tec_enabled(tec_arroyo, log=print):
    """Turn the Arroyo TEC output on so the control loop tracks the setpoint.

    ``TEC:T`` only stores the setpoint. With ``TEC:OUT`` off, the commanded
    temperature moves and the measured temperature stays put. The status is
    read after the serial input buffer is cleared, because a stale ``0``
    left by the enable command otherwise looks like a failed enable.
    """
    output, raw = _arroyo_output_state(tec_arroyo)
    if output == 1:
        log("Arroyo TEC control loop is enabled.")
        return
    try:
        mode = _arroyo_query_text(tec_arroyo, "TEC:MODE?")
    except Exception:
        mode = ""
    if mode.split(",")[0].strip().upper() not in ("T", "TEMP", "TEMPERATURE"):
        log(f"Arroyo TEC mode is {mode or 'unknown'}; setting temperature mode.")
        _arroyo_write(tec_arroyo, "TEC:MODE:T")
    # Drop E-202 left by an earlier "TEC:OUT 1 " (trailing space) command.
    _arroyo_error_text(tec_arroyo)
    log(f"Arroyo TEC output is {raw}; enabling the control loop.")
    last_detail = raw
    for attempt in range(1, 4):
        _arroyo_write(tec_arroyo, "TEC:OUT 1")
        time.sleep(0.25)
        try:
            output, raw = _arroyo_output_state(tec_arroyo)
        except Exception as exc:
            last_detail = str(exc)
            log(f"Arroyo TEC output readback failed on attempt {attempt}: {exc}")
            continue
        if output == 1:
            log("Arroyo TEC control loop is enabled.")
            return
        error_text = _arroyo_error_text(tec_arroyo)
        last_detail = f"TEC:OUT? {raw}"
        if error_text:
            last_detail = f"{last_detail}; {error_text}"
        log(f"Arroyo TEC output still off after attempt {attempt}: {last_detail}")
    raise RuntimeError(
        "Arroyo TEC control loop is not enabled (TEC:OUT). "
        f"{last_detail}. "
        "The setpoint can change, but measured temperature will not follow."
    )


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


def _finite(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return number == number and number not in (float("inf"), float("-inf"))


def _offset_setpoint(current, delta, low, high):
    """Pick current+delta, or current-delta if that stays inside low..high."""
    trial = current + delta
    if trial < low or trial > high:
        trial = current - delta
    if trial < low or trial > high or trial == current:
        raise ValueError(
            f"setpoint {current} has no room for a {delta} step inside {low}..{high}"
        )
    return trial


class _SelfTest:
    """Collect pass/fail lines for one instrument and echo them to the log."""

    def __init__(self, device, log=print):
        self.device = device
        self.log = log
        self.checks = []

    def add(self, name, passed, detail):
        self.checks.append({
            "name": name,
            "passed": bool(passed),
            "detail": str(detail),
        })
        mark = "PASS" if passed else "FAIL"
        self.log(f"  [{mark}] {self.device} {name}: {detail}")

    def result(self):
        passed = bool(self.checks) and all(item["passed"] for item in self.checks)
        mark = "PASS" if passed else "FAIL"
        self.log(f"{self.device} self test {mark}")
        return {"device": self.device, "passed": passed, "checks": self.checks}


def _ptc_call(tec_thorlab, method, label):
    """Call a PTC1 getter that fills a one-element list. Raises on a bad status."""
    slot = [0]
    status = method(slot)
    if status != 0:
        raise RuntimeError(f"{label} failed with status {status}")
    return slot[0]


def self_test_thorlab(tec_thorlab, log=print, delta_c=1.0):
    """Check the Thorlabs PTC1 setpoint, sensor, and loop telemetry.

    Writes a 1 C trial target, reads it back, then restores the original
    target. The PTC1 SDK has no TEC output-enable command, so loop status
    is USB mode, a clear error register, and a live current and voltage read.
    """
    report = _SelfTest("Thorlabs TEC", log)
    original_mc = None
    try:
        ident = _ptc_call(tec_thorlab, tec_thorlab.get_id, "identity")
        report.add("identity", bool(str(ident).strip()), str(ident).strip() or "empty identity")
    except Exception as exc:
        report.add("identity", False, exc)
        return report.result()

    try:
        mode = int(_ptc_call(tec_thorlab, tec_thorlab.get_mode, "mode"))
        report.add(
            "loop status",
            mode == 0,
            "USB mode" if mode == 0 else f"knob mode ({mode}); remote control is not active",
        )
    except Exception as exc:
        report.add("loop status", False, exc)

    try:
        error = int(_ptc_call(tec_thorlab, tec_thorlab.get_error, "error register"))
        report.add("error register", error == 0, f"0x{error:X}")
    except Exception as exc:
        report.add("error register", False, exc)

    try:
        actual_mc = int(_ptc_call(
            tec_thorlab, tec_thorlab.get_actual_temperature, "actual temperature"
        ))
        min_mc = int(_ptc_call(tec_thorlab, tec_thorlab.get_min_temperature, "min temperature"))
        max_mc = int(_ptc_call(tec_thorlab, tec_thorlab.get_max_temperature, "max temperature"))
        in_range = min_mc <= actual_mc <= max_mc
        report.add(
            "temperature read",
            in_range,
            f"{actual_mc / 1000:.3f} C (limits {min_mc / 1000:.1f}..{max_mc / 1000:.1f} C)",
        )
    except Exception as exc:
        report.add("temperature read", False, exc)

    try:
        current_ma = int(_ptc_call(tec_thorlab, tec_thorlab.get_actual_current, "current"))
        voltage_mv = int(_ptc_call(tec_thorlab, tec_thorlab.get_actual_voltage, "voltage"))
        sane = abs(current_ma) <= 3000 and abs(voltage_mv) <= 20000
        report.add(
            "current and voltage",
            sane,
            f"{current_ma / 1000:.3f} A, {voltage_mv / 1000:.3f} V",
        )
    except Exception as exc:
        report.add("current and voltage", False, exc)

    try:
        original_mc = int(_ptc_call(
            tec_thorlab, tec_thorlab.get_target_temperature, "target temperature"
        ))
        low_mc = int(_ptc_call(tec_thorlab, tec_thorlab.get_min_temperature, "min temperature"))
        high_mc = int(_ptc_call(tec_thorlab, tec_thorlab.get_max_temperature, "max temperature"))
        # The DLL documents 5 C to 45 C. Stay inside both that and the device limits.
        low_mc = max(low_mc, 5000)
        high_mc = min(high_mc, 45000)
        trial_mc = int(_offset_setpoint(original_mc, int(round(delta_c * 1000)), low_mc, high_mc))
        status = tec_thorlab.set_target_temperature(trial_mc)
        readback_mc = int(_ptc_call(
            tec_thorlab, tec_thorlab.get_target_temperature, "target readback"
        ))
        report.add(
            "temperature set",
            status == 0 and readback_mc == trial_mc,
            f"wrote {trial_mc / 1000:.3f} C, read {readback_mc / 1000:.3f} C",
        )
    except Exception as exc:
        report.add("temperature set", False, exc)
    finally:
        if original_mc is not None:
            try:
                tec_thorlab.set_target_temperature(int(original_mc))
                restored = int(_ptc_call(
                    tec_thorlab, tec_thorlab.get_target_temperature, "restore target"
                ))
                report.add(
                    "setpoint restored",
                    restored == int(original_mc),
                    f"{restored / 1000:.3f} C",
                )
            except Exception as exc:
                report.add("setpoint restored", False, exc)
    return report.result()


def self_test_arroyo(tec_arroyo, log=print, delta_c=1.0, low_c=5.0, high_c=50.0):
    """Check Arroyo temperature set/read and that the control loop is on.

    Writes a 1 C trial setpoint, reads it back, then restores the original
    setpoint. Loop status passes only when the mode is temperature and
    TEC:OUT is on. Measured temperature is read, but it is not required to
    equal the setpoint.
    """
    report = _SelfTest("Arroyo TEC", log)
    original_c = None
    try:
        ident = str(tec_arroyo.get_id()).strip()
        report.add("identity", bool(ident), ident or "empty identity")
    except Exception as exc:
        report.add("identity", False, exc)

    try:
        mode = str(tec_arroyo.get_mode()).strip()
        output = int(tec_arroyo.get_output())
        temperature_mode = mode.split(",")[0].strip().upper() in ("T", "TEMP", "TEMPERATURE")
        report.add(
            "loop status",
            temperature_mode and output == 1,
            f"mode {mode or 'blank'}, TEC:OUT {output}",
        )
    except Exception as exc:
        report.add("loop status", False, exc)

    try:
        measured_c = float(tec_arroyo.get_temp())
        report.add(
            "temperature read",
            _finite(measured_c) and low_c <= measured_c <= 90.0,
            f"{measured_c:.3f} C, {float(tec_arroyo.get_curr()):.3f} A, "
            f"{float(tec_arroyo.get_volt()):.3f} V",
        )
    except Exception as exc:
        report.add("temperature read", False, exc)

    try:
        original_c = float(tec_arroyo.get_set_temp())
        trial_c = float(_offset_setpoint(original_c, float(delta_c), low_c, high_c))
        tec_arroyo.set_temp(trial_c)
        readback_c = float(tec_arroyo.get_set_temp())
        report.add(
            "temperature set",
            abs(readback_c - trial_c) <= 0.05,
            f"wrote {trial_c:.3f} C, read {readback_c:.3f} C",
        )
    except Exception as exc:
        report.add("temperature set", False, exc)
    finally:
        if original_c is not None:
            try:
                tec_arroyo.set_temp(float(original_c))
                restored_c = float(tec_arroyo.get_set_temp())
                report.add(
                    "setpoint restored",
                    abs(restored_c - float(original_c)) <= 0.05,
                    f"{restored_c:.3f} C",
                )
            except Exception as exc:
                report.add("setpoint restored", False, exc)
    return report.result()


def _osa_no_error(osa):
    error = osa.query("SYST:ERR?").strip()
    return error, error.startswith("0") or error.startswith("+0")


def self_test_osa(osa, log=print, sweep_timeout_s=120):
    """Check OSA identity, the wavelength window, one sweep, and the error queue.

    Uses the wavelength window already on the instrument. The sweep must
    finish, and the peak marker must land inside that window.
    """
    report = _SelfTest("OSA", log)
    try:
        ident = str(osa.get_id()).strip()
        report.add("identity", bool(ident), ident or "empty identity")
    except Exception as exc:
        report.add("identity", False, exc)
        return report.result()

    start_nm = stop_nm = None
    try:
        start_nm = float(osa.query(":SENS:WAV:STAR?")) * 1e9
        stop_nm = float(osa.query(":SENS:WAV:STOP?")) * 1e9
        resolution_nm = float(osa.get_resolution_bandwidth())
        window_ok = _finite(start_nm) and _finite(stop_nm) and start_nm < stop_nm
        report.add(
            "wavelength window",
            window_ok and _finite(resolution_nm) and resolution_nm > 0,
            f"{start_nm:.3f} to {stop_nm:.3f} nm, resolution {resolution_nm:.4f} nm",
        )
    except Exception as exc:
        report.add("wavelength window", False, exc)

    try:
        log(f"  OSA sweep started (timeout {sweep_timeout_s:.0f} s)")
        osa.sweep_done = False
        osa.clear()
        osa.write(":INIT:SMODE SINGLE")
        osa.write(":INIT")
        deadline = time.monotonic() + sweep_timeout_s
        finished = False
        while time.monotonic() < deadline:
            status = int(osa.query(":STAT:OPER:EVEN?"))
            if status & 1:
                finished = True
                osa.sweep_done = True
                break
            time.sleep(0.2)
        if not finished:
            report.add("sweep", False, f"did not finish within {sweep_timeout_s:.0f} s")
        else:
            osa.write(":CALC:MARK1:MAX")
            peak_nm = float(osa.query(":CALC:MARK1:X?")) * 1e9
            peak_dbm = float(osa.query(":CALC:MARK1:Y?"))
            in_window = start_nm is not None and (start_nm - 1.0) <= peak_nm <= (stop_nm + 1.0)
            power_ok = _finite(peak_dbm) and -200.0 <= peak_dbm <= 40.0
            report.add(
                "sweep",
                in_window and power_ok,
                f"peak {peak_nm:.3f} nm at {peak_dbm:.2f} dBm",
            )
    except Exception as exc:
        report.add("sweep", False, exc)

    try:
        error, ok = _osa_no_error(osa)
        report.add("error queue", ok, error)
    except Exception as exc:
        report.add("error queue", False, exc)
    return report.result()


def _opm_wavelength_nm(opm, channel):
    return float(opm.device.query(f"SENS{channel}:POW:WAV?")) * 1e9


def self_test_opm(opm, log=print, channels=(1, 2, 3, 4), wavelength_nm=1310.0):
    """Check the Keysight OPM identity, wavelength set/read, and a power read.

    Each channel is set to wavelength_nm and then restored to the wavelength
    it had at the start of that channel's check. A dark port still passes
    when the power reading is a finite number.
    """
    report = _SelfTest("OPM", log)
    device = getattr(opm, "device", None)
    if device is None:
        report.add("identity", False, "OPM is not connected")
        return report.result()
    try:
        device.write("*CLS")
        ident = str(opm.identify()).strip()
        report.add("identity", "N774" in ident.upper() or "KEYSIGHT" in ident.upper(), ident or "empty identity")
    except Exception as exc:
        report.add("identity", False, exc)
        return report.result()

    for channel in channels:
        original_nm = None
        try:
            original_nm = _opm_wavelength_nm(opm, channel)
            opm.set_wavelength(channel, float(wavelength_nm))
            readback_nm = _opm_wavelength_nm(opm, channel)
            report.add(
                f"channel {channel} wavelength",
                abs(readback_nm - float(wavelength_nm)) <= 0.1,
                f"wrote {float(wavelength_nm):.3f} nm, read {readback_nm:.3f} nm",
            )
        except Exception as exc:
            report.add(f"channel {channel} wavelength", False, exc)
        finally:
            if original_nm is not None:
                try:
                    opm.set_wavelength(channel, float(original_nm))
                except Exception as exc:
                    report.add(f"channel {channel} wavelength restored", False, exc)
        try:
            power = float(opm.read_power(channel))
            report.add(
                f"channel {channel} power",
                _finite(power),
                f"{power:.3f}",
            )
        except Exception as exc:
            report.add(f"channel {channel} power", False, exc)

    try:
        error = str(opm.get_errors()).strip()
        report.add("error queue", error.startswith("0") or error.startswith("+0"), error)
    except Exception as exc:
        report.add("error queue", False, exc)
    return report.result()


def _missing_instrument(device, detail, log):
    report = _SelfTest(device, log)
    report.add("connection", False, detail)
    return report.result()


def run_equipment_self_tests(instruments=None, cfg=None, log=print, opm_channels=(1, 2, 3, 4)):
    """Run TEC, OSA, and OPM self tests on handles that are already open.

    Thorlabs TEC, Arroyo TEC, and the OSA are not opened here. Start the
    test station first. The OPM is opened from cfg['opm_visa'] when it is
    not already in ``instruments``, and that temporary connection is closed
    before return.
    """
    instruments = instruments or {}
    cfg = cfg or {}
    log("Instrument self test")
    results = []

    thorlab = instruments.get("tec_thorlab")
    if thorlab is None:
        results.append(_missing_instrument(
            "Thorlabs TEC", "not open. Start the test station first.", log
        ))
    else:
        results.append(self_test_thorlab(thorlab, log=log))

    arroyo = instruments.get("tec_arroyo")
    if arroyo is None:
        results.append(_missing_instrument(
            "Arroyo TEC", "not open. Start the test station first.", log
        ))
    else:
        results.append(self_test_arroyo(arroyo, log=log))

    osa = instruments.get("osa")
    if osa is None:
        results.append(_missing_instrument(
            "OSA", "not open. Start the test station first.", log
        ))
    else:
        results.append(self_test_osa(osa, log=log))

    opm = instruments.get("opm")
    opened_opm = None
    if opm is None:
        if not cfg.get("opm_visa"):
            results.append(_missing_instrument(
                "OPM", "opm_visa is not set in lab_config.yaml", log
            ))
        else:
            try:
                log(f"  opening OPM at {cfg['opm_visa']}")
                opened_opm = open_opm(cfg)
                opm = opened_opm
            except Exception as exc:
                results.append(_missing_instrument("OPM", f"open failed: {exc}", log))
    if opm is not None:
        try:
            results.append(self_test_opm(opm, log=log, channels=opm_channels))
        finally:
            if opened_opm is not None:
                try:
                    _close_opm(opened_opm)
                    log("  OPM closed")
                except Exception as exc:
                    log(f"  OPM close failed: {exc}")

    failed = [item["device"] for item in results if not item["passed"]]
    if failed:
        log("Instrument self test finished. Failed: " + ", ".join(failed))
    else:
        log("Instrument self test finished. All passed.")
    return results


def read_tec_state_arroyo(tec_arroyo):
    """Read Arroyo TEC only; Thorlabs columns left as NaN for CSV compatibility."""
    temp_arroyo = tec_arroyo.get_temp()
    return {
        "temp": temp_arroyo,
        "temp_thorlab": float("nan"),
        "curr_thorlab": float("nan"),
        "volt_thorlab": float("nan"),
        "temp_arroyo": temp_arroyo,
        "curr_arroyo": tec_arroyo.get_curr(),
        "volt_arroyo": tec_arroyo.get_volt(),
    }


def _reading_power(reading):
    """ER-1000 uses power_dbm; ER-100 uses relative power_db."""
    if hasattr(reading, "power_dbm"):
        return reading.power_dbm
    return getattr(reading, "power_db", float("nan"))


def _collect_er_both(meter_a, meter_b, duration_s):
    """Read both PER meters for duration_s in parallel. Returns (list_a, list_b)."""
    import threading

    buckets = {"A": [], "B": []}
    errors = {}

    def _worker(key, meter):
        try:
            buckets[key] = list(meter.iter_er(duration_s))
        except Exception as exc:
            errors[key] = exc

    threads = [
        threading.Thread(target=_worker, args=("A", meter_a), daemon=True),
        threading.Thread(target=_worker, args=("B", meter_b), daemon=True),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if errors:
        detail = "; ".join(f"{key}: {exc}" for key, exc in errors.items())
        raise RuntimeError(f"PER meter read failed: {detail}")
    return buckets["A"], buckets["B"]


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


def one_way_setpoint(elapsed_s, duration_s, start, end):
    """Linear setpoint from start to end across duration_s."""
    if elapsed_s <= 0 or duration_s <= 0:
        return float(start)
    if elapsed_s >= duration_s:
        return float(end)
    return float(start + (end - start) * (elapsed_s / duration_s))


def channel_ramp_limits(channel, temp0, temp1):
    """Even channels ramp temp0 to temp1. Odd channels ramp temp1 to temp0."""
    if int(channel) % 2 == 0:
        return float(temp0), float(temp1)
    return float(temp1), float(temp0)


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
        ensure_arroyo_tec_enabled(tec_arroyo)
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


def _set_thorlab_temperature(tec_thorlab, temp_c):
    tec_thorlab.set_target_temperature(int(round(1000 * temp_c)))


def _read_thorlab_state(tec_thorlab):
    """Read the Thorlabs TEC. Temperature, current, and voltage are in C, A, and V."""
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
        "curr_thorlab": curr_thorlab[0] / 1000,
        "volt_thorlab": volt_thorlab[0] / 1000,
    }


def per_meter_cal(
    tec_thorlab,
    per_meter_er100,
    data_dir,
    temp0=30,
    temp1=36,
    ramp_time_s=600,
    ramp_step_s=1.0,
    note="",
):
    """Ramp the Thorlabs TEC up and down while recording the ER-100.

    The TEC is commanded to temp0 first. It then ramps from temp0 to temp1
    over the first half of ramp_time_s and back to temp0 over the second half.
    The setpoint is rewritten every ramp_step_s seconds. PER and the measured
    Thorlabs temperature are recorded between those writes. et_0 is time since
    the test started, and et_tecAdj is time since the ramp started.
    """
    if ramp_time_s <= 0:
        raise ValueError("ramp_time_s must be positive")
    if ramp_step_s <= 0 or ramp_step_s > ramp_time_s / 2:
        raise ValueError("ramp_step_s must be positive and no more than ramp_time_s / 2")
    columns = [
        "time", "et_0", "et_tecAdj", "temp_tgt",
        "temp", "temp_thorlab", "curr_thorlab", "volt_thorlab",
        "per_dB", "theta", "pow",
    ]
    path = _new_csv_path(data_dir, "per meter cal")
    writer = _RunWriter(path, columns, {"note": note})
    print(f"Writing {path}")
    t0 = time.time()
    started = False
    try:
        per_meter_er100.stop(flush=True)
        per_meter_er100.set_mode("er")
        per_meter_er100.set_capture(True)
        _set_thorlab_temperature(tec_thorlab, temp0)
        per_meter_er100.start()
        started = True
        ramp_mark = time.monotonic()
        t_tec = time.time()
        next_write = ramp_mark
        commanded = float(temp0)
        while True:
            elapsed = time.monotonic() - ramp_mark
            if elapsed >= ramp_time_s:
                break
            if time.monotonic() >= next_write:
                commanded = triangle_setpoint(elapsed, ramp_time_s, temp0, temp1)
                _set_thorlab_temperature(tec_thorlab, commanded)
                next_write = time.monotonic() + ramp_step_s
            remain = ramp_time_s - (time.monotonic() - ramp_mark)
            slice_s = min(next_write - time.monotonic(), remain)
            if slice_s <= 0:
                continue
            for reading in per_meter_er100.iter_er(slice_s):
                now = time.time()
                tec_state = _read_thorlab_state(tec_thorlab)
                row = {
                    "time": _now(),
                    "et_0": round(now - t0, 1),
                    "et_tecAdj": round(now - t_tec, 1),
                    "temp_tgt": commanded,
                    "per_dB": reading.per_db,
                    "theta": reading.angle_deg,
                    "pow": reading.power_db,
                }
                row.update(tec_state)
                writer.append([row])
                print(
                    row["et_0"],
                    round(commanded, 3),
                    round(tec_state["temp_thorlab"], 3),
                    reading.per_db,
                )
        _set_thorlab_temperature(tec_thorlab, temp0)
    except KeyboardInterrupt:
        print(f"Interrupted. Partial data: {path}")
    except Exception:
        print(f"Partial data: {path}")
        raise
    finally:
        if started:
            per_meter_er100.stop(flush=True)
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


def _grid_wavelength_nm(bank, channel):
    """Return the Kurtz grid wavelength for one bank and channel."""
    if bank == "A":
        index = 2 * channel
    elif bank == "B":
        index = 2 * channel + 1
    else:
        raise ValueError(f"bank must be 'A' or 'B', got {bank!r}")
    if channel < 0 or channel >= LANES_PER_BANK:
        raise ValueError(f"channel must be 0..{LANES_PER_BANK - 1}, got {channel}")
    return float(WAVELENGTH_GRID_NM[index])


def _uw_to_dbm(power_uw):
    """Convert microwatts to dBm. 0 dBm is 1 mW, which is 1000 uW."""
    power_uw = np.asarray(power_uw, dtype=float)
    if np.any(power_uw <= 0):
        raise ValueError(f"CLM power must be positive uW to convert to dBm: {power_uw}")
    return 10.0 * np.log10(power_uw / 1000.0)


def _strongest_peaks(wavelength_nm, power_dbm, count=16, floor_db=10.0):
    """Return the count strongest peaks from scipy, sorted by wavelength.

    Samples more than floor_db below the trace maximum are clipped to that
    floor before the peak search, so weak peaks are ignored. Returned powers
    are the original samples, not the clipped values.
    """
    from scipy.signal import find_peaks

    wavelength_nm = np.asarray(wavelength_nm, dtype=float)
    power_dbm = np.asarray(power_dbm, dtype=float)
    if power_dbm.size == 0 or not np.any(np.isfinite(power_dbm)):
        raise ValueError("OSA trace has no finite power samples")
    threshold = float(np.nanmax(power_dbm) - floor_db)
    clipped = np.where(np.isfinite(power_dbm), power_dbm, threshold)
    np.maximum(clipped, threshold, out=clipped)
    maxima, _ = find_peaks(clipped)
    above = maxima[power_dbm[maxima] > threshold]
    print(
        f"OSA trace peaks: {len(maxima)} local maxima, "
        f"{len(above)} above {threshold:.2f} dBm "
        f"({floor_db:g} dB below the maximum)"
    )
    if len(above) < count:
        raise ValueError(
            f"Expected at least {count} OSA peaks within {floor_db:g} dB of "
            f"the maximum, found {len(above)}"
        )
    strongest = above[np.argsort(power_dbm[above])[-count:]]
    order = np.argsort(wavelength_nm[strongest])
    chosen = strongest[order]
    return wavelength_nm[chosen], power_dbm[chosen]


def load_clm_to_OSA_OPM_loss(cfg):
    """Return fitted 2x2 coupler path losses for OSA and OPM ports."""
    result = {
        "clm_osa_loss_A": c3db_loss(WAVELENGTH_GRID_NM_A)[cfg["two-by-two-coupler"]["A-OSA"]],
        "clm_osa_loss_B": c3db_loss(WAVELENGTH_GRID_NM_B)[cfg["two-by-two-coupler"]["B-OSA"]],
        "clm_opm_loss_A": c3db_loss(WAVELENGTH_GRID_NM_A)[cfg["two-by-two-coupler"]["A-OPM"]],
        "clm_opm_loss_B": c3db_loss(WAVELENGTH_GRID_NM_B)[cfg["two-by-two-coupler"]["B-OPM"]],
    }
    return result


def _dbm_to_mw(power_dbm):
    return 10.0 ** (np.asarray(power_dbm, dtype=float) / 10.0)


def prepare_tile_mission(tb, tile, cfg, tile_index=None, settle_s=30, log=print):
    """Initialize one tile into Mission with all lanes on, then settle.

    Reopens the package MCI before traffic. Skips CMIS SoftwareReset when it
    returns Aardvark -103 (common right after a hardware reset).
    Returns the (possibly refreshed) tile.
    """
    if tb is None:
        raise ValueError("testboard handle tb is required")
    if settle_s < 0:
        raise ValueError("settle_s must be zero or positive")
    if tile_index is None:
        tile_index = cfg.get("tile_index")
    label = f"tile {tile_index}" if tile_index is not None else "tile"

    log(f"Preparing {label} for optical test...")
    tile = ensure_tile_ready(tb, tile_index, log=log)

    try:
        mca.initialize_tile(tb, tile=tile, software_reset=True)
    except Exception as first_exc:
        log(
            f"WARNING: initialize_tile(software_reset=True) failed on {label}: {first_exc}\n"
            "Continuing with initialize_tile(software_reset=False) after package reopen..."
        )
        reopen_package_tiles(tb, log=log)
        tile = tb.module.packages[0].tiles[int(tile_index)]
        mca.initialize_tile(tb, tile=tile, software_reset=False)

    apply_control_config(tile, **cfg["control"])
    time.sleep(0.2)
    mca.optical_power_on(tile)
    log(f"Enabling all {N_LANES} channels on {label} (fire-up)...")
    mca.clm_lane_enable(tile, chEnableMask=[1] * N_LANES)
    log(
        f"{label} fire-up complete; waiting {settle_s}s settle "
        "before measurements..."
    )
    time.sleep(settle_s)
    return tile


def _power_tag_mW(power_tgt_mw) -> str:
    """Format target power for config filenames: 8 -> 8mW, 6.4 -> 6p4mW."""
    value = float(power_tgt_mw)
    if value == int(value):
        return f"{int(value)}mW"
    text = f"{value:g}".replace(".", "p")
    return f"{text}mW"


def _sanitize_name(text: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(text).strip())
    return cleaned or "unknown"


def read_live_laser_voa_pic_temp(tile):
    """Read live 16-ch laser bias, 16-ch VOA bias, and PIC temperature."""
    laser_a = list(tile.read_bias_current_monitors(tile.Bank.BANK_A))
    laser_b = list(tile.read_bias_current_monitors(tile.Bank.BANK_B))
    voa_a = list(tile.read_aux_bias_current_monitors(tile.Bank.BANK_A))
    voa_b = list(tile.read_aux_bias_current_monitors(tile.Bank.BANK_B))
    laser = [float(x) for x in laser_a + laser_b]
    voa = [float(x) for x in voa_a + voa_b]
    if len(laser) != 16:
        raise ValueError(f"Expected 16 laser bias monitors, got {len(laser)}")
    if len(voa) != 16:
        raise ValueError(f"Expected 16 VOA bias monitors, got {len(voa)}")
    temp_pic = float(tile.read_temp_pic())
    return {"laser": laser, "voa": voa, "temp_pic": temp_pic}


def export_clm_config_yaml(
    tile,
    output_dir,
    *,
    tile_index,
    dut_sn,
    operator="",
    power_tgt_mw=None,
    laser_currents=None,
    voa_currents=None,
    temp_pic=None,
):
    """Write compact CLM config YAML (current / responsivity / temp) after tap cal.

    Filename:
    ``{tile_index}_0_tile_{dut_sn}_{tile_fru_sn}_config_file_{powerTag}.yaml``

    Optional ``laser_currents`` / ``voa_currents`` / ``temp_pic`` override FRU
    values (used after tap cal when live monitors are written to FRU).
    ``operator`` is accepted for call-site compatibility but is not used in the
    filename.
    """
    if power_tgt_mw is None:
        power_tgt_mw = float(
            load_test_spec().get("clm_test_spec", {}).get("target_power", 8)
        )
    fru = mca.get_fru_full(tile)
    laser = (
        [float(x) for x in laser_currents]
        if laser_currents is not None
        else [float(x) for x in fru["bias_currents"]]
    )
    voa = (
        [float(x) for x in voa_currents]
        if voa_currents is not None
        else [float(x) for x in fru["voa_currents"]]
    )
    pic_t = float(temp_pic) if temp_pic is not None else float(fru["temp_pic"])
    payload = {
        "current": {
            "laser": laser,
            "voa": voa,
        },
        "responsivity": {
            "mux": [float(x) for x in fru["mux_mpd_tap_ratios"]],
            "pic": [float(x) for x in fru["pic_mpd_tap_ratios"]],
        },
        "temp": {
            "mux": float(fru["temp_mux"]),
            "pic": pic_t,
        },
    }
    tile_fru_sn = _sanitize_name(fru.get("serial_number", "unknown"))
    filename = (
        f"{int(tile_index)}_0_tile_{_sanitize_name(dut_sn)}_{tile_fru_sn}_"
        f"config_file_{_power_tag_mW(power_tgt_mw)}.yaml"
    )
    path = Path(output_dir) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.dump(payload, handle, default_flow_style=False, sort_keys=False)
    print(f"Wrote CLM config YAML: {path}")
    return path


def _tap_cal_sample(tile, osa, loss_A, loss_B, trace="TRA"):
    """One CLM+OSA pass; PIC and MUX (aux) powers vs OSA-referred power."""
    clm_power_A = np.asarray(
        tile.read_optical_power_monitors(tile.Bank.BANK_A), dtype=float
    )
    clm_power_B = np.asarray(
        tile.read_optical_power_monitors(tile.Bank.BANK_B), dtype=float
    )
    aux_power_A = np.asarray(
        tile.read_aux_optical_power_monitors(tile.Bank.BANK_A), dtype=float
    )
    aux_power_B = np.asarray(
        tile.read_aux_optical_power_monitors(tile.Bank.BANK_B), dtype=float
    )
    if len(clm_power_A) != 8 or len(clm_power_B) != 8:
        raise ValueError(
            f"Expected 8 CLM powers per bank, got A={len(clm_power_A)} B={len(clm_power_B)}"
        )
    if len(aux_power_A) != 8 or len(aux_power_B) != 8:
        raise ValueError(
            f"Expected 8 aux (MUX) powers per bank, got A={len(aux_power_A)} B={len(aux_power_B)}"
        )
    osa.write(f":TRAC:ACTIVE {trace}")
    osa.sweep_done = False
    osa.single_sweep()
    wavelength_nm, power_dbm = osa.get_data()
    peak_wavelength, peak_power = _strongest_peaks(wavelength_nm, power_dbm, count=16)
    osa_peak_A = peak_power[0:16:2]
    osa_peak_B = peak_power[1:16:2]
    measured_dbm_A = osa_peak_A + loss_A
    measured_dbm_B = osa_peak_B + loss_B
    measured_mw = np.concatenate([_dbm_to_mw(measured_dbm_A), _dbm_to_mw(measured_dbm_B)])
    clm_mw = np.concatenate([clm_power_A, clm_power_B]) / 1000.0
    aux_mw = np.concatenate([aux_power_A, aux_power_B]) / 1000.0
    if np.any(clm_mw <= 0) or np.any(aux_mw <= 0) or np.any(measured_mw <= 0):
        raise ValueError(
            "PIC, MUX (aux), and measured powers must be positive for tap correction"
        )
    corr_pic = measured_mw / clm_mw
    corr_mux = measured_mw / aux_mw
    return {
        "clm_power_A": clm_power_A,
        "clm_power_B": clm_power_B,
        "aux_power_A": aux_power_A,
        "aux_power_B": aux_power_B,
        "clm_dbm_A": _uw_to_dbm(clm_power_A),
        "clm_dbm_B": _uw_to_dbm(clm_power_B),
        "aux_dbm_A": _uw_to_dbm(aux_power_A),
        "aux_dbm_B": _uw_to_dbm(aux_power_B),
        "osa_peak_A": osa_peak_A,
        "osa_peak_B": osa_peak_B,
        "peak_wavelength": peak_wavelength,
        "measured_dbm_A": measured_dbm_A,
        "measured_dbm_B": measured_dbm_B,
        "clm_mw": clm_mw,
        "aux_mw": aux_mw,
        "corr_pic": corr_pic,
        "corr_mux": corr_mux,
    }


def clm_tap_ratio_cal_osa(
    tile,
    osa,
    cfg,
    settle_s=60,
    trace="TRA",
    write_fru=True,
    note="",
    tile_index=None,
    tb=None,
    log=print,
    n_samples=3,
    data_dir=None,
    dut_sn="",
    operator="",
):
    """Recalibrate PIC and MUX MPD tap ratios using OSA + coupler path loss.

    Takes n_samples passes of PIC (optical) and MUX (aux) monitors vs OSA peaks,
    uses per-channel median corrections, optionally writes both tap arrays to
    FRU. After taps, samples live laser/VOA/PIC temp, writes those to FRU
    setpoints/app settings, and exports a compact config YAML.
    """
    if tile is None:
        raise ValueError("tile handle is required")
    if osa is None:
        raise ValueError("osa handle is required")
    if settle_s < 0:
        raise ValueError("settle_s must be zero or positive")
    if n_samples < 1:
        raise ValueError("n_samples must be at least 1")
    if tile_index is None:
        tile_index = cfg.get("tile_index")
    if data_dir is None:
        data_dir = cfg.get("data_dir")

    losses = load_clm_to_OSA_OPM_loss(cfg)
    loss_A = np.asarray(losses["clm_osa_loss_A"], dtype=float)
    loss_B = np.asarray(losses["clm_osa_loss_B"], dtype=float)
    rows = []

    label = f"tile {tile_index}" if tile_index is not None else "tile"
    log(f"Tap ratio OSA cal {label}" + (f" ({note})" if note else ""))
    tile = prepare_tile_mission(
        tb, tile, cfg, tile_index=tile_index, settle_s=settle_s, log=log
    )

    samples = []
    for pass_i in range(n_samples):
        sample = _tap_cal_sample(tile, osa, loss_A, loss_B, trace=trace)
        samples.append(sample)
        log(
            f"{label} tap-cal pass {pass_i + 1}/{n_samples} "
            f"corr_pic={np.array2string(sample['corr_pic'], precision=4)} "
            f"corr_mux={np.array2string(sample['corr_mux'], precision=4)}"
        )
        log(
            f"PIC power {label} bank A (uW): "
            f"{np.array2string(sample['clm_power_A'], precision=3)}"
        )
        log(
            f"PIC power {label} bank B (uW): "
            f"{np.array2string(sample['clm_power_B'], precision=3)}"
        )
        log(
            f"MUX aux power {label} bank A (uW): "
            f"{np.array2string(sample['aux_power_A'], precision=3)}"
        )
        log(
            f"MUX aux power {label} bank B (uW): "
            f"{np.array2string(sample['aux_power_B'], precision=3)}"
        )

    corr_pic = np.median(np.vstack([s["corr_pic"] for s in samples]), axis=0)
    corr_mux = np.median(np.vstack([s["corr_mux"] for s in samples]), axis=0)
    clm_mw = np.median(np.vstack([s["clm_mw"] for s in samples]), axis=0)
    aux_mw = np.median(np.vstack([s["aux_mw"] for s in samples]), axis=0)
    clm_dbm_A = np.median(np.vstack([s["clm_dbm_A"] for s in samples]), axis=0)
    clm_dbm_B = np.median(np.vstack([s["clm_dbm_B"] for s in samples]), axis=0)
    aux_dbm_A = np.median(np.vstack([s["aux_dbm_A"] for s in samples]), axis=0)
    aux_dbm_B = np.median(np.vstack([s["aux_dbm_B"] for s in samples]), axis=0)
    osa_peak_A = np.median(np.vstack([s["osa_peak_A"] for s in samples]), axis=0)
    osa_peak_B = np.median(np.vstack([s["osa_peak_B"] for s in samples]), axis=0)
    measured_dbm_A = np.median(np.vstack([s["measured_dbm_A"] for s in samples]), axis=0)
    measured_dbm_B = np.median(np.vstack([s["measured_dbm_B"] for s in samples]), axis=0)
    peak_wavelength = np.median(
        np.vstack([s["peak_wavelength"] for s in samples]), axis=0
    )

    fru = mca.get_fru_full(tile)
    pic_old = list(fru["pic_mpd_tap_ratios"])
    mux_old = list(fru["mux_mpd_tap_ratios"])
    if len(pic_old) != 16:
        raise ValueError(f"Expected 16 PIC tap ratios, got {len(pic_old)}")
    if len(mux_old) != 16:
        raise ValueError(f"Expected 16 MUX tap ratios, got {len(mux_old)}")
    pic_new = [round(float(pic_old[ch] / corr_pic[ch]), 4) for ch in range(16)]
    mux_new = [round(float(mux_old[ch] / corr_mux[ch]), 4) for ch in range(16)]

    log(f"old pic tap ratios: {pic_old}")
    log(f"new pic tap ratios: {pic_new}")
    log(f"old mux tap ratios: {mux_old}")
    log(f"new mux tap ratios: {mux_new}")
    log(f"median PIC corr (measured/pic): {np.array2string(corr_pic, precision=4)}")
    log(f"median MUX corr (measured/aux): {np.array2string(corr_mux, precision=4)}")

    live = None
    if write_fru:
        tile.flash_write_fru_optical_measurements(
            pic_new, mux_new, fru["mux_freq_errors"]
        )
        log(f"Wrote PIC and MUX tap ratios to FRU for {label}")

        live = read_live_laser_voa_pic_temp(tile)
        log(
            f"Live laser bias (mA): "
            f"{np.array2string(np.asarray(live['laser']), precision=3)}"
        )
        log(
            f"Live VOA bias (mA): "
            f"{np.array2string(np.asarray(live['voa']), precision=3)}"
        )
        log(f"Live PIC temp (C): {live['temp_pic']:.3f}")

        # VOA lives in app settings; keep existing power / freq targets.
        tile.flash_write_fru_app_settings(
            voa_currents=live["voa"],
            power_targets=list(fru["power_targets"]),
            freq_error_targets=list(fru["freq_error_targets"]),
        )
        # Laser + PIC temp in setpoints; keep MUX temp, MPD currents, and gains.
        tile.flash_write_fru_setpoints(
            live["temp_pic"],
            float(fru["temp_mux"]),
            live["laser"],
            list(fru.get("pic_mpd_currents") or [0] * 16),
            list(fru.get("mux_mpd_currents") or [0] * 16),
            list(fru.get("mux_mpd_gain_settings") or [0] * 16),
        )
        log(
            f"Wrote live laser/VOA bias and PIC temp to FRU for {label} "
            f"(MUX temp left at {float(fru['temp_mux']):.3f} C)"
        )

    if data_dir:
        try:
            yaml_kwargs = {}
            if live is not None:
                yaml_kwargs = {
                    "laser_currents": live["laser"],
                    "voa_currents": live["voa"],
                    "temp_pic": live["temp_pic"],
                }
            yaml_path = export_clm_config_yaml(
                tile,
                data_dir,
                tile_index=tile_index if tile_index is not None else 0,
                dut_sn=dut_sn or "unknown",
                operator=operator or "",
                **yaml_kwargs,
            )
            log(f"CLM config YAML: {yaml_path}")
        except Exception as exc:
            log(f"WARNING: failed to write CLM config YAML: {exc}")

    for ch in range(16):
        bank = "A" if ch < 8 else "B"
        lane = ch if ch < 8 else ch - 8
        rows.append({
            "tile": tile_index,
            "bank": bank,
            "ch": lane,
            "clm_power_uW": float(clm_mw[ch] * 1000.0),
            "clm_power_dBm": float(
                clm_dbm_A[lane] if bank == "A" else clm_dbm_B[lane]
            ),
            "aux_power_uW": float(aux_mw[ch] * 1000.0),
            "aux_power_dBm": float(
                aux_dbm_A[lane] if bank == "A" else aux_dbm_B[lane]
            ),
            "osa_peak_power_dBm": float(
                osa_peak_A[lane] if bank == "A" else osa_peak_B[lane]
            ),
            "osa_peak_wavelength_nm": float(
                peak_wavelength[0:16:2][lane]
                if bank == "A"
                else peak_wavelength[1:16:2][lane]
            ),
            "clm_osa_loss_dB": float(loss_A[lane] if bank == "A" else loss_B[lane]),
            "power_from_osa_dBm": float(
                measured_dbm_A[lane] if bank == "A" else measured_dbm_B[lane]
            ),
            "corr_pic": float(corr_pic[ch]),
            "corr_mux": float(corr_mux[ch]),
            "pic_mpd_tap_ratio_old": float(pic_old[ch]),
            "pic_mpd_tap_ratio_new": float(pic_new[ch]),
            "mux_mpd_tap_ratio_old": float(mux_old[ch]),
            "mux_mpd_tap_ratio_new": float(mux_new[ch]),
            "note": note,
        })

    return pd.DataFrame(rows)


def clm_to_OSA_loss_cal(tile, osa, trace="TRA"):
    """Compare CLM monitor power with OSA peaks and return the path loss.

    Each bank's optical power monitor is 8 values in uW. The OSA trace is
    swept and the 16 strongest peaks are sorted by wavelength. Even peaks
    are bank A and odd peaks are bank B, matching the interleaved frequency
    grid. Loss is CLM power in dBm minus the OSA peak power in dBm.
    """
    clm_power_A = np.asarray(
        tile.read_optical_power_monitors(tile.Bank.BANK_A), dtype=float
    )
    clm_power_B = np.asarray(
        tile.read_optical_power_monitors(tile.Bank.BANK_B), dtype=float
    )
    if len(clm_power_A) != 8 or len(clm_power_B) != 8:
        raise ValueError(
            f"Expected 8 CLM powers per bank, got A={len(clm_power_A)} B={len(clm_power_B)}"
        )

    osa.write(f":TRAC:ACTIVE {trace}")
    osa.sweep_done = False
    osa.single_sweep()
    wavelength_nm, power_dbm = osa.get_data()
    peak_wavelength, peak_power = _strongest_peaks(wavelength_nm, power_dbm, count=16)

    osa_peak_wavelength_A = peak_wavelength[0:16:2]
    osa_peak_wavelength_B = peak_wavelength[1:16:2]
    osa_peak_power_A = peak_power[0:16:2]
    osa_peak_power_B = peak_power[1:16:2]
    result = {
        "clm_power_A": clm_power_A,
        "clm_power_B": clm_power_B,
        "osa_peak_power_A": osa_peak_power_A,
        "osa_peak_power_B": osa_peak_power_B,
        "clm_osa_loss_A": _uw_to_dbm(clm_power_A) - osa_peak_power_A,
        "clm_osa_loss_B": _uw_to_dbm(clm_power_B) - osa_peak_power_B,
        "osa_peak_wavelength_A": osa_peak_wavelength_A,
        "osa_peak_wavelength_B": osa_peak_wavelength_B,
    }
    for name, values in result.items():
        print(name, np.array2string(values, precision=3))
    return result

def clm_power_n_wavelength_stability(
    tile,
    osa,
    clm_osa_loss_A,
    clm_osa_loss_B,
    test_duration=60 * 15,
    sample_interval=10,
    trace="TRA",
    idx_tile=3,
):
    """Record CLM power and OSA wavelength until test_duration seconds have passed.

    Each sample calls clm_to_OSA_loss_cal for the live CLM powers and OSA
    peaks. The loss arrays returned by that call are not used. clm_osa_loss_A
    and clm_osa_loss_B are the calibration losses supplied by the caller.
    OSA peak power plus that loss is the OSA power referred back to the CLM.
    Each row also stores that channel's LD and VOA bias in mA, and the tile
    PIC temperature in C as T_PIC. idx_tile is written on every row.
    wavelength_error_nm is the OSA peak minus the Kurtz grid wavelength for
    that bank and channel.
    """
    loss_A = np.asarray(clm_osa_loss_A, dtype=float)
    loss_B = np.asarray(clm_osa_loss_B, dtype=float)
    if len(loss_A) != 8 or len(loss_B) != 8:
        raise ValueError(
            f"Expected 8 loss values per bank, got A={len(loss_A)} B={len(loss_B)}"
        )
    if test_duration <= 0:
        raise ValueError("test_duration must be positive")
    if sample_interval <= 0:
        raise ValueError("sample_interval must be positive")

    columns = [
        "time", "et_s", "idx_tile", "bank", "ch",
        "clm_power_uW", "clm_power_dBm",
        "osa_peak_power_dBm", "osa_peak_wavelength_nm", "wavelength_error_nm",
        "clm_osa_loss_dB", "power_from_osa_dBm",
        "LD_mA", "VOA_mA", "T_PIC",
    ]
    frames = []
    t0 = time.monotonic()
    deadline = t0 + test_duration
    next_sample = t0
    try:
        while time.monotonic() < deadline:
            wait_s = next_sample - time.monotonic()
            if wait_s > 0:
                time.sleep(min(wait_s, deadline - time.monotonic()))
            if time.monotonic() >= deadline:
                break
            sample = clm_to_OSA_loss_cal(tile, osa, trace=trace)
            live = read_live_laser_voa_pic_temp(tile)
            et_s = round(time.monotonic() - t0, 1)
            stamp = _now()
            rows = []
            banks = (
                ("A", 0, sample["clm_power_A"], sample["osa_peak_power_A"],
                 sample["osa_peak_wavelength_A"], loss_A),
                ("B", 8, sample["clm_power_B"], sample["osa_peak_power_B"],
                 sample["osa_peak_wavelength_B"], loss_B),
            )
            for bank, bank_offset, clm_uw, osa_dbm, wavelength_nm, loss_db in banks:
                clm_dbm = _uw_to_dbm(clm_uw)
                for channel in range(8):
                    rows.append({
                        "time": stamp,
                        "et_s": et_s,
                        "idx_tile": idx_tile,
                        "bank": bank,
                        "ch": channel,
                        "clm_power_uW": clm_uw[channel],
                        "clm_power_dBm": clm_dbm[channel],
                        "osa_peak_power_dBm": osa_dbm[channel],
                        "osa_peak_wavelength_nm": wavelength_nm[channel],
                        "wavelength_error_nm": (
                            wavelength_nm[channel] - _grid_wavelength_nm(bank, channel)
                        ),
                        "clm_osa_loss_dB": loss_db[channel],
                        "power_from_osa_dBm": osa_dbm[channel] + loss_db[channel],
                        "LD_mA": live["laser"][bank_offset + channel],
                        "VOA_mA": live["voa"][bank_offset + channel],
                        "T_PIC": live["temp_pic"],
                    })
            frames.append(pd.DataFrame(rows, columns=columns))
            print(f"sample et_s={et_s}")
            next_sample += sample_interval
    except KeyboardInterrupt:
        print("Interrupted. Returning samples collected so far.")
    if not frames:
        return pd.DataFrame(columns=columns)
    return pd.concat(frames, ignore_index=True)


STABILITY_METRIC = "15 min power stability"
MUX_DITHER_OFF_STABILITY = "power stability with MUX dither off"


def set_mux_dither_enable(tile, enable, log=print):
    """Write MuxDitherEnable on both banks and require the read-back to match."""
    if tile is None:
        raise ValueError("tile handle is required")
    wanted = bool(enable)
    for bank in mca.ClmTile.Bank:
        bank_i = int(bank)
        name = "A" if bank_i == 0 else "B"
        tile.write_mux_dither_enable(bank_i, wanted)
        state = bool(tile.read_mux_dither_enable(bank_i))
        log(f"MUX dither bank {name}: {'on' if state else 'off'}")
        if state != wanted:
            raise RuntimeError(
                f"MUX dither bank {name} read back {'on' if state else 'off'} "
                f"after write {'on' if wanted else 'off'}."
            )


def stability_report_frame(summary, metric_name=STABILITY_METRIC):
    """Keep tile, bank, channel, and the 3-sigma stability column under metric_name."""
    columns = ["tile", "bank", "channel", STABILITY_METRIC]
    if summary is None or len(summary) == 0:
        return pd.DataFrame(columns=["tile", "bank", "channel", metric_name])
    missing = [name for name in columns if name not in summary.columns]
    if missing:
        raise ValueError(f"Stability summary is missing {missing}")
    frame = summary.loc[:, columns].rename(columns={STABILITY_METRIC: metric_name})
    return frame.reset_index(drop=True)


def summarize_clm_stability(df, power_tgt_mw=8, skip_initial_s=60):
    """Reduce a stability dataframe to one row per tile, bank, and channel.

    Samples with et_s below skip_initial_s are left out before the means.
    skip_initial_s of 0 keeps every sample. wavelength error is the mean of
    wavelength_error_nm. power error is the mean of power_from_osa_dBm minus
    10*log10(power_tgt_mw). 15 min power stability is 3 times the sample
    standard deviation of that same power error. Fewer than two samples
    leaves the stability value empty.
    """
    metrics = ["wavelength error", "power error", STABILITY_METRIC]
    columns = ["tile", "bank", "channel", *metrics]
    if power_tgt_mw <= 0:
        raise ValueError("power_tgt_mw must be positive")
    if skip_initial_s < 0:
        raise ValueError("skip_initial_s must be zero or positive")
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=columns)

    work = df.copy()
    if skip_initial_s > 0 and "et_s" in work.columns:
        elapsed = pd.to_numeric(work["et_s"], errors="coerce")
        work = work[elapsed >= float(skip_initial_s)]
    if len(work) == 0:
        return pd.DataFrame(columns=columns)

    target_dbm = 10.0 * np.log10(float(power_tgt_mw))
    work["power_error_dB"] = work["power_from_osa_dBm"] - target_dbm
    rows = []
    grouped = work.groupby(["idx_tile", "bank", "ch"], sort=True)
    for (idx_tile, bank, channel), group in grouped:
        power_error = group["power_error_dB"]
        deviation = power_error.std(ddof=1)
        rows.append({
            "tile": int(idx_tile),
            "bank": bank,
            "channel": int(channel),
            "wavelength error": float(group["wavelength_error_nm"].mean()),
            "power error": float(power_error.mean()),
            STABILITY_METRIC: (
                None if pd.isna(deviation) else float(3.0 * deviation)
            ),
        })
    return pd.DataFrame(rows, columns=columns)


def clm_power_and_frequency_test(
    tile,
    osa,
    cfg,
    *,
    tb=None,
    tile_index=None,
    test_duration=60 * 15,
    sample_interval=10,
    settle_s=30,
    prepare_tile=True,
    data_dir=None,
    power_tgt_mw=None,
    serial_number="",
    skip_initial_s=60,
    note="",
    log=print,
    mux_dither=None,
    csv_stem=None,
    plot="power_and_frequency",
    plot_heading="",
    log_name="Power and frequency",
):
    """Run power and wavelength stability on one tile and summarize results.

    Optionally initializes the tile, enables all channels, and waits settle_s
    after fire-up. Coupler OSA path losses come from load_clm_to_OSA_OPM_loss.
    mux_dither True or False is written on both banks after that prepare step
    and before the first sample. None leaves the dither bit unchanged.
    Pass/fail numbers omit samples before skip_initial_s. The saved CSV and
    plot keep the full trace. plot is ``power_and_frequency`` or ``power``.
    Returns ``(raw_df, summary_df)``.
    """
    if tile is None:
        raise ValueError("tile handle is required")
    if osa is None:
        raise ValueError("osa handle is required")
    if tile_index is None:
        tile_index = cfg.get("tile_index", 3)
    if data_dir is None:
        data_dir = cfg.get("data_dir")
    if power_tgt_mw is None:
        spec = load_test_spec()
        power_tgt_mw = float(spec.get("clm_test_spec", {}).get("target_power", 8))

    if plot not in ("power_and_frequency", "power"):
        raise ValueError("plot must be 'power_and_frequency' or 'power'")
    label = f"tile {tile_index}"
    log(f"{log_name} test {label}" + (f" ({note})" if note else ""))

    if prepare_tile:
        tile = prepare_tile_mission(
            tb, tile, cfg, tile_index=tile_index, settle_s=settle_s, log=log
        )
    if mux_dither is not None:
        set_mux_dither_enable(tile, mux_dither, log=log)

    losses = load_clm_to_OSA_OPM_loss(cfg)
    raw_df = clm_power_n_wavelength_stability(
        tile,
        osa,
        losses["clm_osa_loss_A"],
        losses["clm_osa_loss_B"],
        test_duration=test_duration,
        sample_interval=sample_interval,
        trace="TRA",
        idx_tile=int(tile_index),
    )
    if data_dir and len(raw_df):
        stem = csv_stem or f"clm_power_n_wavelength_stability_tile{tile_index}"
        csv_path = save_run(
            raw_df,
            data_dir,
            stem,
            params={"note": note, "tile_index": tile_index},
        )
        log(f"{log_name} CSV: {csv_path}")
        try:
            import clm_plots as plots

            if plot == "power":
                plot_path = plots.plot_power_vs_time(
                    raw_df,
                    output_dir=data_dir,
                    file_stem=csv_path.stem,
                    show=False,
                    serial_number=serial_number,
                    tile_index=tile_index,
                    heading=plot_heading or "power vs time",
                )
            else:
                plot_path = plots.plot_power_and_frequency(
                    raw_df,
                    output_dir=data_dir,
                    file_stem=csv_path.stem,
                    show=False,
                    serial_number=serial_number,
                    tile_index=tile_index,
                )
            log(f"{log_name} plot: {plot_path}")
        except Exception as exc:
            log(f"WARNING: {log_name} plot save failed: {exc}")

    if skip_initial_s > 0:
        log(
            f"Pass/fail uses samples with elapsed time >= {skip_initial_s:g}s "
            f"on {label}."
        )
    summary = summarize_clm_stability(
        raw_df, power_tgt_mw=power_tgt_mw, skip_initial_s=skip_initial_s
    )
    print(f"{log_name} summary {label}:\n{summary}")
    return raw_df, summary


def _lanes_off(tile):
    """Disable every lane. This is the optical-off step between fiber moves."""
    mca.clm_lane_enable(tile, chEnableMask=[0] * N_LANES)


def _ask_fiber_connected(port_name):
    """Ask the operator to connect one CLM port. True means Yes."""
    import tkinter as tk
    from tkinter import messagebox

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        return bool(messagebox.askyesno(
            "CLM PER test",
            f"Connect CLM port {port_name} to the PER meter.\n"
            "Click Yes when the fiber is connected.",
            parent=root,
        ))
    finally:
        root.destroy()


def clm_per_test(
    tile,
    tec_thorlab,
    tec_arroyo,
    per_meter,
    data_dir,
    channels=range(LANES_PER_BANK),
    temp0=30,
    temp1=36,
    channel_s=600,
    channel_settle_s=20,
    ramp_step_s=1.0,
    tile_index=None,
    note="",
):
    """Measure PER on port A, then port B, and save the combined frame.

    All lanes are disabled before each prompt and again at the end. No on a
    prompt skips that port. Each port still writes the per_over_temp_3 CSV.
    The combined CSV is clm per test_<timestamp>.csv.
    """
    channels = list(channels)
    frames = []
    try:
        for bank, port_name in ((0, "A"), (1, "B")):
            _lanes_off(tile)
            if not _ask_fiber_connected(port_name):
                print(f"Port {port_name} skipped.")
                continue
            frame = per_over_temp_3(
                tile,
                tec_thorlab,
                tec_arroyo,
                per_meter,
                data_dir,
                bank=bank,
                channels=channels,
                temp0=temp0,
                temp1=temp1,
                channel_s=channel_s,
                channel_settle_s=channel_settle_s,
                ramp_step_s=ramp_step_s,
                tile_index=tile_index,
                note=note,
            )
            if frame is not None and len(frame):
                frames.append(frame)
    finally:
        _lanes_off(tile)
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if len(combined):
        save_run(combined, data_dir, "clm per test", params={"note": note})
    return combined


def _ask_per_fibers_dual():
    """Yes/No: connect A→ER-1000 and B→ER-100. False if No or dialog closed."""
    import tkinter as tk
    from tkinter import messagebox

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        return bool(
            messagebox.askyesno(
                "PER fiber hookup",
                "All CLM channels are off.\n\n"
                "Connect CLM port A to per_meter (ER-1000) input.\n"
                "Connect CLM port B to per_meter_er100 (ER-100) input.\n\n"
                "Click Yes when both fibers are connected.",
                parent=root,
            )
        )
    finally:
        root.destroy()


def per_over_temp_3_dual_meter(
    tile,
    tec_arroyo,
    per_meter,
    per_meter_er100,
    data_dir,
    channels=range(LANES_PER_BANK),
    temp0=25,
    temp1=45,
    channel_s=300,
    channel_settle_s=20,
    ramp_step_s=1.0,
    tile_index=None,
    note="",
    log=print,
):
    """PER vs Arroyo one-way ramp with both meters at once.

    Even channels ramp temp0 to temp1. Odd channels ramp temp1 to temp0.
    Each ramp lasts channel_s. For each channel index, enables bank-A and
    bank-B lanes together. ``per_meter`` (ER-1000) records bank A;
    ``per_meter_er100`` records bank B. Thorlabs TEC is not used.
    """
    channels = list(channels)
    if channel_s <= 0:
        raise ValueError("channel_s must be positive")
    if channel_settle_s < 0:
        raise ValueError("channel_settle_s must be zero or positive")
    if ramp_step_s <= 0 or ramp_step_s > channel_s:
        raise ValueError("ramp_step_s must be positive and no more than channel_s")
    columns = [
        "time", "et_0", "et_chEnb", "et_tecAdj", "temp_tgt",
        "temp", "temp_thorlab", "temp_arroyo",
        "curr_thorlab", "volt_thorlab", "curr_arroyo", "volt_arroyo",
        "bank", "ch", "tile_index", "per_dB", "theta", "pow",
    ]
    path = _new_csv_path(data_dir, "per over temp 3 dual meter")
    writer = _RunWriter(path, columns, {"note": note})
    log(f"Writing {path}")
    t0 = time.time()
    started = False
    try:
        for meter in (per_meter, per_meter_er100):
            meter.stop(flush=True)
            meter.set_mode("er")
            meter.set_capture(True)
        ensure_arroyo_tec_enabled(tec_arroyo, log=log)
        if channels:
            first_start, _first_end = channel_ramp_limits(channels[0], temp0, temp1)
            set_temperature_arroyo(tec_arroyo, first_start)
        for channel in tqdm(channels, desc="channel"):
            ramp_start, ramp_end = channel_ramp_limits(channel, temp0, temp1)
            set_temperature_arroyo(tec_arroyo, ramp_start)
            log(
                f"ch={channel} ramp {ramp_start:.1f} -> {ramp_end:.1f} C "
                f"over {channel_s:.0f}s"
            )
            # Bank A and bank B same channel index, both meters live.
            mca.clm_lane_enable(
                tile,
                chEnableMask=lane_mask(
                    [channel, LANES_PER_BANK + channel]
                ),
            )
            t_ch = time.time()
            settle_deadline = time.monotonic() + channel_settle_s
            while time.monotonic() < settle_deadline:
                time.sleep(min(0.5, settle_deadline - time.monotonic()))
                if started:
                    per_meter._discard_input()
                    per_meter_er100._discard_input()
            if not started:
                per_meter.start()
                per_meter_er100.start()
                started = True
            ramp_mark = time.monotonic()
            t_tec = time.time()
            next_write = ramp_mark
            commanded = float(ramp_start)
            while True:
                elapsed = time.monotonic() - ramp_mark
                if elapsed >= channel_s:
                    break
                if time.monotonic() >= next_write:
                    commanded = one_way_setpoint(
                        elapsed, channel_s, ramp_start, ramp_end
                    )
                    set_temperature_arroyo(tec_arroyo, commanded)
                    next_write = time.monotonic() + ramp_step_s
                remain = channel_s - (time.monotonic() - ramp_mark)
                slice_s = min(next_write - time.monotonic(), remain)
                if slice_s <= 0:
                    continue
                readings_a, readings_b = _collect_er_both(
                    per_meter, per_meter_er100, slice_s
                )
                now = time.time()
                tec_state = read_tec_state_arroyo(tec_arroyo)
                rows = []
                for bank, readings in ((0, readings_a), (1, readings_b)):
                    for reading in readings:
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
                            "pow": _reading_power(reading),
                        }
                        row.update(tec_state)
                        rows.append(row)
                if rows:
                    writer.append(rows)
                    log(
                        f"et={rows[0]['et_0']} ch={channel} "
                        f"T={commanded:.2f} "
                        f"A_samples={len(readings_a)} B_samples={len(readings_b)}"
                    )
            set_temperature_arroyo(tec_arroyo, ramp_end)
            _lanes_off(tile)
        set_temperature_arroyo(tec_arroyo, temp0)
    except KeyboardInterrupt:
        log(f"Interrupted. Partial data: {path}")
    except Exception:
        log(f"Partial data: {path}")
        raise
    finally:
        if started:
            try:
                per_meter.stop(flush=True)
            except Exception:
                pass
            try:
                per_meter_er100.stop(flush=True)
            except Exception:
                pass
        _lanes_off(tile)
    frame = _load_saved(path)
    if len(frame) and "et_0" in frame.columns:
        frame["et_s"] = frame["et_0"]
        frame["et_min"] = frame["et_0"] / 60
    return frame


def clm_per_test_dual_meter(
    tile,
    tec_arroyo,
    per_meter,
    per_meter_er100,
    data_dir,
    cfg=None,
    tb=None,
    channels=range(LANES_PER_BANK),
    temp0=25,
    temp1=45,
    channel_s=300,
    channel_settle_s=20,
    ramp_step_s=1.0,
    tile_index=None,
    serial_number="",
    prepare_tile=True,
    settle_s=30,
    note="",
    log=print,
):
    """Acceptance PER: Arroyo-only ramp, dual meters, one fiber-confirm popup.

    After tile init, enables MUX dither on both banks. Then disables all
    channels, prompts for A→ER-1000 and B→ER-100, and runs
    per_over_temp_3_dual_meter. Returns ``(raw_df, per_summary_df)``.
    """
    if tile is None:
        raise ValueError("tile handle is required")
    if tec_arroyo is None:
        raise ValueError("tec_arroyo handle is required")
    if per_meter is None or per_meter_er100 is None:
        raise ValueError("both per_meter and per_meter_er100 are required")
    if tile_index is None and cfg is not None:
        tile_index = cfg.get("tile_index")

    log(f"PER dual-meter test tile {tile_index}" + (f" ({note})" if note else ""))
    ensure_arroyo_tec_enabled(tec_arroyo, log=log)
    if prepare_tile:
        if cfg is None or tb is None:
            raise ValueError("prepare_tile requires tb and cfg")
        tile = prepare_tile_mission(
            tb, tile, cfg, tile_index=tile_index, settle_s=settle_s, log=log
        )
        set_mux_dither_enable(tile, True, log=log)

    _lanes_off(tile)
    log("All 16 channels are off.")
    if not _ask_per_fibers_dual():
        log("PER fiber hookup cancelled (No or dialog closed). Skipping PER.")
        empty = pd.DataFrame()
        return empty, channel_min_per(empty)

    raw = per_over_temp_3_dual_meter(
        tile,
        tec_arroyo,
        per_meter,
        per_meter_er100,
        data_dir,
        channels=channels,
        temp0=temp0,
        temp1=temp1,
        channel_s=channel_s,
        channel_settle_s=channel_settle_s,
        ramp_step_s=ramp_step_s,
        tile_index=tile_index,
        note=note,
        log=log,
    )
    if len(raw):
        tile_tag = tile_index if tile_index is not None else "x"
        csv_path = save_run(
            raw,
            data_dir,
            f"clm_per_test_dual_meter_tile{tile_tag}",
            params={"note": note, "tile_index": tile_index},
        )
        log(f"PER raw CSV: {csv_path}")
        try:
            import clm_plots as plots

            for bank, bank_name in ((0, "A"), (1, "B")):
                subset = raw[raw["bank"] == bank]
                if subset.empty:
                    continue
                stem = f"per_tile{tile_tag}_bank{bank_name}"
                result = plots.plot_per_vs_temperature(
                    subset,
                    output_dir=data_dir,
                    file_stem=stem,
                    show=False,
                    serial_number=serial_number,
                    tile_index=tile_index,
                    bank=bank_name,
                )
                paths = result[1] if isinstance(result, tuple) else []
                for path in paths:
                    log(f"PER plot: {path}")
        except Exception as exc:
            log(f"WARNING: PER plot save failed: {exc}")
    summary = channel_min_per(raw)
    log(f"PER summary tile {tile_index}:\n{summary}")
    return raw, summary


def _report_bank(bank):
    if bank in ("A", "B"):
        return bank
    return {0: "A", 1: "B"}[int(bank)]


def _min_inlier_per(group, min_points=20):
    """Minimum per_dB after IsolationForest drops outliers in (theta, per_dB)."""
    from sklearn.ensemble import IsolationForest

    clean = group.dropna(subset=["theta", "per_dB"])
    if len(clean) == 0:
        return None
    if len(clean) < min_points:
        return float(clean["per_dB"].min())
    features = clean.loc[:, ["theta", "per_dB"]].to_numpy(dtype=float)
    labels = IsolationForest(contamination="auto", random_state=0).fit_predict(features)
    inliers = clean.iloc[labels == 1]["per_dB"]
    if len(inliers) == 0:
        inliers = clean["per_dB"]
    return float(inliers.min())


def channel_min_per(df, min_points=20):
    """One report row per tile, bank, and channel.

    min PER is the minimum inlier per_dB. Bank 0/1 is written as A/B.
    """
    columns = ["tile", "bank", "channel", "min PER"]
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=columns)
    work = df.copy()
    if "tile_index" in work.columns:
        work["tile"] = work["tile_index"]
    elif "idx_tile" not in work.columns and "tile" not in work.columns:
        work["tile"] = None
    elif "idx_tile" in work.columns and "tile" not in work.columns:
        work["tile"] = work["idx_tile"]
    rows = []
    grouped = work.groupby(["tile", "bank", "ch"], sort=True, dropna=False)
    for (tile, bank, channel), group in grouped:
        tile_value = None if pd.isna(tile) else int(tile)
        rows.append({
            "tile": tile_value,
            "bank": _report_bank(bank),
            "channel": int(channel),
            "min PER": _min_inlier_per(group, min_points=min_points),
        })
    return pd.DataFrame(rows, columns=columns)
