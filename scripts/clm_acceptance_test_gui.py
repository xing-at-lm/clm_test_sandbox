# -*- coding: utf-8 -*-
"""CLM acceptance-test GUI.

Three tabs:
  1. DUT and Operator  - serial number, operator, comment
  2. Test station      - start / close the instruments listed in lab_config
  3. Acceptance test   - checkable list of tests to run later

Add a station instrument in clm_experiments._station_handlers() and list its
name under test_station.instruments in lab_config.yaml. Add an acceptance
test by appending an AcceptanceTest to ACCEPTANCE_TESTS.
"""

from __future__ import annotations

import ast
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import tkinter as tk
from tkinter import messagebox, scrolledtext, ttk


def _script_dir() -> Path:
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

import clm_experiments as clm_exp
import generate_clm_test_report as clm_report


def _format_elapsed(seconds: float) -> str:
    """Compact human elapsed time, e.g. ``12m 34s`` or ``45.2s``."""
    total = max(0.0, float(seconds))
    if total < 60.0:
        return f"{total:.1f}s"
    minutes, secs = divmod(int(round(total)), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    return f"{minutes}m {secs}s"


# Report metric -> clm_test_spec.yaml key. The 6 min dither-off section
# uses the same power_stability window as the 15 min section.
_REPORT_LIMIT_SOURCE = {
    "wavelength error": "wavelength_error",
    "power error": "power_error",
    "15 min power stability": "power_stability",
    clm_exp.MUX_DITHER_OFF_STABILITY: "power_stability",
    "min PER": "PER",
}

# ---------------------------------------------------------------------------
# Acceptance tests. Append new entries; the GUI builds checkboxes from this.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AcceptanceTest:
    test_id: str
    label: str
    description: str = ""
    run: Callable[["StationSession", "DutMeta"], None] | None = None


def _stub_test(session: "StationSession", meta: "DutMeta", name: str) -> None:
    session.log(
        f"[{name}] not implemented yet. "
        f"DUT={meta.serial_number!r} operator={meta.operator!r}"
    )


def _tile_from_session(session: "StationSession", tile_index: int):
    tb = session.tb
    if tb is None:
        raise RuntimeError("Start the test station before running tile tests.")
    return tb.module.packages[0].tiles[tile_index]


def run_fw_upgrade(session: "StationSession", meta: "DutMeta") -> None:
    """Flash firmware on each selected tile (one at a time)."""
    if session.tb is None:
        raise RuntimeError("Start the test station before running FW upgrade.")
    note = meta.comment or f"acceptance FW upgrade DUT={meta.serial_number}"
    session.log(
        f"FW upgrade starting for DUT {meta.serial_number!r}, "
        f"tiles {meta.tile_indexes}, operator {meta.operator!r}"
    )
    for idx in meta.tile_indexes:
        tile = _tile_from_session(session, idx)
        session.log(f"FW upgrade tile {idx} ...")
        fw_path = clm_exp.clm_fw_update(tile, note=f"{note} tile={idx}")
        session.log(f"FW upgrade tile {idx} finished: {fw_path}")


def run_tap_ratio_cal_osa(session: "StationSession", meta: "DutMeta") -> None:
    """Recalibrate PIC tap ratios one tile at a time via OSA + coupler loss."""
    osa = session.get("osa")
    if session.tb is None or osa is None:
        raise RuntimeError(
            "Start the test station with testboard and osa before tap-ratio calibration."
        )
    note = meta.comment or f"acceptance tap ratio OSA cal DUT={meta.serial_number}"
    session.log(
        f"Tap ratio OSA cal starting for DUT {meta.serial_number!r}, "
        f"tiles {meta.tile_indexes}, operator {meta.operator!r}"
    )
    total_rows = 0
    for idx in meta.tile_indexes:
        tile = _tile_from_session(session, idx)
        session.log(f"Tap ratio OSA cal tile {idx} ...")
        summary = clm_exp.clm_tap_ratio_cal_osa(
            tile,
            osa,
            session.cfg,
            settle_s=60,
            write_fru=True,
            note=f"{note} tile={idx}",
            tile_index=idx,
            tb=session.tb,
            log=session.log,
            n_samples=3,
            data_dir=session.output_dir(),
            dut_sn=meta.serial_number,
            operator=meta.operator,
        )
        total_rows += len(summary)
        session.log(f"Tap ratio OSA cal tile {idx} finished: {len(summary)} channel rows")
    session.log(f"Tap ratio OSA cal finished: {total_rows} channel rows, FRU updated")


def run_power_and_frequency(session: "StationSession", meta: "DutMeta") -> None:
    """15-min CLM power / wavelength stability; stores summary for the final report."""
    osa = session.get("osa")
    if session.tb is None or osa is None:
        raise RuntimeError(
            "Start the test station with testboard and osa before power and frequency."
        )
    note = meta.comment or f"acceptance power/freq DUT={meta.serial_number}"
    # Tap-ratio cal (earlier in the same run) already connected fibers and fired up the tile.
    selected_ids = set(session.results.get("selected_test_ids") or [])
    prepare_tile = "tap_ratios" not in selected_ids
    session.log(
        f"Power and frequency starting for DUT {meta.serial_number!r}, "
        f"tiles {meta.tile_indexes}, operator {meta.operator!r}"
        + (
            " (skipping re-init/power-on; tap-ratio cal already prepared the tile)"
            if not prepare_tile
            else ""
        )
    )
    summaries = []
    for idx in meta.tile_indexes:
        tile = _tile_from_session(session, idx)
        session.log(f"Power and frequency tile {idx} (15 min) ...")
        _raw_df, summary = clm_exp.clm_power_and_frequency_test(
            tile,
            osa,
            session.cfg,
            tb=session.tb,
            tile_index=idx,
            test_duration=60 * 15,
            sample_interval=10,
            settle_s=30,
            prepare_tile=prepare_tile,
            data_dir=session.output_dir(),
            serial_number=meta.serial_number,
            note=f"{note} tile={idx}",
            log=session.log,
        )
        summaries.append(summary)
        session.log(
            f"Power and frequency tile {idx} finished: {len(summary)} channel rows"
        )
    if summaries:
        combined = pd.concat(summaries, ignore_index=True)
        _append_summary(session, "stability_summary", combined)
        stored = session.results["stability_summary"]
        session.log(
            f"Power and frequency stored {len(stored)} summary rows for report"
        )


def run_power_and_frequency_mux_dither_off(
    session: "StationSession", meta: "DutMeta"
) -> None:
    """Turn MUX dither off, then collect the power/frequency trace for 6 min."""
    osa = session.get("osa")
    if session.tb is None or osa is None:
        raise RuntimeError(
            "Start the test station with testboard and osa before "
            "power and frequency with MUX dither off."
        )
    note = meta.comment or f"acceptance power/freq MUX dither off DUT={meta.serial_number}"
    selected_ids = set(session.results.get("selected_test_ids") or [])
    prepare_tile = not ({"tap_ratios", "power_and_frequency"} & selected_ids)
    session.log(
        f"Power and frequency with MUX dither off starting for DUT "
        f"{meta.serial_number!r}, tiles {meta.tile_indexes}, "
        f"operator {meta.operator!r}"
        + (
            " (skipping re-init/power-on; an earlier test already prepared the tile)"
            if not prepare_tile
            else ""
        )
    )
    summaries = []
    for idx in meta.tile_indexes:
        tile = _tile_from_session(session, idx)
        session.log(f"Power and frequency with MUX dither off, tile {idx} (6 min) ...")
        _raw_df, summary = clm_exp.clm_power_and_frequency_test(
            tile,
            osa,
            session.cfg,
            tb=session.tb,
            tile_index=idx,
            test_duration=60 * 6,
            sample_interval=10,
            settle_s=30,
            prepare_tile=prepare_tile,
            data_dir=session.output_dir(),
            serial_number=meta.serial_number,
            note=f"{note} tile={idx}",
            log=session.log,
            mux_dither=False,
            csv_stem=f"clm_power_n_wavelength_stability_mux_dither_off_tile{idx}",
            plot="power",
            plot_heading="power vs time, MUX dither off",
            log_name="Power and frequency, MUX dither off",
        )
        summaries.append(
            clm_exp.stability_report_frame(summary, clm_exp.MUX_DITHER_OFF_STABILITY)
        )
        session.log(
            f"Power and frequency with MUX dither off, tile {idx} finished: "
            f"{len(summaries[-1])} channel rows"
        )
    if summaries:
        combined = pd.concat(summaries, ignore_index=True)
        _append_summary(session, "mux_dither_off_summary", combined)
        stored = session.results.get("mux_dither_off_summary")
        if stored is not None:
            session.log(
                f"MUX dither off stored {len(stored)} summary rows for report"
            )


def _report_spec_from_yaml(metric_names: list[str] | None = None) -> dict:
    """Build report limit tables from clm_test_spec.yaml (fallback to report defaults)."""
    try:
        yaml_spec = clm_exp.load_test_spec().get("clm_test_spec", {}) or {}
    except Exception:
        yaml_spec = {}
    out = {}
    for report_name, yaml_key in _REPORT_LIMIT_SOURCE.items():
        if metric_names is not None and report_name not in metric_names:
            continue
        entry = yaml_spec.get(yaml_key)
        if (
            isinstance(entry, (list, tuple))
            and len(entry) == 2
            and isinstance(entry[0], (list, tuple))
            and len(entry[0]) == 2
        ):
            out[report_name] = [list(entry[0]), str(entry[1])]
            continue
        fallback_name = (
            report_name
            if report_name in clm_report.spec_config_dict
            else "15 min power stability"
        )
        if fallback_name in clm_report.spec_config_dict:
            out[report_name] = clm_report.spec_config_dict[fallback_name]
    return out


def run_per_test(session: "StationSession", meta: "DutMeta") -> None:
    """Arroyo-only dual-meter PER; stores min-PER summary for the final report."""
    tec_arroyo = session.get("tec_arroyo")
    per_meter = session.get("per_meter")
    per_meter_er100 = session.get("per_meter_er100")
    if session.tb is None or tec_arroyo is None:
        raise RuntimeError(
            "Start the test station with testboard and tec_arroyo before PER."
        )
    if per_meter is None or per_meter_er100 is None:
        raise RuntimeError(
            "Start the test station with per_meter and per_meter_er100 before PER."
        )
    note = meta.comment or f"acceptance PER DUT={meta.serial_number}"
    session.log(
        f"PER starting for DUT {meta.serial_number!r}, "
        f"tiles {meta.tile_indexes}, operator {meta.operator!r}"
    )
    summaries = []
    for idx in meta.tile_indexes:
        tile = _tile_from_session(session, idx)
        session.log(f"PER dual-meter tile {idx} ...")
        _raw, summary = clm_exp.clm_per_test_dual_meter(
            tile,
            tec_arroyo,
            per_meter,
            per_meter_er100,
            session.output_dir(),
            cfg=session.cfg,
            tb=session.tb,
            channels=range(8),
            temp0=25,
            temp1=50,
            channel_s=300,
            channel_settle_s=20,
            tile_index=idx,
            serial_number=meta.serial_number,
            prepare_tile=True,
            settle_s=30,
            note=f"{note} tile={idx}",
            log=session.log,
        )
        if summary is not None and len(summary):
            summaries.append(summary)
        session.log(f"PER tile {idx} finished: {len(summary)} summary rows")
    if summaries:
        combined = pd.concat(summaries, ignore_index=True)
        _append_summary(session, "per_summary", combined)
        stored = session.results["per_summary"]
        session.log(f"PER stored {len(stored)} summary rows for report")


def _append_summary(session: "StationSession", key: str, frame: pd.DataFrame) -> None:
    """Add rows to a report table. Later tiles must not replace earlier ones."""
    if frame is None or len(frame) == 0:
        return
    existing = session.results.get(key)
    if existing is not None and len(existing):
        session.results[key] = pd.concat([existing, frame], ignore_index=True)
    else:
        session.results[key] = frame


def _tile_meta(meta: "DutMeta", tile_index: int) -> "DutMeta":
    """Same DUT record, restricted to the tile now under test."""
    return DutMeta(
        serial_number=meta.serial_number,
        operator=meta.operator,
        comment=meta.comment,
        tile_indexes=[tile_index],
    )


def _fw_version_label(session: "StationSession", meta: "DutMeta") -> str:
    """Live tile FW version if readable, else version parsed from clm_test_spec.yaml."""
    if session.tb is not None and meta.tile_indexes:
        try:
            tile = _tile_from_session(session, meta.tile_indexes[0])
            version = tile.read_fw_version()
            if version is not None and str(version).strip():
                text = str(version).strip()
                # Common forms: "1.14", "v1.14", (1, 14), etc.
                if text.lower().startswith("v") and text[1:2].isdigit():
                    text = text[1:]
                return text
        except Exception as exc:
            session.log(f"Could not read tile FW version: {exc}")

    try:
        filename = str(
            clm_exp.load_test_spec().get("firmware", {}).get("filename", "")
        )
    except Exception:
        filename = ""
    # e.g. clm_rel_marlow_tile_v1.14.bin -> 1.14
    marker = "_v"
    if marker in filename:
        tail = filename.rsplit(marker, 1)[-1]
        return tail.split(".bin", 1)[0]
    return filename or ""


def generate_acceptance_report(session: "StationSession", meta: "DutMeta") -> None:
    """Write .docx / .html / .pdf after the selected acceptance tests finish."""
    frames = [
        frame
        for frame in (
            session.results.get("stability_summary"),
            session.results.get("mux_dither_off_summary"),
            session.results.get("per_summary"),
        )
        if frame is not None and len(frame)
    ]
    if not frames:
        session.log("No reportable results yet; skipping test report.")
        return
    results = frames[0].copy()
    for frame in frames[1:]:
        results = results.merge(frame, on=["tile", "bank", "channel"], how="outer")
    # Outer merge can promote tile/channel to float; keep them integers for the report.
    for col in ("tile", "channel"):
        if col in results.columns:
            results[col] = pd.to_numeric(results[col], errors="coerce").astype("Int64")

    metric_names = [
        name
        for name in (
            "wavelength error",
            "power error",
            "15 min power stability",
            clm_exp.MUX_DITHER_OFF_STABILITY,
            "min PER",
        )
        if name in results.columns
    ]
    target_mw = float(
        clm_exp.load_test_spec().get("clm_test_spec", {}).get("target_power", 8)
    )
    dict_test = {
        "test date": datetime.now().strftime("%m/%d/%Y"),
        "tester": meta.operator or "",
        "target power": f"{target_mw:g} mW/ch",
        "FW version": _fw_version_label(session, meta),
    }
    dut = {
        "Serial Number": meta.serial_number,
        "Part Number": session.cfg.get("part_number", ""),
    }
    output_dir = session.output_dir()
    session.log(f"Generating test report in {output_dir} ...")
    paths = clm_report.generate_clm_test_report(
        dict_test,
        dut,
        _report_spec_from_yaml(metric_names),
        results=results,
        output_dir=output_dir,
    )
    session.results["report_paths"] = paths
    for kind, path in paths.items():
        if path is not None:
            session.log(f"Report {kind}: {path}")


ACCEPTANCE_TESTS: list[AcceptanceTest] = [
    AcceptanceTest(
        "fw_upgrade",
        "FW upgrade",
        "Flash the tile firmware from clm_test_spec.yaml.",
        run=run_fw_upgrade,
    ),
    AcceptanceTest(
        "tap_ratios",
        "Tap ratio (re-)calibration using OSA",
        "Compare PIC and MUX (aux) monitors to OSA+coupler loss; rewrite both tap arrays.",
        run=run_tap_ratio_cal_osa,
    ),
    AcceptanceTest(
        "power_and_frequency",
        "power and frequency",
        "15 min CLM power and wavelength stability via OSA; feeds the final report.",
        run=run_power_and_frequency,
    ),
    AcceptanceTest(
        "power_and_frequency_mux_dither_off",
        "power and frequency with MUX dither off",
        "Turn MUX dither off, then collect 6 min of the same OSA power and wavelength data. "
        "Report section is power stability with MUX dither off, using the power stability limit.",
        run=run_power_and_frequency_mux_dither_off,
    ),
    AcceptanceTest(
        "per",
        "PER",
        "Dual-meter PER vs Arroyo ramp (25-45 C, 300 s). Even channels ramp up, odd channels ramp down. A=ER-1000, B=ER-100.",
        run=run_per_test,
    ),
]


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

@dataclass
class DutMeta:
    serial_number: str = ""
    operator: str = ""
    comment: str = ""
    tile_indexes: list[int] = field(default_factory=lambda: [3])


def parse_tile_indexes(text: str) -> list[int]:
    """Parse ``[3]``, ``[3, 4]``, or ``3`` into a list of tile indexes."""
    text = text.strip()
    if not text:
        raise ValueError("Tile index cannot be empty")
    value = ast.literal_eval(text)
    if isinstance(value, int):
        indexes = [value]
    elif isinstance(value, (list, tuple)):
        indexes = [int(item) for item in value]
    else:
        raise ValueError(f"Tile index must be an int or list, got {type(value).__name__}")
    if not indexes:
        raise ValueError("Tile index list is empty")
    return indexes


@dataclass
class StationSession:
    cfg: dict
    instruments: dict[str, Any] = field(default_factory=dict)
    running: bool = False
    results: dict[str, Any] = field(default_factory=dict)
    run_dir: Path | None = None
    log_lines: list[str] = field(default_factory=list, repr=False)
    _log_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _log: Callable[[str], None] = field(default=lambda _msg: None, repr=False)

    def log(self, message: str) -> None:
        self._log(message)

    def log_text(self) -> str:
        with self._log_lock:
            if not self.log_lines:
                return ""
            return "\n".join(self.log_lines) + "\n"

    def get(self, name: str) -> Any:
        return self.instruments.get(name)

    def clear_results(self) -> None:
        self.results.clear()
        self.run_dir = None

    def output_dir(self) -> Path:
        """Per-run results folder if set, else lab_config data_dir."""
        if self.run_dir is not None:
            return self.run_dir
        return Path(self.cfg.get("data_dir", "."))

    @property
    def tile(self):
        handle = self.instruments.get("testboard")
        if isinstance(handle, dict):
            return handle.get("tile")
        return None

    @property
    def tb(self):
        handle = self.instruments.get("testboard")
        if isinstance(handle, dict):
            return handle.get("tb")
        return None


def start_test_station(session: StationSession, tile_indexes: list[int] | None = None) -> None:
    """Called when Start test station is clicked."""
    if session.running:
        session.log("Test station is already running.")
        return
    if tile_indexes:
        session.cfg["tile_index"] = int(tile_indexes[0])
        session.cfg["tile_indexes"] = list(tile_indexes)
        session.log(f"Using tile index(es) {tile_indexes}")
    instruments = clm_exp.clm_start_test_station(session.cfg, log=session.log)
    session.instruments.clear()
    session.instruments.update(instruments)
    session.running = bool(instruments)
    requested = list(session.cfg.get("test_station", {}).get("instruments", []))
    failed = [name for name in requested if name not in instruments]
    if failed:
        session.log(
            f"Continuing with {len(instruments)} instrument(s). "
            f"Unavailable: {', '.join(failed)}."
        )


def close_test_station(session: StationSession) -> None:
    """Called when Close down is clicked."""
    if not session.instruments and not session.running:
        session.log("Test station is already closed.")
        return
    clm_exp.clm_close_test_station(session.instruments, log=session.log)
    session.running = False


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------

class ClmAcceptanceTestGui(tk.Tk):
    def __init__(self, cfg: dict | None = None):
        super().__init__()
        self.title("CLM Acceptance Test")
        self._apply_ui_fonts()
        self.geometry("1000x700")
        self.minsize(820, 560)
        self._set_app_icon()

        self.cfg = cfg if cfg is not None else clm_exp.load_config(SCRIPT_DIR / "lab_config.yaml")
        self.session = StationSession(cfg=self.cfg, _log=self._append_log)
        self._busy = False

        self.sn_var = tk.StringVar()
        self.operator_var = tk.StringVar()
        default_tile = self.cfg.get("tile_index", 3)
        self.tile_indexes_var = tk.StringVar(value=f"[{default_tile}]")
        self._test_vars: dict[str, tk.BooleanVar] = {
            test.test_id: tk.BooleanVar(value=False) for test in ACCEPTANCE_TESTS
        }

        self._build()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _apply_ui_fonts(self) -> None:
        """Use 12 pt for the form text. Tabs were 10; body text was 9 or 10."""
        import tkinter.font as tkfont

        for name in ("TkDefaultFont", "TkTextFont", "TkFixedFont"):
            tkfont.nametofont(name).configure(size=12)

    def _set_app_icon(self) -> None:
        icon_path = SCRIPT_DIR / "assets" / "laser_diode.png"
        try:
            self._app_icon = tk.PhotoImage(file=str(icon_path))
            self.iconphoto(True, self._app_icon)
        except Exception:
            self._app_icon = None

    def _build(self) -> None:
        style = ttk.Style(self)
        style.configure(
            "TNotebook.Tab",
            font=("Segoe UI", 12, "bold"),
            padding=[16, 6],
        )

        notebook = ttk.Notebook(self)
        notebook.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        notebook.add(self._build_dut_tab(notebook), text="  DUT and Operator  ")
        notebook.add(self._build_station_tab(notebook), text="  Test station  ")
        notebook.add(self._build_acceptance_tab(notebook), text="  Acceptance test  ")

        log_frame = ttk.LabelFrame(self, text="Log")
        log_frame.pack(fill=tk.BOTH, expand=False, padx=8, pady=(0, 8))
        self.log_text = scrolledtext.ScrolledText(log_frame, height=10, state=tk.DISABLED)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        self.status_var = tk.StringVar(value="Station: closed")
        ttk.Label(self, textvariable=self.status_var, anchor="w").pack(
            fill=tk.X, padx=8, pady=(0, 8)
        )

    def _build_dut_tab(self, parent) -> ttk.Frame:
        frame = ttk.Frame(parent, padding=12)
        frame.columnconfigure(1, weight=1)

        ttk.Label(frame, text="DUT SN").grid(row=0, column=0, sticky="w", pady=4)
        ttk.Entry(frame, textvariable=self.sn_var).grid(
            row=0, column=1, sticky="ew", pady=4, padx=(8, 0)
        )

        ttk.Label(frame, text="Operator name").grid(row=1, column=0, sticky="w", pady=4)
        ttk.Entry(frame, textvariable=self.operator_var).grid(
            row=1, column=1, sticky="ew", pady=4, padx=(8, 0)
        )

        ttk.Label(frame, text="Tile index(es)").grid(row=2, column=0, sticky="w", pady=4)
        ttk.Entry(frame, textvariable=self.tile_indexes_var).grid(
            row=2, column=1, sticky="ew", pady=4, padx=(8, 0)
        )
        ttk.Label(
            frame,
            text="Example: [3] or [3, 4]",
            foreground="#555",
        ).grid(row=3, column=1, sticky="w", padx=(8, 0))

        ttk.Label(frame, text="Comment").grid(row=4, column=0, sticky="nw", pady=4)
        self.comment_text = tk.Text(frame, height=8, wrap=tk.WORD)
        self.comment_text.grid(row=4, column=1, sticky="nsew", pady=4, padx=(8, 0))
        frame.rowconfigure(4, weight=1)
        return frame

    def _build_station_tab(self, parent) -> ttk.Frame:
        frame = ttk.Frame(parent, padding=12)
        names = self.cfg.get("test_station", {}).get("instruments", [])
        ttk.Label(
            frame,
            text="Instruments from lab_config.test_station:",
        ).pack(anchor="w")
        list_box = tk.Listbox(frame, height=max(6, len(names) + 1))
        list_box.pack(fill=tk.BOTH, expand=True, pady=8)
        for name in names:
            list_box.insert(tk.END, name)
        if not names:
            list_box.insert(tk.END, "(none configured)")

        buttons = ttk.Frame(frame)
        buttons.pack(fill=tk.X)
        self.start_button = ttk.Button(
            buttons, text="Start test station", command=self._on_start_station
        )
        self.start_button.pack(side=tk.LEFT, padx=(0, 8))
        self.close_button = ttk.Button(
            buttons, text="Close down", command=self._on_close_station
        )
        self.close_button.pack(side=tk.LEFT, padx=(0, 8))
        self.self_test_button = ttk.Button(
            buttons, text="Self test instruments", command=self._on_self_test
        )
        self.self_test_button.pack(side=tk.LEFT)
        return frame

    def _build_acceptance_tab(self, parent) -> ttk.Frame:
        frame = ttk.Frame(parent, padding=12)
        ttk.Label(
            frame,
            text="Select the tests to run.",
        ).pack(anchor="w", pady=(0, 8))

        list_frame = ttk.Frame(frame)
        list_frame.pack(fill=tk.BOTH, expand=True)
        for test in ACCEPTANCE_TESTS:
            row = ttk.Frame(list_frame)
            row.pack(fill=tk.X, pady=2)
            ttk.Checkbutton(
                row,
                text=test.label,
                variable=self._test_vars[test.test_id],
            ).pack(side=tk.LEFT)
            if test.description:
                ttk.Label(row, text=f"  — {test.description}", foreground="#555").pack(
                    side=tk.LEFT
                )

        ttk.Button(
            frame, text="Run selected tests", command=self._on_run_tests
        ).pack(anchor="w", pady=(12, 0))
        return frame

    def _dut_meta(self) -> DutMeta:
        return DutMeta(
            serial_number=self.sn_var.get().strip(),
            operator=self.operator_var.get().strip(),
            comment=self.comment_text.get("1.0", tk.END).strip(),
            tile_indexes=parse_tile_indexes(self.tile_indexes_var.get()),
        )

    def _append_log(self, message: str) -> None:
        text = str(message)
        with self.session._log_lock:
            self.session.log_lines.append(text)

        def write() -> None:
            self.log_text.configure(state=tk.NORMAL)
            self.log_text.insert(tk.END, text + "\n")
            self.log_text.see(tk.END)
            self.log_text.configure(state=tk.DISABLED)
            self.status_var.set(
                "Station: running" if self.session.running else "Station: closed"
            )

        self.after(0, write)

    def _save_acceptance_log(self) -> None:
        """Write the log pane text into the run folder. Name includes SN and date."""
        run_dir = self.session.run_dir
        serial = str(self.session.results.get("log_serial") or "").strip()
        started = self.session.results.get("log_started")
        if run_dir is None or not serial or not isinstance(started, datetime):
            return
        safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in serial) or "unknown"
        stamp = started.strftime("%m-%d-%Y_%H-%M-%S")
        path = Path(run_dir) / f"acceptance_log_{safe}_{stamp}.txt"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            self.session.log(f"Acceptance log: {path}")
            path.write_text(self.session.log_text(), encoding="utf-8")
        except Exception as exc:
            self.session.log(f"WARNING: Could not save acceptance log: {exc}")

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        state = tk.DISABLED if busy else tk.NORMAL
        self.start_button.configure(state=state)
        self.close_button.configure(state=state)
        self.self_test_button.configure(state=state)

    def _run_in_thread(self, work: Callable[[], None], label: str) -> None:
        if self._busy:
            messagebox.showinfo("Busy", "Another station action is still running.")
            return

        def runner() -> None:
            self.after(0, lambda: self._set_busy(True))
            try:
                work()
            except Exception as exc:
                err_msg = str(exc)
                self._append_log(f"{label} failed: {err_msg}")
                self._append_log(traceback.format_exc())
                self.after(
                    0,
                    lambda msg=err_msg: messagebox.showerror(label, msg),
                )
            finally:
                if label == "Acceptance test":
                    self._save_acceptance_log()
                self.after(0, lambda: self._set_busy(False))

        threading.Thread(target=runner, daemon=True).start()

    def _on_start_station(self) -> None:
        try:
            tile_indexes = parse_tile_indexes(self.tile_indexes_var.get())
        except Exception as exc:
            messagebox.showerror("Tile index(es)", str(exc))
            return

        def work() -> None:
            start_test_station(self.session, tile_indexes=tile_indexes)

        self._run_in_thread(work, "Start test station")

    def _on_close_station(self) -> None:
        self._run_in_thread(
            lambda: close_test_station(self.session),
            "Close down",
        )

    def _on_self_test(self) -> None:
        def work() -> None:
            results = clm_exp.run_equipment_self_tests(
                self.session.instruments,
                cfg=self.session.cfg,
                log=self.session.log,
            )
            lines = [
                f"{item['device']}: {'PASS' if item['passed'] else 'FAIL'}"
                for item in results
            ]
            failed = any(not item["passed"] for item in results)
            text = "\n".join(lines)
            self.after(
                0,
                lambda: (
                    messagebox.showwarning("Instrument self test", text)
                    if failed
                    else messagebox.showinfo("Instrument self test", text)
                ),
            )

        self._run_in_thread(work, "Instrument self test")

    def _on_run_tests(self) -> None:
        selected = [
            test for test in ACCEPTANCE_TESTS if self._test_vars[test.test_id].get()
        ]
        if not selected:
            messagebox.showinfo("Acceptance test", "Select at least one test.")
            return
        meta = self._dut_meta()
        if not meta.serial_number:
            messagebox.showwarning("DUT and Operator", "Enter the DUT serial number.")
            return

        selected_ids = {t.test_id for t in selected}
        # Coupler fibers are needed once per tile, just before that tile's
        # first OSA test. PER asks for its own fibers when it starts.
        coupler_test_ids = {
            "tap_ratios",
            "power_and_frequency",
            "power_and_frequency_mux_dither_off",
        }
        needs_coupler_prompt = bool(selected_ids & coupler_test_ids)

        def work() -> None:
            self.session.clear_results()
            now = datetime.now()
            self.session.results["selected_test_ids"] = sorted(selected_ids)
            self.session.results["log_serial"] = meta.serial_number
            self.session.results["log_started"] = now
            run_dir = (
                Path(self.session.cfg.get("data_dir", "."))
                / f"_{meta.serial_number}_{now.day}_{now.month}_{now.year}"
            )
            run_dir.mkdir(parents=True, exist_ok=True)
            self.session.run_dir = run_dir
            self.session.results["run_dir"] = str(run_dir)
            self.session.log(f"Results directory: {run_dir}")
            if meta.comment:
                note_path = run_dir / "operator_note.txt"
                note_path.write_text(meta.comment + "\n", encoding="utf-8")
                self.session.log(f"Operator note: {note_path}")
            self.session.log(
                f"Running {len(selected)} test(s) on each of "
                f"{meta.tile_indexes} for DUT {meta.serial_number}"
            )
            run_t0 = time.perf_counter()
            ran_any = False
            stop_run = False
            for idx in meta.tile_indexes:
                self.session.cfg["tile_index"] = int(idx)
                tile_meta = _tile_meta(meta, idx)
                self.session.log(f"=== Tile {idx} ===")
                tile_t0 = time.perf_counter()
                asked_coupler = False
                for test in selected:
                    if (
                        needs_coupler_prompt
                        and not asked_coupler
                        and test.test_id in coupler_test_ids
                    ):
                        asked_coupler = True
                        proceed = messagebox.askyesno(
                            "Laser safety",
                            f"Tile {idx}: please connect DUT output to coupler "
                            "input (A-->white, B-->blue).",
                        )
                        if not proceed:
                            self.session.log(
                                f"Laser safety: cancelled for tile {idx} "
                                "(No or dialog closed). Remaining tiles will not run."
                            )
                            stop_run = True
                            break
                    self.session.log(f"--- Tile {idx}: {test.label} ---")
                    test_t0 = time.perf_counter()
                    if test.run is None:
                        self.session.log(f"{test.label}: no runner registered")
                    else:
                        test.run(self.session, tile_meta)
                    ran_any = True
                    elapsed = time.perf_counter() - test_t0
                    self.session.log(
                        f"Tile {idx} {test.label}: elapsed {_format_elapsed(elapsed)}"
                    )
                self.session.log(
                    f"Tile {idx} elapsed {_format_elapsed(time.perf_counter() - tile_t0)}"
                )
                if stop_run:
                    break
            total_elapsed = time.perf_counter() - run_t0
            self.session.log(
                f"Selected tests total elapsed: {_format_elapsed(total_elapsed)}"
            )
            if not ran_any:
                self.session.log("No tests ran.")
                return
            self.session.log("Selected tests finished.")
            generate_acceptance_report(self.session, meta)
            self.session.log("Closing test station after acceptance run...")
            close_test_station(self.session)
            self.session.log("Acceptance run complete; test station closed.")

        self._run_in_thread(work, "Acceptance test")

    def _on_close(self) -> None:
        if self.session.running or self.session.instruments:
            if not messagebox.askyesno(
                "Close",
                "The test station is still open. Close it and exit?",
            ):
                return
            try:
                close_test_station(self.session)
            except Exception as exc:
                messagebox.showerror("Close down", str(exc))
        self.destroy()


def main() -> None:
    app = ClmAcceptanceTestGui()
    app.mainloop()


if __name__ == "__main__":
    main()
