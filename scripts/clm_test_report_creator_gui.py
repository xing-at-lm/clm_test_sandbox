# -*- coding: utf-8 -*-
"""Build a CLM test report from CSVs already saved in a results folder.

One tab:
  - results folder under lab_config data_dir
  - spec yaml (any file name)
  - checklist of report sections
  - Run writes docx, html, and pdf into that folder
"""

from __future__ import annotations

import re
import sys
import threading
import traceback
from datetime import datetime
from pathlib import Path

import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk


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

# Report metric, yaml key, which CSV family supplies it.
SECTIONS = (
    ("wavelength error", "wavelength_error", "power"),
    ("power error", "power_error", "power"),
    ("15 min power stability", "power_stability", "power"),
    (clm_exp.MUX_DITHER_OFF_STABILITY, "power_stability", "mux_dither_off"),
    ("min PER", "PER", "per"),
)
POWER_METRICS = tuple(name for name, _key, kind in SECTIONS if kind == "power")
DITHER_OFF_METRIC = clm_exp.MUX_DITHER_OFF_STABILITY
PER_METRIC = "min PER"

_FOLDER_RE = re.compile(
    r"^_(?P<sn>.+)_(?P<day>\d+)_(?P<month>\d+)_(?P<year>\d+)$"
)
_TILE_STAMP_RE = re.compile(
    r"tile(\d+)_(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})$"
)
POWER_GLOB = "clm_power_n_wavelength_stability_tile*.csv"
DITHER_OFF_GLOB = "clm_power_n_wavelength_stability_mux_dither_off_tile*.csv"
PER_GLOB = "clm_per_test_dual_meter_tile*.csv"


def data_root() -> Path:
    cfg = clm_exp.load_config(SCRIPT_DIR / "lab_config.yaml")
    return Path(cfg.get("data_dir", "."))


def result_folders(root: Path | None = None) -> list[Path]:
    """Immediate subfolders of data_dir, newest first."""
    root = data_root() if root is None else Path(root)
    if not root.is_dir():
        return []
    folders = [path for path in root.iterdir() if path.is_dir()]
    folders.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return folders


def folder_identity(folder: Path) -> tuple[str, str]:
    """Serial number and mm/dd/YYYY from a ``_SN_day_month_year`` folder."""
    match = _FOLDER_RE.match(folder.name)
    if not match:
        raise ValueError(
            f"Folder name {folder.name!r} is not _serial_day_month_year."
        )
    serial = match.group("sn")
    day = int(match.group("day"))
    month = int(match.group("month"))
    year = int(match.group("year"))
    if not (1 <= month <= 12 and 1 <= day <= 31):
        raise ValueError(f"Folder name {folder.name!r} has an invalid date.")
    return serial, f"{month:02d}/{day:02d}/{year}"


def newest_per_tile(folder: Path, pattern: str) -> list[Path]:
    """Newest matching CSV for each tile, using the timestamp in the file name."""
    chosen: dict[str, tuple[str, Path]] = {}
    for path in Path(folder).glob(pattern):
        match = _TILE_STAMP_RE.search(path.stem)
        if not match:
            continue
        tile, stamp = match.group(1), match.group(2)
        current = chosen.get(tile)
        if current is None or stamp > current[0]:
            chosen[tile] = (stamp, path)
    return [item[1] for _tile, item in sorted(chosen.items(), key=lambda pair: int(pair[0]))]


def _read_csvs(paths: list[Path], log) -> pd.DataFrame:
    frames = []
    for path in paths:
        log(f"Reading {path.name}")
        frames.append(pd.read_csv(path))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _keep_metrics(frame: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    columns = ["tile", "bank", "channel", *[name for name in metrics if name in frame.columns]]
    return frame.loc[:, columns]


def load_section_results(
    folder: Path,
    sections: list[str],
    power_tgt_mw=None,
    log=print,
) -> pd.DataFrame:
    """Summarize the newest power and PER CSVs, keeping only checked metrics."""
    folder = Path(folder)
    frames = []
    power_metrics = [name for name in sections if name in POWER_METRICS]
    if power_metrics:
        paths = newest_per_tile(folder, POWER_GLOB)
        if not paths:
            log("No power and wavelength CSV for the checked power sections. Skipping them.")
        else:
            raw = _read_csvs(paths, log)
            if power_tgt_mw is None:
                raise ValueError(
                    "clm_test_spec.target_power is required to score power error."
                )
            log(f"Power error is versus {float(power_tgt_mw):g} mW from the spec file.")
            summary = clm_exp.summarize_clm_stability(raw, power_tgt_mw=power_tgt_mw)
            summary = _keep_metrics(summary, power_metrics)
            if summary.empty:
                log("Power and wavelength CSVs had no samples after the 60 s startup skip.")
            else:
                frames.append(summary)

    if DITHER_OFF_METRIC in sections:
        paths = newest_per_tile(folder, DITHER_OFF_GLOB)
        if not paths:
            log("No MUX dither off CSV for power stability with MUX dither off. Skipping it.")
        else:
            raw = _read_csvs(paths, log)
            if power_tgt_mw is None:
                raise ValueError(
                    "clm_test_spec.target_power is required to score power stability."
                )
            log("Power stability with MUX dither off uses the 6 min dither-off CSV.")
            summary = clm_exp.summarize_clm_stability(raw, power_tgt_mw=power_tgt_mw)
            summary = clm_exp.stability_report_frame(summary, DITHER_OFF_METRIC)
            summary = _keep_metrics(summary, [DITHER_OFF_METRIC])
            if summary.empty:
                log("MUX dither off CSVs had no samples after the 60 s startup skip.")
            else:
                frames.append(summary)

    if PER_METRIC in sections:
        paths = newest_per_tile(folder, PER_GLOB)
        if not paths:
            log("No PER CSV for min PER. Skipping that section.")
        else:
            raw = _read_csvs(paths, log)
            summary = clm_exp.channel_min_per(raw)
            summary = _keep_metrics(summary, [PER_METRIC])
            if summary.empty:
                log("PER CSV had no channel rows.")
            else:
                frames.append(summary)

    if not frames:
        return pd.DataFrame()
    if len(frames) == 1:
        results = frames[0].copy()
    else:
        results = frames[0]
        for frame in frames[1:]:
            results = results.merge(frame, on=["tile", "bank", "channel"], how="outer")
    for col in ("tile", "channel"):
        if col in results.columns:
            results[col] = pd.to_numeric(results[col], errors="coerce").astype("Int64")
    return results


def spec_for_sections(spec_path: Path, sections: list[str]) -> tuple[dict, dict]:
    """Limits for the checked sections, plus the loaded yaml dict."""
    loaded = clm_exp.load_test_spec(spec_path)
    yaml_spec = loaded.get("clm_test_spec", {}) or {} if isinstance(loaded, dict) else {}
    key_for = {name: key for name, key, _kind in SECTIONS}
    out = {}
    for name in sections:
        entry = yaml_spec.get(key_for[name])
        if (
            isinstance(entry, (list, tuple))
            and len(entry) == 2
            and isinstance(entry[0], (list, tuple))
            and len(entry[0]) == 2
        ):
            out[name] = [list(entry[0]), str(entry[1])]
        else:
            raise ValueError(
                f"{spec_path.name} clm_test_spec.{key_for[name]} must be "
                "[[low, high], unit]. "
                f"Read {entry!r}."
            )
    return out, loaded if isinstance(loaded, dict) else {}


def _fw_version_label(spec: dict) -> str:
    filename = str((spec.get("firmware") or {}).get("filename", ""))
    marker = "_v"
    if marker in filename:
        return filename.rsplit(marker, 1)[-1].split(".bin", 1)[0]
    return filename


def _target_power_label(spec: dict) -> str:
    section = spec.get("clm_test_spec") or {}
    if "target_power" not in section:
        return ""
    return f"{float(section['target_power']):g} mW/ch"


def create_report(folder: Path, spec_path: Path, sections: list[str], log=print) -> dict:
    """Write the report into ``folder``. Returns the path dict from the writer."""
    folder = Path(folder)
    spec_path = Path(spec_path)
    if not folder.is_dir():
        raise ValueError(f"Results folder does not exist: {folder}")
    if not spec_path.is_file():
        raise ValueError(f"Spec file does not exist: {spec_path}")
    if not sections:
        raise ValueError("Select at least one report section.")

    serial, test_date = folder_identity(folder)
    spec_config, loaded = spec_for_sections(spec_path, sections)
    power_tgt_mw = None
    if any(name in POWER_METRICS or name == DITHER_OFF_METRIC for name in sections):
        section = loaded.get("clm_test_spec") or {}
        if "target_power" not in section:
            raise ValueError("clm_test_spec.target_power is missing from the spec file.")
        power_tgt_mw = float(section["target_power"])
    results = load_section_results(
        folder, sections, power_tgt_mw=power_tgt_mw, log=log
    )
    if results.empty:
        raise ValueError("No reportable rows in the selected folder.")

    present = [name for name in sections if name in results.columns]
    missing = [name for name in sections if name not in present]
    for name in missing:
        log(f"Section {name!r} has no data and is left out of the report.")
    if not present:
        raise ValueError("None of the checked sections had data.")

    spec_config = {name: spec_config[name] for name in present if name in spec_config}
    if not spec_config:
        raise ValueError(f"No limits for the checked sections in {spec_path.name}.")
    for name, limits in spec_config.items():
        low, high = limits[0]
        log(f"Pass/fail {name}: {low} to {high} {limits[1]} from {spec_path.name}")

    dict_test = {
        "test date": test_date,
        "tester": "",
        "target power": _target_power_label(loaded),
        "FW version": _fw_version_label(loaded),
    }
    dut = {"Serial Number": serial, "Part Number": ""}
    created = datetime.now()
    file_suffix = (
        f"_created_on_{created.hour}-{created.day}-{created.month}-{created.year}"
    )
    log(f"Generating test report in {folder} ...")
    paths = clm_report.generate_clm_test_report(
        dict_test,
        dut,
        spec_config,
        results=results,
        output_dir=folder,
        file_suffix=file_suffix,
    )
    for kind, path in paths.items():
        if path is not None:
            log(f"Report {kind}: {path}")
    return paths


class ClmTestReportCreatorGui(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("CLM Test Report Creator")
        self._apply_ui_fonts()
        self.geometry("900x640")
        self.minsize(760, 520)

        self.folder_var = tk.StringVar()
        self.spec_var = tk.StringVar()
        self._section_vars = {
            name: tk.BooleanVar(value=True) for name, _key, _kind in SECTIONS
        }
        self._folders: list[Path] = []
        self._busy = False

        self._build()
        self._reload_folders()
        self._seed_spec()

    def _apply_ui_fonts(self) -> None:
        import tkinter.font as tkfont

        for name in ("TkDefaultFont", "TkTextFont", "TkFixedFont"):
            tkfont.nametofont(name).configure(size=12)

    def _build(self) -> None:
        style = ttk.Style(self)
        style.configure(
            "TNotebook.Tab",
            font=("Segoe UI", 12, "bold"),
            padding=[16, 6],
        )
        notebook = ttk.Notebook(self)
        notebook.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        notebook.add(self._build_report_tab(notebook), text="  Test report  ")

        log_frame = ttk.LabelFrame(self, text="Log")
        log_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))
        self.log_text = scrolledtext.ScrolledText(log_frame, height=12, state=tk.DISABLED)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

    def _build_report_tab(self, parent) -> ttk.Frame:
        frame = ttk.Frame(parent, padding=12)

        ttk.Label(frame, text="Results folder").grid(row=0, column=0, sticky="w")
        self.folder_combo = ttk.Combobox(
            frame, textvariable=self.folder_var, state="readonly", width=62
        )
        self.folder_combo.grid(row=1, column=0, sticky="ew", pady=(4, 12))
        ttk.Button(frame, text="Refresh", command=self._reload_folders).grid(
            row=1, column=1, padx=(8, 0), sticky="w"
        )

        ttk.Label(frame, text="Spec file").grid(row=2, column=0, sticky="w")
        self.spec_combo = ttk.Combobox(
            frame, textvariable=self.spec_var, state="readonly", width=62
        )
        self.spec_combo.grid(row=3, column=0, sticky="ew", pady=(4, 12))
        ttk.Button(frame, text="Browse", command=self._browse_spec).grid(
            row=3, column=1, padx=(8, 0), sticky="w"
        )

        checks = ttk.LabelFrame(frame, text="Report sections", padding=8)
        checks.grid(row=4, column=0, columnspan=2, sticky="ew")
        for name, _key, _kind in SECTIONS:
            ttk.Checkbutton(checks, text=name, variable=self._section_vars[name]).pack(
                anchor="w", pady=2
            )

        self.run_button = ttk.Button(frame, text="Run", command=self._on_run)
        self.run_button.grid(row=5, column=0, sticky="w", pady=(16, 0))

        frame.columnconfigure(0, weight=1)
        return frame

    def _append_log(self, message: str) -> None:
        stamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, f"{stamp}  {message}\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)

    def _reload_folders(self) -> None:
        root = data_root()
        self._folders = result_folders(root)
        names = [path.name for path in self._folders]
        self.folder_combo["values"] = names
        if names and self.folder_var.get() not in names:
            self.folder_var.set(names[0])
        self._append_log(f"{len(names)} results folder(s) under {root}")

    def _selected_folder(self) -> Path | None:
        name = self.folder_var.get().strip()
        for path in self._folders:
            if path.name == name:
                return path
        return None

    def _seed_spec(self) -> None:
        default = SCRIPT_DIR / "clm_test_spec.yaml"
        values = [str(default)] if default.is_file() else []
        self.spec_combo["values"] = values
        if values:
            self.spec_var.set(values[0])

    def _browse_spec(self) -> None:
        chosen = filedialog.askopenfilename(
            title="Spec file",
            initialdir=str(SCRIPT_DIR),
            filetypes=[("YAML", "*.yaml *.yml"), ("All files", "*.*")],
        )
        if not chosen:
            return
        values = list(self.spec_combo["values"])
        if chosen not in values:
            values.append(chosen)
            self.spec_combo["values"] = values
        self.spec_var.set(chosen)

    def _selected_sections(self) -> list[str]:
        return [name for name, _key, _kind in SECTIONS if self._section_vars[name].get()]

    def _on_run(self) -> None:
        if self._busy:
            messagebox.showinfo("Busy", "A report is already being created.")
            return
        folder = self._selected_folder()
        if folder is None:
            messagebox.showwarning("Results folder", "Select a results folder.")
            return
        spec_path = Path(self.spec_var.get().strip())
        if not spec_path.is_file():
            messagebox.showwarning("Spec file", "Select a spec yaml file.")
            return
        sections = self._selected_sections()
        if not sections:
            messagebox.showwarning("Report sections", "Select at least one section.")
            return

        self._busy = True
        self.run_button.configure(state=tk.DISABLED)
        self._append_log(f"Report for {folder.name} using {spec_path.name}")

        def work() -> None:
            try:
                create_report(folder, spec_path, sections, log=self._log_from_thread)
            except Exception as exc:
                detail = traceback.format_exc()
                self.after(0, lambda: self._append_log(detail.rstrip()))
                self.after(
                    0,
                    lambda: messagebox.showerror("Test report", str(exc)),
                )
            finally:
                self.after(0, self._run_finished)

        threading.Thread(target=work, daemon=True, name="clm-report-creator").start()

    def _log_from_thread(self, message: str) -> None:
        self.after(0, lambda: self._append_log(message))

    def _run_finished(self) -> None:
        self._busy = False
        self.run_button.configure(state=tk.NORMAL)


def main() -> None:
    app = ClmTestReportCreatorGui()
    app.mainloop()


if __name__ == "__main__":
    main()
