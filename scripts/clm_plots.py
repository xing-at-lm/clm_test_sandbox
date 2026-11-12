"""Figures for CLM bench runs. These functions only read a dataframe or a CSV."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_palette("bright")


def _load(df_or_path):
    if isinstance(df_or_path, (str, Path)):
        frame = pd.read_csv(df_or_path)
    else:
        frame = df_or_path.copy()
    if "et_s" in frame.columns and "et_min" not in frame.columns:
        frame["et_min"] = frame["et_s"] / 60
    if "et_s" in frame.columns and "et_h" not in frame.columns:
        frame["et_h"] = frame["et_s"] / 3600
    return frame


def _prepare_power_frame(df_or_path):
    """Elapsed minutes, integer channel, and CLM power in dBm."""
    frame = _load(df_or_path)
    if "et_min" not in frame.columns and "et_s" in frame.columns:
        frame["et_min"] = frame["et_s"] / 60.0
    if "ch" in frame.columns:
        frame["ch"] = pd.to_numeric(frame["ch"], errors="coerce").astype(int)
    if "clm_power_uW" in frame.columns:
        power_uw = pd.to_numeric(frame["clm_power_uW"], errors="coerce")
        frame["clm_reported_power_dbm"] = np.where(
            power_uw > 0, 10.0 * np.log10(power_uw / 1000.0), np.nan
        )
    return frame


def _plot_metric_grid(frame, metrics, serial_number, tile_index, figsize, heading=""):
    """One row per metric, bank A and bank B side by side, versus elapsed time.

    A metric is (column, ylabel, per_channel). per_channel draws one line per
    channel. A tile-level value such as T_PIC is drawn as one line.
    """
    n_rows = len(metrics)
    fig, axes = plt.subplots(n_rows, 2, figsize=figsize, sharex=True)
    axes = np.atleast_2d(axes)
    for row, (column, ylabel, per_channel) in enumerate(metrics):
        for col, bank in enumerate(("A", "B")):
            ax = axes[row, col]
            subset = frame[frame["bank"].astype(str) == bank] if "bank" in frame.columns else frame
            if column in subset.columns and not subset.empty:
                if per_channel:
                    sns.lineplot(
                        data=subset,
                        x="et_min",
                        y=column,
                        hue="ch",
                        palette="bright",
                        ax=ax,
                        legend=(col == 1),
                    )
                else:
                    series = subset.groupby("et_min", as_index=False)[column].mean()
                    sns.lineplot(
                        data=series, x="et_min", y=column, ax=ax, legend=False
                    )
            ax.set_title(_per_plot_heading("", serial_number, tile_index, bank))
            ax.set_ylabel(ylabel)
            ax.set_xlabel("elapsed time (min)")
            ax.grid(True)
            if col == 1 and ax.get_legend() is not None:
                ax.legend(title="ch", loc="center left", bbox_to_anchor=(1.02, 0.5))
    if str(heading).strip():
        fig.suptitle(str(heading).strip())
        fig.tight_layout(rect=(0, 0, 1, 0.97))
    else:
        fig.tight_layout()
    return fig


def _save_or_show(fig, output_dir, filename, show):
    path = None
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / filename
        fig.savefig(path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return path


def _bias_metrics(frame):
    """LD, VOA, and PIC temperature rows when those columns are present."""
    metrics = []
    if "LD_mA" in frame.columns:
        metrics.append(("LD_mA", "LD bias (mA)", True))
    if "VOA_mA" in frame.columns:
        metrics.append(("VOA_mA", "VOA bias (mA)", True))
    if "T_PIC" in frame.columns:
        metrics.append(("T_PIC", "T_PIC (C)", False))
    return metrics


def plot_power_and_frequency(
    df_or_path,
    output_dir=None,
    file_stem="power_and_frequency",
    show=True,
    serial_number=None,
    tile_index=None,
):
    """OSA power, CLM power, wavelength error, LD, VOA, and T_PIC versus time.

    CLM power is ``clm_power_uW`` converted to dBm. LD and VOA are one line
    per channel. T_PIC is one line for the tile. One column per bank.
    When ``output_dir`` is set, saves ``{file_stem}_power_freq.png``.
    """
    frame = _prepare_power_frame(df_or_path)
    metrics = [
        ("power_from_osa_dBm", "channel power (dBm)", True),
        ("clm_reported_power_dbm", "clm reported power (dbm)", True),
        ("wavelength_error_nm", "wavelength error (nm)", True),
        *_bias_metrics(frame),
    ]
    fig = _plot_metric_grid(
        frame,
        metrics,
        serial_number,
        tile_index,
        figsize=(12, 3.4 * len(metrics)),
    )
    return _save_or_show(fig, output_dir, f"{file_stem}_power_freq.png", show)


def plot_power_vs_time(
    df_or_path,
    output_dir=None,
    file_stem="power_vs_time",
    show=True,
    serial_number=None,
    tile_index=None,
    heading="power vs time",
):
    """OSA power, CLM power, LD, VOA, and T_PIC versus elapsed time.

    CLM power is ``clm_power_uW`` converted to dBm. LD and VOA are one line
    per channel. T_PIC is one line for the tile. One column per bank.
    When ``output_dir`` is set, saves ``{file_stem}_power.png``.
    """
    frame = _prepare_power_frame(df_or_path)
    metrics = [
        ("power_from_osa_dBm", "channel power (dBm)", True),
        ("clm_reported_power_dbm", "clm reported power (dbm)", True),
        *_bias_metrics(frame),
    ]
    fig = _plot_metric_grid(
        frame,
        metrics,
        serial_number,
        tile_index,
        figsize=(12, 3.4 * len(metrics)),
        heading=heading,
    )
    return _save_or_show(fig, output_dir, f"{file_stem}_power.png", show)


def plot_power_stability(df_or_path, power_targets=(12, 10, 8), p_ylim=(6000, 14000)):
    """Power and VOA current versus time, one panel pair per lane.

    Rows are limited to the bank that owns the lane. Pass p_ylim=None to
    leave the power axis on autoscale.
    """
    frame = _load(df_or_path)
    for bank in (0, 1):
        for channel in range(8):
            global_channel = 8 * bank + channel
            enable_column = f"ch{global_channel}_enable"
            subset = frame[frame["bank"] == bank]
            if enable_column in subset.columns:
                subset = subset[subset[enable_column] == 1]
            if "pow_tgt" in subset.columns:
                subset = subset[subset["pow_tgt"].isin(power_targets)]
            if subset.empty:
                continue
            _line_by_condition(subset, f"p_mon{channel}", f"ch{channel}, bank{bank}", p_ylim)
            _line_by_condition(subset, f"I_VOA{channel}", f"ch{channel}, bank{bank}", None)
    return frame


def _line_by_condition(subset, y_column, title, ylim):
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.lineplot(
        data=subset,
        x="et_s",
        y=y_column,
        linestyle="-",
        marker="",
        hue="pow_tgt",
        style="Nch_enabled",
        palette="bright",
        ax=ax,
    )
    ax.set_title(title)
    ax.grid(True)
    if ylim is not None:
        ax.set_ylim(ylim)
    fig.tight_layout()
    plt.show()


def _channels_long(frame, prefix, time_col):
    """Stack bank lanes into one value column labeled ch0..ch15."""
    pieces = []
    for bank in (0, 1):
        subset = frame[frame["bank"] == bank]
        for channel in range(8):
            column = f"{prefix}{channel}"
            piece = subset[[time_col, column]].copy()
            piece = piece.rename(columns={column: "value"})
            piece["channel"] = f"ch{bank * 8 + channel}"
            pieces.append(piece)
    long = pd.concat(pieces, ignore_index=True)
    long["channel"] = pd.Categorical(
        long["channel"],
        categories=[f"ch{i}" for i in range(16)],
        ordered=True,
    )
    return long


def _plot_channel_traces(frame, prefix, ylabel, title, time_col):
    long = _channels_long(frame, prefix, time_col)
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.lineplot(
        data=long,
        x=time_col,
        y="value",
        hue="channel",
        palette="bright",
        linewidth=1,
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("elapsed time (s)")
    ax.set_ylabel(ylabel)
    ax.grid(True)
    ax.legend(ncol=2, fontsize=8, loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout()
    plt.show()


def plot_power_level_sweep(df_or_path):
    """PIC temperature, channel power, LD bias, and VOA bias versus time.

    T_PIC is taken from bank 0 because the same tile temperature is stored
    on both bank rows. Channel traces use et_run_s when the sweep stitched
    dwells onto one clock.
    """
    frame = _load(df_or_path)
    if frame.empty:
        print("No samples to plot.")
        return frame
    time_col = "et_run_s" if "et_run_s" in frame.columns else "et_s"
    pic = frame[frame["bank"] == 0]
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.lineplot(data=pic, x=time_col, y="T_PIC", ax=ax)
    ax.set_title("PIC temperature")
    ax.set_xlabel("elapsed time (s)")
    ax.set_ylabel("T_PIC (C)")
    ax.grid(True)
    fig.tight_layout()
    plt.show()

    _plot_channel_traces(frame, "p_mon", "power monitor", "channel power", time_col)
    _plot_channel_traces(frame, "I_LD", "LD bias (mA)", "LD bias", time_col)
    _plot_channel_traces(frame, "I_VOA", "VOA bias (mA)", "VOA bias", time_col)
    return frame


def plot_thermal_watch(df_or_path):
    """Temperature, resistance, and drive voltage versus elapsed seconds."""
    frame = _load(df_or_path)
    fig, (ax, ax1) = plt.subplots(2, 1, figsize=(6, 6))
    axt = ax1.twinx()
    frame.plot("et_s", "T", ax=ax)
    frame.plot("et_s", "R", ax=ax1)
    frame.plot("et_s", "V", color="r", ax=axt)
    ax.grid(True)
    ax1.grid(True)
    ax.set_ylabel("Temp (C)")
    ax1.set_ylabel("R")
    axt.set_ylabel("drive (V)", color="r")
    fig.tight_layout()
    plt.show()
    return frame


def plot_thermal_check(df_or_path, kp, ki, kd, interval_s):
    """Temperature setpoint and drive voltage versus elapsed minutes."""
    frame = _load(df_or_path)
    fig, ax = plt.subplots(figsize=(8, 5))
    axt = ax.twinx()
    frame.plot("et_min", ["temp_tgt", "temp"], ax=ax)
    frame.plot("et_min", "volt", color="r", ax=axt)
    ax.set_ylabel("temp(C)")
    axt.set_ylabel("drive (V)", color="r")
    ax.legend(loc="center left")
    axt.legend(loc="center right")
    ax.grid(True)
    ax.set_title(f"Kp {kp:.3f}, Ki {ki:.3f}, Kd {kd:.3f}; update period {interval_s}")
    fig.tight_layout()
    plt.show()
    return frame


def plot_per_meter_cal(df_or_path):
    """PER and angle versus time and temperature, then PER versus angle."""
    frame = _load(df_or_path)
    fig, ax = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    sns.lineplot(data=frame, x="et_min", y="temp", ax=ax[0])
    sns.lineplot(data=frame, x="et_min", y="per_dB", ax=ax[1])
    sns.lineplot(data=frame, x="et_min", y="theta", ax=ax[2])
    ax[0].set_ylabel("temp (C)")
    ax[1].set_ylabel("PER (dB)")
    ax[2].set_ylabel("polarization angle (deg)")
    ax[2].set_xlabel("elapsed time (min)")
    for axis in ax:
        axis.grid(True)
    fig.tight_layout()
    plt.show()

    fig, ax = plt.subplots(2, 1, figsize=(7, 7), sharex=True)
    sns.lineplot(data=frame, x="temp", y="per_dB", ax=ax[0])
    sns.lineplot(data=frame, x="temp", y="theta", ax=ax[1])
    ax[0].set_ylabel("PER (dB)")
    ax[1].set_ylabel("polarization angle (deg)")
    ax[1].set_xlabel("temp (C)")
    for axis in ax:
        axis.grid(True)
    fig.tight_layout()
    plt.show()

    fig, ax = plt.subplots(figsize=(6, 6))
    sns.scatterplot(data=frame, x="theta", y="per_dB", marker="o", ax=ax)
    ax.set_xlabel("polarization angle (deg)")
    ax.set_ylabel("PER (dB)")
    ax.grid(True)
    fig.tight_layout()
    plt.show()
    return frame


def _per_bank_label(bank):
    """Bank 0/1 is written as A/B. A and B are kept as given."""
    text = str(bank).strip()
    if text.upper() in ("A", "B"):
        return text.upper()
    try:
        as_int = int(float(text))
    except (TypeError, ValueError):
        return text
    return {0: "A", 1: "B"}.get(as_int, text)


def _per_plot_heading(kind, serial_number=None, tile_index=None, bank=None):
    """``PER vs time — <sn>, tile <idx>, bank <A|B>``."""
    bits = []
    serial = "" if serial_number is None else str(serial_number).strip()
    if serial:
        bits.append(serial)
    if tile_index is not None and str(tile_index).strip() != "":
        try:
            as_int = int(tile_index)
            if float(tile_index) == as_int:
                tile_index = as_int
        except (TypeError, ValueError):
            pass
        bits.append(f"tile {tile_index}")
    if bank is not None and str(bank).strip() != "":
        bits.append(f"bank {_per_bank_label(bank)}")
    if not bits:
        return kind
    identity = ", ".join(bits)
    if not str(kind).strip():
        return identity
    return f"{kind} — {identity}"


def plot_per_vs_temperature(
    df_or_path,
    output_dir=None,
    file_stem="per_vs_temperature",
    show=True,
    serial_number=None,
    tile_index=None,
    bank=None,
):
    """PER / theta / temp versus time, then PER versus theta.

    When ``output_dir`` is set, saves:
      ``{file_stem}_time.png`` and ``{file_stem}_theta.png``.
    Titles include serial number, tile index, and bank when those are given.
    Tile index and bank fall back to a single value in those frame columns.
    """
    frame = _load(df_or_path)
    if "et_min" not in frame.columns and "et_0" in frame.columns:
        frame["et_min"] = frame["et_0"] / 60.0
    if "et_min" not in frame.columns and "et_s" in frame.columns:
        frame["et_min"] = frame["et_s"] / 60.0
    if tile_index is None and "tile_index" in frame.columns:
        tile_values = frame["tile_index"].dropna().unique()
        if len(tile_values) == 1:
            tile_index = tile_values[0]
    if bank is None and "bank" in frame.columns:
        bank_values = frame["bank"].dropna().unique()
        if len(bank_values) == 1:
            bank = bank_values[0]
    time_title = _per_plot_heading("PER vs time", serial_number, tile_index, bank)
    theta_title = _per_plot_heading("PER vs theta", serial_number, tile_index, bank)
    # Arroyo-only runs may leave temp_thorlab empty; prefer temp then temp_arroyo.
    temp_col = "temp"
    if temp_col not in frame.columns or frame[temp_col].isna().all():
        if "temp_arroyo" in frame.columns:
            temp_col = "temp_arroyo"

    out = Path(output_dir) if output_dir is not None else None
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(3, 1, figsize=(8, 10))
    sns.lineplot(
        data=frame, x="et_min", y=temp_col, hue="ch", palette="bright", ax=ax[0]
    )
    sns.lineplot(
        data=frame,
        x="et_min",
        y="per_dB",
        linestyle="-",
        marker="",
        hue="ch",
        palette="bright",
        ax=ax[1],
    )
    sns.lineplot(
        data=frame, x="et_min", y="theta", hue="ch", palette="bright", ax=ax[2]
    )
    ax[0].set_ylabel("temp (C)")
    ax[1].set_ylabel("PER (dB)")
    ax[2].set_ylabel("theta (deg)")
    ax[2].set_xlabel("et_min")
    for axis in ax:
        axis.grid(True)
    fig.suptitle(time_title)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    paths = []
    if out is not None:
        path_time = out / f"{file_stem}_time.png"
        fig.savefig(path_time, dpi=150)
        paths.append(path_time)
    if show:
        plt.show()
    else:
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 6))
    sns.scatterplot(
        data=frame, x="theta", y="per_dB", hue="ch", marker="o", palette="bright", ax=ax
    )
    ax.set_xlabel("polarization angle (deg)")
    ax.set_ylabel("PER (dB)")
    ax.set_title(theta_title)
    ax.grid(True)
    fig.tight_layout()
    if out is not None:
        path_theta = out / f"{file_stem}_theta.png"
        fig.savefig(path_theta, dpi=150)
        paths.append(path_theta)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return frame if not paths else (frame, paths)


def plot_per_vs_frequency(df_or_path):
    frame = _load(df_or_path)
    fig, ax = plt.subplots(2, 1, figsize=(8, 8))
    sns.lineplot(data=frame, x="freq_offset", y="per_dB", linestyle="-", marker="o", hue="ch", palette="bright", ax=ax[0])
    sns.lineplot(data=frame, x="freq_offset", y="theta", hue="ch", marker="o", palette="bright", ax=ax[1])
    ax[1].set_xlabel("freq_offset")
    for axis in ax:
        axis.grid(True)
    fig.tight_layout()
    plt.show()

    fig, ax = plt.subplots(figsize=(6, 6))
    sns.scatterplot(data=frame, x="theta", y="per_dB", hue="ch", marker="o", palette="bright", ax=ax)
    ax.grid(True)
    ax.set_title("(PER, theta) over [-20, 20] GHz frequency change")
    fig.tight_layout()
    plt.show()

    fig, ax = plt.subplots(figsize=(6, 6))
    sns.scatterplot(data=frame, x="et_h", y="temp", hue="ch", linestyle=":", marker="o", palette="bright", ax=ax)
    ax.grid(True)
    ax.set_ylim((31, 33))
    ax.set_title("PM fiber temp over time")
    fig.tight_layout()
    plt.show()
    return frame


def plot_overnight(df_or_path):
    """Time series, PER versus temperature, and the temperature-step histogram."""
    frame = _load(df_or_path)

    fig, ax = plt.subplots(3, 1, figsize=(8, 9))
    frame.plot("et_h", "temp", marker="", ax=ax[0])
    frame.plot("et_h", "per_dB", marker="", ax=ax[1])
    frame.plot("et_h", "theta", marker="", ax=ax[2])
    ax[2].set_xlabel("et_h")
    for axis in ax:
        axis.grid(True)
    fig.tight_layout()
    plt.show()

    fig, ax = plt.subplots(2, 1, figsize=(6, 6))
    frame.plot("temp", "per_dB", linestyle=":", marker=".", ax=ax[0])
    frame.plot("temp", "theta", linestyle=":", marker=".", ax=ax[1])
    for axis in ax:
        axis.grid(True)
    fig.tight_layout()
    plt.show()

    settled = frame[frame["et_s"] > 3600]
    fig, ax = plt.subplots(figsize=(5, 5))
    settled.plot("theta", "per_dB", linestyle="", marker=".", ax=ax)
    ax.grid(True)
    ax.set_xlim((-6, 8))
    ax.set_ylim((15, 35))
    fig.tight_layout()
    plt.show()

    steps = settled.copy()
    steps["dt"] = steps["et_s"].diff()
    steps["dT_C"] = steps["temp"].diff()
    steps = steps[steps["dt"] > 0]
    plt.hist(steps["dT_C"], bins=30, color="skyblue", edgecolor="black")
    plt.title("fiber n temp sensor in foam, temp change over 15s")
    plt.xlabel("dT (C)")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.xlim((-0.2, 0.2))
    plt.show()
    return frame
