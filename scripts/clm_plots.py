"""Figures for CLM bench runs. These functions only read a dataframe or a CSV."""

from pathlib import Path

import matplotlib.pyplot as plt
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


def plot_per_vs_temperature(
    df_or_path,
    output_dir=None,
    file_stem="per_vs_temperature",
    show=True,
):
    """PER / theta / temp versus time, then PER versus theta.

    When ``output_dir`` is set, saves:
      ``{file_stem}_time.png`` and ``{file_stem}_theta.png``.
    """
    frame = _load(df_or_path)
    if "et_min" not in frame.columns and "et_0" in frame.columns:
        frame["et_min"] = frame["et_0"] / 60.0
    if "et_min" not in frame.columns and "et_s" in frame.columns:
        frame["et_min"] = frame["et_s"] / 60.0
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
    fig.tight_layout()
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
