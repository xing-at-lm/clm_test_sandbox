# -*- coding: utf-8 -*-
"""OZ Optics ER-1000 polarization extinction-ratio meter.

Remote protocol from the OP-ER-1000 Operating Instructions (11 Aug 2020).
The meter uses a Silicon Labs CP210x USB-UART bridge, so the PC sees a COM port.

Factory serial settings: 115200 baud, 8 data bits, no parity, 1 stop bit,
no flow control. Every command is ASCII, is not case sensitive, and must end
with both CR and LF. A successful command finishes with a line containing
``Done``. Responses themselves end with ``\\r\\n``.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass

import serial
import serial.tools.list_ports

BAUD_RATES = (9600, 19200, 38400, 57600, 115200, 230400)
DEFAULT_BAUD = 115200

_MODES = {
    "er": "M0",
    "normal": "M0",
    "m0": "M0",
    "logging": "M1",
    "log": "M1",
    "m1": "M1",
    "power": "M2",
    "m2": "M2",
}

# The meter sends the degree sign as byte 0xF8 (latin-1 "ø"). Accept the
# usual degree signs as well.
_ER_LINE = re.compile(
    r"([<>]?[+-]?\d+(?:\.\d+)?)dB\s+"
    r"([<>]?[+-]?\d+(?:\.\d+)?)[°ºø]?\s+"
    r"([<>]?[+-]?\d+(?:\.\d+)?)dBm",
    re.IGNORECASE,
)
_POWER_QUERY = re.compile(
    r"Power\s*=\s*([<>]?[+-]?\d+(?:\.\d+)?)\s*dBm",
    re.IGNORECASE,
)
_PLAIN_POWER = re.compile(r"^[+-]?\d+(?:\.\d+)?$")
_ERROR_LINE = re.compile(r"Error\s*\(?(\d+)\)?", re.IGNORECASE)
_LG_LINE = re.compile(
    r"LG\s*([+-]?\d+(?:\.\d+)?)\s*,\s*([+-]?\d+(?:\.\d+)?)\s*,\s*([+-]?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)


class OzER1000Error(RuntimeError):
    """Command rejected by the meter, or the meter reported a fault."""

    def __init__(self, message: str, code: int | None = None):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class ErReading:
    """One normal-mode sample: extinction ratio, keyway angle, and peak power."""

    per_db: float
    angle_deg: float
    power_dbm: float

    def as_list(self) -> list[float]:
        return [self.per_db, self.angle_deg, self.power_dbm]


@dataclass(frozen=True)
class LogResult:
    """Worst-case result returned when a logging run is stopped.

    ``per_db``, ``angle_deg``, and ``power_dbm`` come from the ``LOG:`` line.
    ``delta_deg`` is the span between the minimum and maximum angle (θmax − θmin).
    The ``lg_*`` fields are the compact ``LG`` record the meter appends.
    """

    delta_deg: float | None
    per_db: float | None
    angle_deg: float | None
    power_dbm: float | None
    lg_per_db: float | None
    lg_angle_deg: float | None
    lg_delta_deg: float | None
    raw: tuple[str, ...]


def list_available_ports() -> list[str]:
    """Print COM ports and return their device names."""
    ports = list(serial.tools.list_ports.comports())
    print("Available COM ports:")
    if not ports:
        print("  No COM ports found.")
        return []
    names = []
    for port_info in ports:
        print(f"  {port_info.device} - {port_info.description} [{port_info.manufacturer}]")
        names.append(port_info.device)
    return names


def _number(token: str) -> float:
    """Parse a meter number, dropping a leading range marker (``>`` or ``<``)."""
    token = token.strip()
    if token[:1] in "<>":
        token = token[1:]
    return float(token)


def parse_er_line(line: str) -> ErReading | None:
    """Parse ``+8.46dB -24.0ø -31.0944dBm``. Return None if the line is not a sample."""
    match = _ER_LINE.search(line)
    if match is None:
        return None
    return ErReading(
        per_db=_number(match.group(1)),
        angle_deg=_number(match.group(2)),
        power_dbm=_number(match.group(3)),
    )


def parse_log(lines: list[str]) -> LogResult:
    """Parse the block returned by ``G0`` at the end of a logging run."""
    delta = per = angle = power = None
    lg_per = lg_angle = lg_delta = None
    for line in lines:
        if line.lower().startswith("delta:"):
            delta = _number(line.split(":", 1)[1])
            continue
        reading = parse_er_line(line)
        if reading is not None:
            per, angle, power = reading.per_db, reading.angle_deg, reading.power_dbm
            continue
        lg = _LG_LINE.search(line)
        if lg is not None:
            lg_per = float(lg.group(1))
            lg_angle = float(lg.group(2))
            lg_delta = float(lg.group(3))
    return LogResult(delta, per, angle, power, lg_per, lg_angle, lg_delta, tuple(lines))


class ER1000:
    """Serial driver for the OZ Optics ER-1000.

    Opening the port turns echo off and capture on, which is the setup the
    meter needs before measurement lines are sent to the PC. The polarizer
    is not started until :meth:`start` or one of the measurement methods.

    Use as a context manager, or call :meth:`close` when finished. Closing
    stops a measurement that is still running.
    """

    def __init__(self, port: str, baudrate: int = DEFAULT_BAUD, timeout: float = 5.0):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self._rx = bytearray()
        self._ser = serial.Serial(
            port=port,
            baudrate=baudrate,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=timeout,
            write_timeout=timeout,
        )
        # Drop anything left from a previous session, then stop a run that
        # might still be streaming so the following commands get a clean reply.
        self._discard_input()
        try:
            self.stop(flush=True)
        except OzER1000Error:
            self._discard_input()
        self.set_echo(False)
        self.set_capture(True)
        try:
            print(f"Connected to {port}: {self.version()}")
        except OzER1000Error as exc:
            print(f"Opened {port}, but the version query failed: {exc}")

    def close(self) -> None:
        if self._ser is None or not self._ser.is_open:
            return
        try:
            self._ser.timeout = 0.5
            self.stop(timeout=0.8)
        except Exception:
            pass
        self._ser.close()

    def __enter__(self) -> "ER1000":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @property
    def is_open(self) -> bool:
        return self._ser is not None and self._ser.is_open

    def _discard_input(self) -> None:
        self._rx.clear()
        self._ser.reset_input_buffer()

    def _send(self, command: str, *, flush: bool = True) -> None:
        if flush:
            self._discard_input()
        self._ser.write(f"{command.strip()}\r\n".encode("ascii"))

    def _next_line(self) -> str | None:
        """Return the next complete line, or None when the read times out.

        A trailing fragment is kept in ``self._rx`` so the next read can finish it.
        """
        while b"\n" not in self._rx:
            waiting = self._ser.in_waiting
            chunk = self._ser.read(waiting or 1)
            if not chunk:
                return None
            self._rx.extend(chunk)
        raw, _, rest = self._rx.partition(b"\n")
        self._rx = bytearray(rest)
        return raw.decode("latin-1").strip("\r").strip()

    def _read_response(self, timeout: float | None = None, echo_of: str | None = None) -> list[str]:
        """Read lines until ``Done``. Raise if the meter returns an error."""
        deadline = time.monotonic() + (self.timeout if timeout is None else timeout)
        lines: list[str] = []
        echoed = echo_of.strip().lower() if echo_of else None
        while time.monotonic() < deadline:
            text = self._next_line()
            if text is None:
                continue
            if not text:
                continue
            if echoed is not None and text.lower() == echoed:
                continue
            if text.endswith("Done") and text != "Done":
                head = text[: -len("Done")].rstrip()
                if head:
                    lines.append(head)
                return lines
            if text == "Done":
                return lines
            error = _ERROR_LINE.search(text)
            if error is not None:
                raise OzER1000Error(text, code=int(error.group(1)))
            if "FATAL ERROR" in text.upper():
                detail = [text]
                for _ in range(6):
                    extra = self._next_line()
                    if not extra:
                        break
                    detail.append(extra)
                raise OzER1000Error("\n".join(detail))
            lines.append(text)
        partial = "; ".join(lines) if lines else "no data"
        raise OzER1000Error(f"Timed out waiting for Done ({partial})")

    def query(self, command: str, timeout: float | None = None, *, flush: bool = True) -> list[str]:
        """Send one command and return the reply lines that precede ``Done``.

        Error 01 means the meter is busy. The manual says to send the command
        again, so this retries once. A line that merely repeats the command is
        dropped, which is what the meter sends when echo is still on.
        """
        command = command.strip()
        for attempt in range(2):
            self._send(command, flush=flush)
            try:
                return self._read_response(timeout=timeout, echo_of=command)
            except OzER1000Error as exc:
                if exc.code == 1 and attempt == 0:
                    time.sleep(0.3)
                    continue
                raise
        return []

    # ------------------------------------------------------------------
    # Identity and setup
    # ------------------------------------------------------------------

    def version(self) -> str:
        """Device name and firmware/hardware version (``V?``)."""
        return " ".join(self.query("V?")).strip()

    def serial_number(self) -> str:
        """Serial number string from ``V2`` (8 to 14 characters)."""
        lines = self.query("V2")
        return lines[0].strip() if lines else ""

    def calibration_info(self) -> str:
        """Serial number and factory calibration date from ``SN?``."""
        return " ".join(self.query("SN?")).strip()

    def manufacture_date(self) -> str:
        """Date of manufacture from ``D?``, for example ``2015-7-31``."""
        lines = self.query("D?")
        return lines[0].strip() if lines else ""

    def command_list(self) -> str:
        """Text returned by ``?``, the meter's own command summary."""
        return "\n".join(self.query("?"))

    def baud_rate(self) -> int:
        """Current baud rate. The reply looks like ``115200 (bps)``."""
        lines = self.query("B?")
        if not lines:
            raise OzER1000Error("B? returned no baud rate")
        return int(lines[0].split()[0])

    def available_baud_rates(self) -> list[int]:
        """Baud rates advertised by ``BL``."""
        rates = []
        for line in self.query("BL"):
            token = line.split()[0].replace(",", "")
            if token.isdigit():
                rates.append(int(token))
        return rates

    def set_baud_rate(self, baud: int) -> None:
        """Change the meter baud rate and then the host port to match.

        The meter sends the acknowledgement number at the old baud and
        ``Done`` at the new baud. Bits, parity, and stop bits cannot be changed.
        """
        if baud not in BAUD_RATES:
            allowed = ", ".join(str(rate) for rate in BAUD_RATES)
            raise ValueError(f"Baud rate must be one of: {allowed}")
        self._send(f"B{baud}")
        first = self._next_line()
        if not first:
            raise OzER1000Error("No reply while changing baud rate; host baud was left unchanged")
        self._ser.baudrate = baud
        self.baudrate = baud
        if first != "Done":
            self._read_response()

    def set_beeper(self, enabled: bool) -> None:
        """Enable (``BP1``) or disable (``BP0``) the key-click buzzer."""
        self.query("BP1" if enabled else "BP0")

    def capture_on(self) -> bool:
        """True when measurement lines are being copied to the serial port."""
        lines = self.query("C?")
        text = " ".join(lines).lower()
        if "on" in text:
            return True
        if "off" in text:
            return False
        raise OzER1000Error(f"Unrecognized capture status: {text}")

    def set_capture(self, enabled: bool) -> None:
        """``C1`` copies readings to the PC. ``C0`` suppresses them."""
        self.query("C1" if enabled else "C0")

    def echo_on(self) -> bool:
        lines = self.query("E?")
        text = " ".join(lines).lower()
        if "on" in text:
            return True
        if "off" in text:
            return False
        raise OzER1000Error(f"Unrecognized echo status: {text}")

    def set_echo(self, enabled: bool) -> None:
        """``E0`` stops the meter from echoing each command it receives."""
        self.query("E0" if not enabled else "E1")

    # ------------------------------------------------------------------
    # Operating mode and run control
    # ------------------------------------------------------------------

    def mode(self) -> str:
        """Current mode text, for example ``Normal Mode`` or ``Logging Mode``."""
        lines = self.query("M?")
        return lines[0].strip() if lines else ""

    def set_mode(self, mode: str) -> str:
        """Set ``er`` (M0), ``logging`` (M1), or ``power`` (M2).

        Logging mode only collects if :meth:`start` was sent first, while the
        meter is still in normal mode. Power mode requires the meter to be
        stopped and in normal mode first. :meth:`read_er`, :meth:`log_er`,
        and :meth:`read_power` already follow those sequences.
        """
        key = mode.strip().lower()
        if key not in _MODES:
            raise ValueError("mode must be 'er', 'logging', or 'power'")
        lines = self.query(_MODES[key])
        return lines[0].strip() if lines else ""

    def is_measuring(self) -> bool:
        """False when ``G?`` reports the meter is idle."""
        lines = self.query("G?")
        text = " ".join(lines).strip().lower()
        return text not in {"", "idle"}

    def start(self) -> list[str]:
        """Start the polarizer (``G1``). The first reading can take a few seconds.

        Any lines that arrive before ``Done`` are returned. In normal mode the
        manual shows ``Done`` first and the samples after it.
        """
        return self.query("G1", timeout=max(self.timeout, 20.0))

    def stop(self, *, flush: bool = False, timeout: float | None = None) -> list[str]:
        """Stop the polarizer (``G0``). A logging run returns its result here.

        The input buffer is left intact so samples already received are not
        thrown away. Pass ``flush=True`` to drop them before stopping.
        """
        if timeout is None:
            timeout = max(self.timeout, 10.0)
        return self.query("G0", timeout=timeout, flush=flush)

    def power_dbm(self) -> float:
        """Latest relative power from ``P?``.

        The meter must already be running (``G1``) in normal or power mode,
        and must have been running for at least one second. In ER mode the
        reported floor is about -47 dBm; in power mode it is about -50 dBm.
        The value is the power along the polarizer axis, not the total power
        in the fiber.
        """
        lines = self.query("P?")
        for line in lines:
            match = _POWER_QUERY.search(line)
            if match is not None:
                return _number(match.group(1))
        raise OzER1000Error(f"P? did not return a power reading: {lines}")

    # ------------------------------------------------------------------
    # Measurements
    # ------------------------------------------------------------------

    def _collect_lines(self, duration_s: float) -> list[str]:
        """Read complete lines for ``duration_s`` seconds without sending a command."""
        deadline = time.monotonic() + duration_s
        previous = self._ser.timeout
        self._ser.timeout = min(0.2, max(duration_s, 0.05))
        lines: list[str] = []
        try:
            while time.monotonic() < deadline:
                text = self._next_line()
                if text and text != "Done":
                    lines.append(text)
        finally:
            self._ser.timeout = previous
        return lines

    def iter_er(self, duration_s: float):
        """Yield ER samples as they arrive for ``duration_s`` seconds.

        The meter must already be running in normal ER mode with capture on.
        This does not send ``G1`` or ``G0``.
        """
        if duration_s <= 0:
            raise ValueError("duration_s must be positive")
        deadline = time.monotonic() + duration_s
        previous = self._ser.timeout
        self._ser.timeout = min(0.2, max(duration_s, 0.05))
        try:
            while time.monotonic() < deadline:
                text = self._next_line()
                if not text or text == "Done":
                    continue
                reading = parse_er_line(text)
                if reading is not None:
                    yield reading
        finally:
            self._ser.timeout = previous

    def read_er(self, duration_s: float = 5.0) -> list[ErReading]:
        """Measure extinction ratio for ``duration_s`` seconds and stop.

        The meter is placed in normal ER mode with capture on, started, and
        stopped when the window ends. Each sample is extinction ratio (dB),
        polarization angle relative to the connector keyway (degrees), and
        the maximum power on that axis (dBm).

        The extinction ratio is updated every quarter turn of the internal
        polarizer. Angle and power are updated only every half turn, so those
        two values repeat on every other sample. The first sample can lag the
        start by several seconds; use a window of at least a few seconds.
        """
        if duration_s <= 0:
            raise ValueError("duration_s must be positive")
        self.stop(flush=True)
        self.set_mode("er")
        self.set_capture(True)
        raw_lines = self.start()
        raw_lines.extend(self._collect_lines(duration_s))
        raw_lines.extend(self.stop())
        readings = []
        for line in raw_lines:
            reading = parse_er_line(line)
            if reading is not None:
                readings.append(reading)
        return readings

    def measure_per(self, wait_time_s: float = 5.0) -> list[list[float]]:
        """Same run as :meth:`read_er`, returned as ``[per_db, angle_deg, power_dbm]`` rows."""
        return [reading.as_list() for reading in self.read_er(wait_time_s)]

    def log_er(self, duration_s: float = 25.0) -> LogResult:
        """Run data-logging mode and return the worst-case extinction ratio.

        The manual's sequence is: stop, normal mode, start, then logging mode.
        Nothing is streamed while logging. Stopping the run (``G0``) returns
        the angle span, the worst-case ER, the matching angle, and the last
        power. A run of at least 25 seconds is the manual's typical minimum.
        """
        if duration_s <= 0:
            raise ValueError("duration_s must be positive")
        self.stop(flush=True)
        self.set_mode("er")
        self.start()
        self.set_mode("logging")
        self.set_capture(True)
        time.sleep(duration_s)
        return parse_log(self.stop())

    def read_power(self, duration_s: float = 2.0) -> list[float]:
        """Stream power-mode readings (dBm) for ``duration_s`` seconds and stop.

        Power mode holds the polarizer at the angle of maximum transmission
        and reports only that axis. The meter must be stopped and in normal
        mode before ``M2``; this method does those steps. Capture has to be
        on or the meter will not transmit the values.
        """
        if duration_s <= 0:
            raise ValueError("duration_s must be positive")
        self.stop(flush=True)
        self.set_mode("er")
        self.set_mode("power")
        self.set_capture(True)
        lines = self.start()
        lines.extend(self._collect_lines(duration_s))
        # C0 ends the stream. Do not flush, or the last readings are discarded.
        lines.extend(self.query("C0", flush=False))
        self.stop(flush=True)
        values = []
        for line in lines:
            token = line.strip()
            if _PLAIN_POWER.match(token):
                values.append(float(token))
                continue
            match = _POWER_QUERY.search(token)
            if match is not None:
                values.append(_number(match.group(1)))
        return values


# Name used by the older notebook-style driver in this folder.


if __name__ == "__main__":
    list_available_ports()
    OzER1000 = ER1000('COM5')

