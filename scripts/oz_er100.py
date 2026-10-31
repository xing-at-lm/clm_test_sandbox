# -*- coding: utf-8 -*-
"""OZ Optics ER-100 polarization extinction-ratio meter.

Remote protocol from the OP-ER-100 Operating Instructions, Communications
chapter. The meter uses a Silicon Labs USB-UART bridge, so the PC sees a COM
port.

Factory serial settings: 9600 baud, 8 data bits, no parity, 1 stop bit, no
flow control. Every command is ASCII, is not case sensitive, has no spaces,
and must end with both CR and LF. A successful command finishes with a line
containing ``Done``. An invalid command returns ``Error (100)``.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass

import serial
import serial.tools.list_ports

BAUD_RATES = (9600, 19200, 38400, 57600, 115200, 128000, 256000)
DEFAULT_BAUD = 9600

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

# The meter may send the degree sign as byte 0xF8 (latin-1 "ø").
_ER_LINE = re.compile(
    r"([<>]?[+-]?\d+(?:\.\d+)?)dB\s+"
    r"([<>]?[+-]?\d+(?:\.\d+)?)[°ºø]?\s+"
    r"([<>]?[+-]?\d+(?:\.\d+)?)(?:dBm|dB)?",
    re.IGNORECASE,
)
_POWER_QUERY = re.compile(
    r"Power\s*=\s*([<>]?[+-]?\d+(?:\.\d+)?)\s*dB",
    re.IGNORECASE,
)
_PLAIN_POWER = re.compile(r"^[+-]?\d+(?:\.\d+)?$")
_ERROR_LINE = re.compile(r"Error\s*\(?(\d+)\)?", re.IGNORECASE)
_DELTA_LINE = re.compile(r"Delta:\s*([<>]?[+-]?\d+(?:\.\d+)?)", re.IGNORECASE)


class OzER100Error(RuntimeError):
    """Command rejected by the meter, or the meter reported a fault."""

    def __init__(self, message: str, code: int | None = None):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class ErReading:
    """One normal-mode sample: extinction ratio, keyway angle, and relative power.

    ``power_db`` is the relative maximum on the polarizer axis, in dB. The
    ER-100 does not report this field in dBm.
    """

    per_db: float
    angle_deg: float
    power_db: float

    def as_list(self) -> list[float]:
        return [self.per_db, self.angle_deg, self.power_db]


@dataclass(frozen=True)
class LogResult:
    """Result returned when a logging run is stopped.

    ``delta_deg`` is the span between the minimum and maximum angle.
    ``per_db``, ``angle_deg``, and ``power_db`` come from the ``LOG:`` line.
    """

    delta_deg: float | None
    per_db: float | None
    angle_deg: float | None
    power_db: float | None
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
    """Parse ``27.30dB +00.3° -5.0178``. Return None if the line is not a sample."""
    match = _ER_LINE.search(line)
    if match is None:
        return None
    return ErReading(
        per_db=_number(match.group(1)),
        angle_deg=_number(match.group(2)),
        power_db=_number(match.group(3)),
    )


def parse_power_line(line: str) -> float | None:
    """Parse ``Power = -05.42dB`` or a bare power-mode sample such as ``-05.02``."""
    match = _POWER_QUERY.search(line)
    if match is not None:
        return _number(match.group(1))
    token = line.strip()
    if _PLAIN_POWER.match(token):
        return float(token)
    return None


def parse_log(lines: list[str]) -> LogResult:
    """Parse ``Delta:`` and ``LOG:`` lines from the end of a logging run."""
    delta = per = angle = power = None
    for line in lines:
        delta_match = _DELTA_LINE.search(line)
        if delta_match is not None and line.lower().lstrip().startswith("delta:"):
            delta = _number(delta_match.group(1))
            continue
        if line.lower().lstrip().startswith("log:"):
            reading = parse_er_line(line)
            if reading is not None:
                per, angle, power = reading.per_db, reading.angle_deg, reading.power_db
    return LogResult(delta, per, angle, power, tuple(lines))


class ER100:
    """Serial driver for the OZ Optics ER-100.

    Opening the port turns echo off and capture on. The polarizer is not
    started until :meth:`start` or one of the measurement methods.

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
        self._discard_input()
        try:
            self.stop(flush=True)
        except OzER100Error:
            self._discard_input()
        self.set_echo(False)
        self.set_capture(True)
        try:
            print(f"Connected to {port}: {self.version()}")
        except OzER100Error as exc:
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

    def __enter__(self) -> "ER100":
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
        """Return the next complete line, or None when the read times out."""
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
                raise OzER100Error(text, code=int(error.group(1)))
            if "FATAL ERROR" in text.upper():
                detail = [text]
                for _ in range(6):
                    extra = self._next_line()
                    if not extra:
                        break
                    detail.append(extra)
                raise OzER100Error("\n".join(detail))
            lines.append(text)
        partial = "; ".join(lines) if lines else "no data"
        raise OzER100Error(f"Timed out waiting for Done ({partial})")

    def _read_following(self, quiet_s: float = 0.4, limit_s: float = 1.5) -> list[str]:
        """Read lines that arrive after ``Done``, such as a logging Delta/LOG pair."""
        previous = self._ser.timeout
        self._ser.timeout = quiet_s
        lines: list[str] = []
        deadline = time.monotonic() + limit_s
        try:
            while time.monotonic() < deadline:
                text = self._next_line()
                if not text or text == "Done":
                    if text == "Done":
                        continue
                    break
                lines.append(text)
                if text.lower().lstrip().startswith("log:"):
                    break
        finally:
            self._ser.timeout = previous
        return lines

    def query(self, command: str, timeout: float | None = None, *, flush: bool = True) -> list[str]:
        """Send one command and return the reply lines that precede ``Done``."""
        command = command.strip()
        self._send(command, flush=flush)
        return self._read_response(timeout=timeout, echo_of=command)

    def version(self) -> str:
        """Device name and firmware/hardware version (``V?``). ``V1`` is the same query."""
        return " ".join(self.query("V?")).strip()

    def serial_number(self) -> str:
        """Serial number string from ``V2`` (8 to 14 characters)."""
        lines = self.query("V2")
        return lines[0].strip() if lines else ""

    def manufacture_date(self) -> str:
        """Date of manufacture from ``D?``, for example ``Feb 17 2002``."""
        lines = self.query("D?")
        return lines[0].strip() if lines else ""

    def command_list(self) -> str:
        """Text returned by ``?``, the meter's own command summary."""
        return "\n".join(self.query("?"))

    def baud_rate(self) -> int:
        """Current baud rate. The reply looks like ``9600 (bps)``."""
        lines = self.query("B?")
        if not lines:
            raise OzER100Error("B? returned no baud rate")
        return int(lines[0].split()[0])

    def available_baud_rates(self) -> list[int]:
        """Baud rates advertised by ``BL``."""
        rates = []
        for line in self.query("BL"):
            for token in line.replace(",", " ").split():
                if token.isdigit():
                    rates.append(int(token))
        return rates

    def set_baud_rate(self, baud: int) -> None:
        """Change the meter baud rate and then the host port to match.

        The acknowledgement number is sent at the old baud and ``Done`` at
        the new baud. Bits, parity, and stop bits cannot be changed.
        """
        if baud not in BAUD_RATES:
            allowed = ", ".join(str(rate) for rate in BAUD_RATES)
            raise ValueError(f"Baud rate must be one of: {allowed}")
        self._send(f"B{baud}")
        first = self._next_line()
        if not first:
            raise OzER100Error("No reply while changing baud rate; host baud was left unchanged")
        self._ser.baudrate = baud
        self.baudrate = baud
        if first != "Done":
            self._read_response()

    def capture_on(self) -> bool:
        """True when measurement lines are being copied to the serial port."""
        lines = self.query("C?")
        text = " ".join(lines).lower()
        if "on" in text:
            return True
        if "off" in text:
            return False
        raise OzER100Error(f"Unrecognized capture status: {text}")

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
        raise OzER100Error(f"Unrecognized echo status: {text}")

    def set_echo(self, enabled: bool) -> None:
        """``E0`` stops the meter from echoing each command it receives."""
        self.query("E0" if not enabled else "E1")

    def mode(self) -> str:
        """Current mode text, for example ``Normal Mode`` or ``Logging Mode``."""
        lines = self.query("M?")
        return lines[0].strip() if lines else ""

    def set_mode(self, mode: str) -> str:
        """Set ``er`` (M0), ``logging`` (M1), or ``power`` (M2)."""
        key = mode.strip().lower()
        if key not in _MODES:
            raise ValueError("mode must be 'er', 'logging', or 'power'")
        lines = self.query(_MODES[key])
        return lines[0].strip() if lines else ""

    def start(self) -> list[str]:
        """Start the polarizer (``G1``). The first reading can take several seconds.

        ``Done`` arrives before the samples. Lines before ``Done`` are returned.
        Samples that follow stay in the buffer for :meth:`iter_er`.
        """
        return self.query("G1", timeout=max(self.timeout, 20.0))

    def stop(self, *, flush: bool = False, timeout: float | None = None) -> list[str]:
        """Stop the polarizer (``G0``).

        A logging run may send ``Delta:`` and ``LOG:`` after ``Done``. Those
        lines are included. Pass ``flush=True`` to drop samples already buffered.
        """
        if timeout is None:
            timeout = max(self.timeout, 10.0)
        lines = self.query("G0", timeout=timeout, flush=flush)
        lines.extend(self._read_following())
        return lines

    def power_db(self) -> float:
        """Latest relative power from ``P?``, in dB.

        The meter must already be running (``G1``) in normal or power mode,
        and must have been running for at least one second. ``+00.00`` means
        a power level is not available yet.
        """
        lines = self.query("P?")
        for line in lines:
            value = parse_power_line(line)
            if value is not None and _POWER_QUERY.search(line):
                return value
        raise OzER100Error(f"P? did not return a power reading: {lines}")

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

        The first two samples are discarded. The manual says they arrive
        before the meter has finished initializing. Each kept sample is
        extinction ratio (dB), polarization angle relative to the connector
        keyway (degrees), and relative maximum power (dB).
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
        return readings[2:]

    def log_er(self, duration_s: float = 25.0) -> LogResult:
        """Run data-logging mode and return the logged extinction ratio.

        The manual's sequence is: stop, normal mode, start, then logging
        mode. Nothing is streamed while logging. Stopping with capture on
        returns the angle span and the ``LOG:`` sample.
        """
        if duration_s <= 0:
            raise ValueError("duration_s must be positive")
        self.stop(flush=True)
        self.set_mode("er")
        self.start()
        self.set_mode("logging")
        time.sleep(duration_s)
        self.set_capture(True)
        return parse_log(self.stop())

    def read_power(self, duration_s: float = 2.0) -> list[float]:
        """Stream power-mode readings (relative dB) for ``duration_s`` seconds.

        Sequence: stop, normal mode, power mode, capture on, start. ``+0.000``
        means the detector overflowed. ``-99.99`` means the light is too dim.
        """
        if duration_s <= 0:
            raise ValueError("duration_s must be positive")
        self.stop(flush=True)
        self.set_mode("er")
        self.set_mode("power")
        self.set_capture(True)
        lines = self.start()
        lines.extend(self._collect_lines(duration_s))
        lines.extend(self.stop(flush=False))
        values = []
        for line in lines:
            if _POWER_QUERY.search(line):
                continue
            value = parse_power_line(line)
            if value is not None:
                values.append(value)
        return values
