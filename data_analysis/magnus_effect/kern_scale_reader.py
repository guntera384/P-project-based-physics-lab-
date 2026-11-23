"""
Read the weight from a KERN PCB scale over RS232 at a fixed interval.

Requirements:
    pip install pyserial

Scale setup (older PCB, manual PCB-BA-e-1315): hold PRINT until the unit
appears, then use MODE to select, SET to confirm:
    - "Pr"   -> "rE Cr"  (data output via remote control commands)
                or "AU PC" (continuous output; then set COMMAND = None)
    - "bAUd" -> same value as BAUDRATE below
    - Data format is fixed: 8 data bits, no parity, 1 stop bit
    - Cable: straight 1:1 (scale pin 2 = TXD, pin 3 = RXD, pin 5 = GND)

Note: the scale itself only produces about 3 new values per second
(~300 ms per measurement). If INTERVAL_S is shorter than that, the most
recent value is printed again and marked as "(repeat)".
"""

import re
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import serial
import serial.tools.list_ports

# ---------------------------------------------------------------------------
# Settings - adjust these as needed
# ---------------------------------------------------------------------------
PORT = "COM6"            # Serial port of the scale (Linux e.g. "/dev/ttyUSB0")
BAUDRATE = 9600          # Must match the baud rate set on the scale
INTERVAL_S = 0.100       # Time between readings in seconds (0.100 = 100 ms)
COMMAND = b"w"           # Older PCB (mode "rE Cr"): b"w" = current weight, b"s" = stable only
                         # Newer PCB (KCP protocol): b"SI\r\n" = current, b"S\r\n" = stable only
                         # None = don't send anything (scale in continuous mode "AU PC")
ONLY_NEW_VALUES = False  # True = print only when the scale delivered a new value
CSV_FILE = True          # True = log to "weights_<date>_<time>.csv" next to this script,
                         # "name.csv" = log to that file (appends), None/False = off
# ---------------------------------------------------------------------------

# Matches e.g. "     12.345 g", "-    0.50 g", "    2.4    " (unstable: no unit)
WEIGHT_PATTERN = re.compile(r"([+-]?)\s*(\d+(?:\.\d+)?)\s*([A-Za-z%]*)")


def parse_weight(line: str):
    """Return (weight, unit) from a line sent by the scale, or None.

    The scale leaves the unit empty while the weight is not yet stable.
    """
    match = WEIGHT_PATTERN.search(line)
    if not match:
        return None
    sign, number, unit = match.groups()
    value = float(number)
    if sign == "-":
        value = -value
    return value, unit


class ScaleReader(threading.Thread):
    """Reads lines from the scale in the background and keeps the latest value."""

    def __init__(self, port: serial.Serial):
        super().__init__(daemon=True)
        self.port = port
        self.lock = threading.Lock()
        self.weight = None       # latest weight
        self.unit = ""           # last known unit
        self.stable = False      # was the latest weight stable?
        self.count = 0           # number of weights received so far
        self.running = True

    def run(self):
        buffer = b""
        while self.running:
            try:
                chunk = self.port.read(self.port.in_waiting or 1)
            except serial.SerialException as exc:
                print(f"Serial error: {exc}")
                break
            if not chunk:
                continue
            buffer += chunk
            while b"\n" in buffer:
                raw, buffer = buffer.split(b"\n", 1)
                self._handle_line(raw)

    def _handle_line(self, raw: bytes):
        text = raw.decode("ascii", errors="ignore").strip()
        if not text:
            return
        result = parse_weight(text)
        if result is None:
            print(f"Unexpected data from scale: {raw!r}")
            return
        weight, unit = result
        with self.lock:
            self.weight = weight
            self.stable = bool(unit)
            if unit:
                self.unit = unit
            self.count += 1

    def snapshot(self):
        with self.lock:
            return self.weight, self.unit, self.stable, self.count


def open_scale() -> serial.Serial:
    try:
        return serial.Serial(
            port=PORT,
            baudrate=BAUDRATE,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=0.2,
        )
    except serial.SerialException as exc:
        print(f"Could not open {PORT}: {exc}")
        print("Available ports:")
        for p in serial.tools.list_ports.comports():
            print(f"  {p.device}  -  {p.description}")
        sys.exit(1)


def main():
    scale = open_scale()
    scale.reset_input_buffer()
    reader = ScaleReader(scale)
    reader.start()

    log = None
    if CSV_FILE:
        if CSV_FILE is True:
            csv_path = Path(__file__).with_name(
                f"weights_{datetime.now():%Y-%m-%d_%H-%M-%S}.csv")
        else:
            csv_path = Path(CSV_FILE)
        log = open(csv_path, "a", encoding="utf-8")
        print(f"Logging to {csv_path.resolve()}")
    if log and log.tell() == 0:
        log.write("timestamp;weight;unit;stable;new\n")

    print(f"Reading from {PORT} every {INTERVAL_S * 1000:.0f} ms - Ctrl+C to stop")
    next_time = time.monotonic()
    last_count = 0

    try:
        while True:
            # Ask for a new value. Requests the scale can't serve yet are
            # simply ignored by it, so asking every interval is harmless.
            if COMMAND:
                scale.write(COMMAND)

            weight, unit, stable, count = reader.snapshot()
            is_new = count != last_count
            last_count = count
            timestamp = datetime.now().isoformat(timespec="milliseconds")

            if weight is None:
                print(f"{timestamp}  waiting for first value from scale...")
            elif is_new or not ONLY_NEW_VALUES:
                flags = ("" if stable else " (unstable)") + ("" if is_new else " (repeat)")
                print(f"{timestamp}  {weight:>10.3f} {unit}{flags}")
                if log:
                    log.write(f"{timestamp};{weight};{unit};{int(stable)};{int(is_new)}\n")
                    log.flush()

            # Fixed-rate scheduling: no drift from processing time
            next_time += INTERVAL_S
            delay = next_time - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                next_time = time.monotonic()    # fell behind, resync
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        reader.running = False
        reader.join(timeout=1)
        scale.close()
        if log:
            log.close()


if __name__ == "__main__":
    main()
