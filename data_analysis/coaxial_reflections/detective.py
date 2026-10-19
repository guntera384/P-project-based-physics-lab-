"""Run detective.cir in ngspice and plot v(2).

Usage: python plot_detective.py [netlist.cir]
"""

import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

NETLIST = Path(sys.argv[1] if len(sys.argv) > 1 else "detective.cir")

with tempfile.TemporaryDirectory() as tmp:
    out = Path(tmp) / "out.txt"
    circuit = Path(tmp) / "run.cir"
    # The netlist has no .end, so the control block and .end are appended here.
    # (If you add a .end to the netlist, remove it again or the results are not written.)
    circuit.write_text(
        f"{NETLIST.read_text()}\n.control\nrun\nwrdata {out} v(2)\n.endc\n.end\n"
    )
    proc = subprocess.run(
        ["ngspice", "-b", str(circuit)], capture_output=True, text=True
    )
    if not out.exists():  # ngspice often exits with 0 even when the run failed
        sys.exit(
            f"ngspice produced no output:\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"
        )
    t, v = np.loadtxt(out, unpack=True)

fig, ax = plt.subplots()
ax.plot(t * 1e6 - 1, v)
ax.set(xlabel="time [µs]", ylabel="V(2) [V]")
ax.grid(True, alpha=0.3)
ax.set_title("Predicted signal (Circuit detective 1)")
ax.set_ylabel("Voltage $V(0,t)$ [V]")
ax.set_xlabel("Time $t$ [$\\mu$s]")
plt.savefig("final_plots/detective_predicted1.pdf")
