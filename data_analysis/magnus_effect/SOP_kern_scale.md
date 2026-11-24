# SOP: Logging weight data from the KERN PCB scale

Use this procedure to record the weight on the KERN PCB scale on a PC, using the script `kern_scale_reader.py`.

## 1. Equipment

- KERN PCB scale with a 9-pin RS232 port
- RS232 cable, **straight 1:1** (do **not** use a null-modem cable or adapter)
- USB-to-RS232 adapter
- PC with Python 3 and the script `kern_scale_reader.py`

## 2. One-time setup

### PC

Install the serial library:

```
pip install pyserial
```

### Scale

Check these two settings if the scale doesn't answer, or if someone else may have changed them. Use the buttons on the scale:
These instructions are likely for the wrong model. 

1. In weighing mode, **hold PRINT** until the unit (e.g. `g`) appears. (never appeared in our setup)
2. Press **MODE** repeatedly until **`Pr`** appears, then press **SET**.
3. Press **MODE** until **`rE Cr`** appears, then press **SET**. The scale returns to weighing mode.
4. Hold **PRINT** again, press **MODE** until **`bAUd`** appears, then press **SET**.
5. Press **MODE** until **`9600`** appears, then press **SET**.

## 3. Connecting

1. **Switch the scale off** before you plug in or unplug the cable.
2. Connect: scale → RS232 cable → USB adapter → PC.
3. Switch the scale on and wait until it shows `0.0 g`.
4. Find the COM port: open **Device Manager → Ports (COM & LPT)** and look for "USB Serial Port (COMx)". The port number can be different on every PC.

## 4. Configuring the script

Open `kern_scale_reader.py` and edit the settings at the top:

| Setting           | Meaning                                                              | Typical value |
|-------------------|----------------------------------------------------------------------|---------------|
| `PORT`            | COM port from step 3.4                                               | `"COM6"`      |
| `BAUDRATE`        | Must match `bAUd` on the scale                                       | `9600`        |
| `INTERVAL_S`      | Time between output lines, in seconds                                | `0.100`       |
| `COMMAND`         | `b"w"` = every weight, `b"s"` = only stable weights                  | `b"w"`        |
| `ONLY_NEW_VALUES` | `True` = print only fresh readings, with no repeats                  | `False`       |
| `CSV_FILE`        | `True` = new file `weights_<date>_<time>.csv` per run, `None` = no file | `True`     |

## 5. Running a measurement

1. Remove everything from the scale and press **TARE** on the scale.
2. Start the script in a terminal in the project folder:

   ```
   python kern_scale_reader.py
   ```

3. Check that weights appear (first `waiting for first value from scale...`, then numbers).
4. Run your experiment.
5. Stop the script with **Ctrl+C**.
6. The CSV file is saved next to the script. The script prints its path when it starts.

Only one program can use the COM port at a time. Close other serial programs, and other copies of the script, before you start.

## 6. Reading the output

```
2026-11-23T13:55:54.729       0.600 g
2026-11-23T13:55:54.829       0.600 g (repeat)
2026-11-23T13:55:55.029       2.400 g (unstable)
```

- **(repeat)**: no new reading yet, so the previous value is shown again.
- **(unstable)**: the scale has not settled yet.

**Important:** the scale delivers a new reading only about every **300 ms**, so roughly 3 per second. The script prints every 100 ms, which means about 2 of every 3 lines are repeats.

The CSV file is separated by semicolons, so German Excel opens it directly in columns:

| Column      | Meaning                                     |
|-------------|---------------------------------------------|
| `timestamp` | PC time of the line                         |
| `weight`    | Weight value                                |
| `unit`      | Unit (e.g. `g`)                             |
| `stable`    | `1` = stable, `0` = not settled yet         |
| `new`       | `1` = fresh reading, `0` = repeated value   |

**For analysis, keep only the rows with `new = 1`.**

## 7. Troubleshooting

| Problem | Fix |
|---|---|
| `Could not open COMx` | Either `PORT` is wrong or another program has the port open. Use one of the ports the script lists. |
| `OSError 22` / "invalid argument" | Unplug the USB adapter, plug it back in, and restart the script. |
| Only `waiting for first value...` | Check the scale settings (section 2): `Pr` = `rE Cr` and `bAUd` = 9600. Check that the cable is straight, not null-modem, and that the scale is on. |
| `Unexpected data from scale: ...Error...` | The scale is overloaded or showing an error. Check the load on the scale. |
| Values only change every ~300 ms | This is normal, because it's the scale's own speed limit (see section 6). |
| CSV file can't be opened or deleted | The script is still running, or OneDrive is syncing the file. Stop the script and wait a moment. |
