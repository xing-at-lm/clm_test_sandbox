# CLM Acceptance Test GUI — Operator Guide

This guide explains how to use the **CLM Acceptance Test** window to run acceptance tests on a DUT.

---

## 1. Purpose

The GUI walks you through one acceptance run:

1. Enter DUT and operator information  
2. Start the lab instruments (test station)  
3. Select which tests to run  
4. Run them and collect results + a pass/fail report  

The Log panel at the bottom shows progress, timing, and any errors.

---

## 2. Before you start

- DUT is installed and powered as required for the selected tests.  
- Lab instruments are on and reachable (testboard / Aardvark, OSA, Arroyo TEC, PER meters as needed).  
- Optical fibers are available for OSA / coupler / PER paths when those tests are selected.  
- Engineering has already set **`lab_config.yaml`** (instrument addresses, data folder) and **`clm_test_spec.yaml`** (firmware file, pass/fail limits). Operators do **not** need to edit these for a normal run.

---

## 3. How to launch

Do this on the lab Windows PC.

### 3.1 Start a PowerShell terminal

Pick one:

- Press **Win**, type `PowerShell`, open **Windows PowerShell**, **or**
- In **File Explorer**, go to the project folder  
  `C:\Users\MountainViewLabUser\gitrepos\clm_test_sandbox`,  
  then click the address bar, type `powershell`, and press **Enter**, **or**
- In **Cursor / VS Code**, open **Terminal → New Terminal** (PowerShell).

### 3.2 Go to the project folder and activate the venv

In PowerShell:

```powershell
cd C:\Users\MountainViewLabUser\gitrepos\clm_test_sandbox
.\.venv\Scripts\Activate.ps1
```

You should see `(.venv)` at the start of the prompt.

If activation is blocked by execution policy, run once (current user only):

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Then run `.\.venv\Scripts\Activate.ps1` again.

### 3.3 Start the GUI

With the venv active:

```powershell
python scripts\clm_acceptance_test_gui.py
```

The window title is **CLM Acceptance Test**. Leave the PowerShell window open while you use the GUI.

---

## 4. Tab 1 — DUT and Operator

| Field | Required? | Notes |
|--------|-----------|--------|
| **DUT SN** | **Yes** | Serial number of the unit under test. Used in the results folder name and the report. |
| **Operator name** | No | Printed on the report. |
| **Tile index(es)** | Yes (default provided) | Example: `[3]` or `[3, 4]`. Which CLM tile(s) to test. |
| **Comment** | No | Free text. If not empty, saved as `operator_note.txt` in the run folder when you start the run. |

Fill these **before** starting the station or running tests.

---

## 5. Tab 2 — Test station

1. Open **Test station**.  
2. Click **Start test station**.  
3. Watch the **Log** and status line until instruments report ready.  

If one instrument fails to open, the GUI may continue with the others and list what was unavailable in the Log. Fix connections and restart the station if a required instrument is missing.

**Close down** closes instruments manually. After a full **Run selected tests** sequence finishes, the GUI also closes the station automatically.

Do not start a second station action while one is still running (you will see a “Busy” message).

---

## 6. Tab 3 — Acceptance test

Check only the tests you need for this run.

| Test | What it does | What you need |
|------|----------------|---------------|
| **FW upgrade** | Flashes tile firmware from `clm_test_spec.yaml`. | Testboard / tile reachable. |
| **Tap ratio (re-)calibration using OSA** | Compares PIC and MUX (aux) power to OSA through the coupler; updates tap ratios and writes a CLM config YAML. Includes ~60 s settle after channels fire up. | OSA + coupler fibers. Laser-safety fiber prompt. |
| **power and frequency** | ~15 minute power and wavelength stability via OSA; feeds the test report. | OSA + coupler fibers. If tap cal is also selected in the same run, the tile is not re-initialized for this test. |
| **PER** | Polarization extinction vs Arroyo temperature ramp (about 25–45 °C). Bank A uses ER-1000; bank B uses ER-100. | Arroyo TEC + both PER meters; PER fiber paths. |

Typical full acceptance order: FW upgrade → tap ratio → power and frequency → PER (select all that apply).

---

## 7. Run flow

1. Confirm DUT SN and tile indexes on **DUT and Operator**.  
2. Confirm the test station is started.  
3. On **Acceptance test**, check the tests to run.  
4. Click **Run selected tests**.  

**Laser safety popup** (when tap and/or power and frequency are selected):

> please connect DUT output to coupler input (A→white, B→blue).

- Click **Yes** only after fibers are connected correctly.  
- Click **No** (or close the dialog) to cancel — no tests run.

While tests run, watch the **Log** for:

- Results directory path  
- Progress of each test  
- Elapsed time per test  
- Total elapsed time for selected tests  
- Report file paths (when a report is generated)  

Wait until you see that the run is complete and the test station is closed (unless you closed it yourself earlier).

---

## 8. Where results go

Results are written under the `data_dir` path in `lab_config.yaml`, in a folder named:

```text
_{DUT_SN}_{day}_{month}_{year}
```

Example: `_ABC123_31_10_2026`

Typical contents (depending on which tests ran):

- Raw CSV data for power/frequency and PER  
- PER plots (PNG)  
- CLM config YAML (after tap calibration)  
- `operator_note.txt` (if Comment was filled in)  
- Test report files (docx / html / pdf) when power/frequency and/or PER produced reportable metrics  

Pass/fail limits for the report come from **`clm_test_spec.yaml`** (wavelength error, power error, power stability, PER).

---

## 9. Quick checklist

1. Launch the GUI.  
2. Enter **DUT SN**, operator, tile index(es), comment if needed.  
3. **Start test station** — confirm Log looks healthy.  
4. Select tests on **Acceptance test**.  
5. **Run selected tests** — answer the fiber prompt if shown.  
6. Monitor the Log until finished; note elapsed times.  
7. Open the results folder under `data_dir` and confirm CSVs / report / note as expected.  
8. Station should be closed at the end of the run (or use **Close down**).

---

## 10. Troubleshooting

| Symptom | What to do |
|---------|------------|
| Warning to enter DUT serial number | Fill **DUT SN** on the first tab. |
| “Select at least one test” | Check one or more boxes on **Acceptance test**. |
| Laser safety cancelled / no tests run | You clicked No or closed the popup. Reconnect fibers and run again, choose Yes. |
| “Busy” / another action still running | Wait for the current start/close/run to finish. |
| Instrument unavailable in Log | Check power, USB/COM/VISA cables, and that the instrument is free (not open in another program). Restart **Start test station**. |
| Report skipped | Only power/frequency and PER feed the report. If neither produced summary data, no report is written. |
| Wrong tile or no response | Confirm **Tile index(es)** matches the physical tile under test. |

If problems continue, capture the Log text and contact the lab engineer who maintains `lab_config.yaml` and `clm_test_spec.yaml`.
