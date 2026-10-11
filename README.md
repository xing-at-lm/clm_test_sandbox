# CLM test sandbox

Place new scripts in `scripts/`. They run against the local checkouts of:

- `../cmis-host-controller` (import as `cmis_host`)
- `../instrument-drivers` (import as `instrument_drivers`)

Both are installed editable into this project's virtual environment, so edits in those repos show up here without a reinstall. Python must be 3.11, 3.12, or 3.13 (`cmis-host-controller` does not support 3.14).

## Setup

From this directory:

```bat
py -3.13 -m venv .venv
.venv\Scripts\python -m pip install -U pip
.venv\Scripts\python -m pip install -r requirements.txt
```

## Writing a script

Run scripts with the sandbox interpreter:

```bat
.venv\Scripts\python scripts\your_script.py
```

Import library code directly:

```python
from cmis_host.platform.clm.scripts.example import run_example_single_tile
import instrument_drivers
```

CLM scripts live in the `cmis_host` package (`cmis_host.platform.clm.scripts`). Instrument-driver helpers under `instrument-drivers/scripts` and `instrument-drivers/examples` are not installed as modules. Locate those files with `clm_test_sandbox.repos`:

```python
from clm_test_sandbox.repos import (
    CMIS_CLM_SCRIPTS,
    INSTRUMENT_DRIVER_EXAMPLES,
    INSTRUMENT_DRIVER_SCRIPTS,
)
```
