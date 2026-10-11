"""Absolute paths to the sibling repositories this sandbox is meant to call."""

from pathlib import Path

SANDBOX_ROOT = Path(__file__).resolve().parents[1]
GITREPOS = SANDBOX_ROOT.parent

CMIS_HOST_CONTROLLER = GITREPOS / "cmis-host-controller"
INSTRUMENT_DRIVERS = GITREPOS / "instrument-drivers"

# CLM helper scripts shipped inside the cmis-host-controller package.
CMIS_CLM_SCRIPTS = (
    CMIS_HOST_CONTROLLER / "src" / "cmis_host" / "platform" / "clm" / "scripts"
)

# Standalone scripts and examples in instrument-drivers. These are not part of
# the installed instrument_drivers package, so call them by path.
INSTRUMENT_DRIVER_SCRIPTS = INSTRUMENT_DRIVERS / "scripts"
INSTRUMENT_DRIVER_EXAMPLES = INSTRUMENT_DRIVERS / "examples"
