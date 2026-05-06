"""Modal app for M6 APG training.

Runs the full CPTServo M6 APG curriculum (5 phases × 20 epochs = 100 epochs,
batch_size=1024, truncation_window=200) on Modal with 64 GB RAM. Persists
checkpoints to a Modal volume; pull `models/apg_best.pt` back locally with:

    modal volume get cptservo-m6 models/apg_best.pt ./models/apg_best.pt

Local M6 attempt died from autograd graph OOM (Windows 32GB host with other
processes competing). Modal's 64GB CPU tier with no contention should handle
the full original spec without dropping batch size or truncation.

Run:
    cd C:\\Users\\Jack\\Documents\\Research\\WIP\\CPTServo
    modal run modal_apg_train.py::train

Bundled local sources:
    - WIP/CPTServo/src/cptservo/   (full package)
    - WIP/CPTServo/configs/        (v1_recipe.yaml — Layer-0 frozen)
    - WIP/CPTServo/data/           (reduced_calibration.json — M2 fitted params)
    - WIP/CPTServo/scripts/        (m6_apg_train.py + run_m3_m4_gates.py)
    - WIP/RbSpec/src/rbspec/       (physical constants used by reduced.py)
"""

from __future__ import annotations

from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Paths — these are LOCAL paths on the user's machine, mounted into the
# container at /app at image-build time.
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent          # WIP/CPTServo/
_RESEARCH = _HERE.parent.parent                  # Research/
_CPTSERVO = _HERE                                 # WIP/CPTServo/
_RBSPEC = _RESEARCH / "WIP" / "RbSpec"           # WIP/RbSpec/

# ---------------------------------------------------------------------------
# Image: Debian slim + Python 3.11 + the deps the project needs
# ---------------------------------------------------------------------------
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch>=2.0",
        "torchdiffeq>=0.2",
        "qutip>=5.0",
        "scipy>=1.10",
        "numpy>=1.24",
        "pyyaml>=6.0",
        "h5py>=3.9",
        "pyarrow>=14.0",
        "matplotlib>=3.7",
        "tqdm>=4.65",
    )
    # Bundle local sources into /app at image-build time
    .add_local_dir(
        local_path=str(_CPTSERVO / "src"),
        remote_path="/app/CPTServo/src",
        ignore=["__pycache__", "*.pyc", "*.egg-info"],
    )
    .add_local_dir(
        local_path=str(_CPTSERVO / "configs"),
        remote_path="/app/CPTServo/configs",
    )
    # Only ship the calibration JSON we actually need — exclude logs/h5/png
    # so Modal doesn't choke on files being written during the build.
    .add_local_file(
        local_path=str(_CPTSERVO / "data" / "reduced_calibration.json"),
        remote_path="/app/CPTServo/data/reduced_calibration.json",
    )
    .add_local_dir(
        local_path=str(_CPTSERVO / "scripts"),
        remote_path="/app/CPTServo/scripts",
        ignore=["__pycache__", "*.pyc"],
    )
    .add_local_dir(
        local_path=str(_RBSPEC / "src"),
        remote_path="/app/RbSpec/src",
        ignore=["__pycache__", "*.pyc", "*.egg-info"],
    )
)

# ---------------------------------------------------------------------------
# Persistent volume — checkpoints survive across runs and can be pulled
# locally via `modal volume get cptservo-m6 ...`
# ---------------------------------------------------------------------------
volume = modal.Volume.from_name("cptservo-m6", create_if_missing=True)

app = modal.App("cptservo-m6-apg", image=image)


@app.function(
    cpu=8.0,
    memory=65536,  # 64 GB RAM
    timeout=14400,  # 4 hours
    volumes={"/cache": volume},
)
def train() -> dict:
    """Run the full M6 APG curriculum on Modal.

    Mirrors `scripts/m6_apg_train.py:main()` with the original spec restored
    (batch_size=1024, truncation_window=200) since memory is no longer the
    binding constraint.
    """
    import os
    import shutil
    import sys

    # Wire up imports for the bundled packages
    sys.path.insert(0, "/app/CPTServo/src")
    sys.path.insert(0, "/app/CPTServo/scripts")
    sys.path.insert(0, "/app/RbSpec/src")

    # cwd matters — scripts use Path(__file__).parents[1] for project root
    os.chdir("/app/CPTServo")

    # Restore the original spec for each phase before invoking the trainer
    import m6_apg_train

    for phase in m6_apg_train.PHASES:
        phase["batch_size"] = 1024
        phase["truncation_window"] = 200

    print(
        "[modal] Running M6 APG curriculum with restored spec "
        "(batch=1024, truncation=200, ~64GB headroom)"
    )
    m6_apg_train.main()

    # Copy fresh outputs to volume so they survive container shutdown
    print("[modal] Copying outputs to /cache volume...")
    if Path("/app/CPTServo/models").exists():
        shutil.copytree(
            "/app/CPTServo/models",
            "/cache/models",
            dirs_exist_ok=True,
        )
    if Path("/app/CPTServo/data").exists():
        shutil.copytree(
            "/app/CPTServo/data",
            "/cache/data",
            dirs_exist_ok=True,
        )
    volume.commit()

    # Report what we have
    saved = list(Path("/cache/models").glob("*.pt")) if Path("/cache/models").exists() else []
    print(f"[modal] Persisted {len(saved)} checkpoints to volume cptservo-m6")
    for p in saved:
        print(f"[modal]   {p.name}: {p.stat().st_size} bytes")

    return {
        "status": "done",
        "checkpoints": [str(p.relative_to("/cache")) for p in saved],
    }


@app.function(volumes={"/cache": volume})
def list_outputs() -> list[str]:
    """Inspect what's in the volume after training."""
    out = []
    if Path("/cache").exists():
        for p in Path("/cache").rglob("*"):
            if p.is_file():
                out.append(f"{p.relative_to('/cache')}  ({p.stat().st_size} bytes)")
    return out


@app.local_entrypoint()
def main() -> None:
    print("Starting M6 APG training on Modal...")
    result = train.remote()
    print(f"Result: {result}")
    print()
    print("To download trained model:")
    print("  modal volume get cptservo-m6 models/apg_best.pt ./models/apg_best.pt")
    print()
    print("To list volume contents:")
    print("  modal volume ls cptservo-m6")
    print("  or: modal run modal_apg_train.py::list_outputs")
