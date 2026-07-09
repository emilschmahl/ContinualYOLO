import platform
import site
import subprocess
import sys
from pathlib import Path


def run(cmd, cwd=None):
    print(f"\n> {' '.join(map(str, cmd))}")
    subprocess.check_call(cmd, cwd=cwd)


def main():
    print("=" * 60)
    print("ContinualYOLO Setup")
    print("=" * 60)

    root = Path(__file__).resolve().parent
    sam2_root = root / "segment-anything-2-real-time"

    # ------------------------------------------------------------
    # Check virtual environment
    # ------------------------------------------------------------
    if sys.prefix == sys.base_prefix:
        print(
            "\nWARNING: No virtual environment appears to be active."
            "\nIt is strongly recommended to activate your venv first.\n"
        )

    # ------------------------------------------------------------
    # Check SAM2 submodule
    # ------------------------------------------------------------
    if not sam2_root.exists():
        sys.exit(
            "\nERROR: Could not find 'segment-anything-2-real-time'.\n\n"
            "Did you clone the repository with submodules?\n\n"
            "Run:\n"
            "    git submodule update --init --recursive"
        )

    if not (sam2_root / "sam2").exists():
        sys.exit(
            "\nERROR: The SAM2 submodule appears to be empty.\n\n"
            "Run:\n"
            "    git submodule update --init --recursive"
        )

    # ------------------------------------------------------------
    # Install requirements
    # ------------------------------------------------------------
    print("\nInstalling Python dependencies...")
    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-r",
            str(root / "requirements.txt"),
        ]
    )

    # ------------------------------------------------------------
    # Link SAM2 into the active environment
    # ------------------------------------------------------------
    print("\nLinking SAM2...")

    site_packages = Path(site.getsitepackages()[0])
    pth_file = site_packages / "sam2.pth"
    pth_file.write_text(str(sam2_root.resolve()), encoding="utf-8")

    print(f"Created: {pth_file}")

    # ------------------------------------------------------------
    # Download checkpoints
    # ------------------------------------------------------------
    print("\nDownloading SAM2 checkpoints...")

    download_script = "download_checkpoints.py"

    run(
        [
            sys.executable,
            str(download_script),
            "--script",
            r"./segment-anything-2-real-time/checkpoints/download_ckpts.sh",
            "--dest",
            r"./segment-anything-2-real-time/checkpoints/"
        ]
    )

    # ------------------------------------------------------------
    # Verify installation
    # ------------------------------------------------------------
    print("\nVerifying installation...")

    try:
        from sam2.build_sam import build_sam2_camera_predictor

        _ = build_sam2_camera_predictor
    except Exception as e:
        sys.exit(
            "\nInstallation completed, but importing SAM2 failed:\n\n"
            f"{e}"
        )

    print("\n" + "=" * 60)
    print("Setup completed successfully!")
    print("=" * 60)
    print("\nYou can now run ContinualYOLO.")


if __name__ == "__main__":
    main()