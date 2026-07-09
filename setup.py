from pathlib import Path
import site
import sys


def main():
    # ContinualYOLO/
    project_root = Path(__file__).resolve().parent

    # ../segment-anything-2-real-time
    sam2_repo = project_root.parent / "segment-anything-2-real-time"

    if not sam2_repo.exists():
        print("ERROR: Could not find the SAM2 repository.")
        print(f"Expected location:\n{sam2_repo}")
        print("\nThe folder structure should be:")
        print(project_root.parent)
        print("├── ContinualYOLO")
        print("└── segment-anything-2-real-time")
        sys.exit(1)

    # Find site-packages of the active virtual environment
    site_packages = Path(site.getsitepackages()[0])

    # Create the .pth file
    pth_file = site_packages / "sam2.pth"
    pth_file.write_text(str(sam2_repo.resolve()), encoding="utf-8")

    print(f"Created {pth_file}")
    print(f"Linked to {sam2_repo.resolve()}")

    # Verify import
    try:
        from sam2.build_sam import build_sam2_camera_predictor
        print("\n✓ SAM2 successfully linked.")
    except Exception as e:
        print("\n✗ The link was created, but the import failed.")
        print(e)
        sys.exit(1)


if __name__ == "__main__":
    main()