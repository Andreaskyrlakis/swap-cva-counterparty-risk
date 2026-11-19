"""Re-run the whole analysis on the saved FRED snapshot (no download), then the tests.

    python run_all.py              # calibration + exposure/CVA + collateral/netting + WWR + tests
    python run_all.py --download   # first download a fresh snapshot (results will change)
"""

import subprocess
import sys

STEPS = ["scripts/02_curve_and_calibration.py", "scripts/03_exposure_cva.py",
         "scripts/04_collateral_netting.py", "scripts/05_wrong_way_risk.py"]

if __name__ == "__main__":
    if "--download" in sys.argv:
        STEPS.insert(0, "scripts/01_download_data.py")
    for step in STEPS:
        print(f"\n=== {step} ===", flush=True)
        subprocess.run([sys.executable, step], check=True)
    subprocess.run([sys.executable, "-m", "pytest", "-q"], check=True)
