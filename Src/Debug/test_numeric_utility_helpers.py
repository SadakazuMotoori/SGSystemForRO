# --------------------------------------------------
# test_numeric_utility_helpers.py
# Purpose:
#   Verify shared numeric helpers used across H2/H1/M15 modules.
# --------------------------------------------------

import sys
from pathlib import Path


_src_dir = Path(__file__).resolve().parents[1]
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from Framework.Utility.Utility import Clamp01, ToFloat


def main():
    if ToFloat("1.25") != 1.25:
        raise AssertionError("ToFloat should parse numeric strings.")

    if ToFloat(None, 3.5) != 3.5:
        raise AssertionError("ToFloat should return the provided default.")

    if Clamp01(-0.2) != 0.0:
        raise AssertionError("Clamp01 should clamp negative values to 0.0.")

    if Clamp01(1.2) != 1.0:
        raise AssertionError("Clamp01 should clamp values above 1.0.")

    if Clamp01("0.45") != 0.45:
        raise AssertionError("Clamp01 should accept string inputs via ToFloat.")

    print("[OK] numeric utility helper test passed.")


if __name__ == "__main__":
    main()
