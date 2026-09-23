"""Offline integrity check.

    python -m scripts.maintenance integrity

Reports persons with no embedding and clips missing their embedding. Works
without the API running. Exits non-zero when problems are found, so cron/CI
can alert.
"""

import argparse
import json
import sys

from src.voice_service import initialize_system, integrity_report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["integrity"])
    parser.parse_args()

    initialize_system()
    report = integrity_report()
    print(json.dumps(report, indent=2))
    return 0 if report["consistent"] else 1


if __name__ == "__main__":
    sys.exit(main())
