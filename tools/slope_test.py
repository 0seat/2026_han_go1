"""경사 한 칸에서 등반과 횡단을 나란히 잰다. 인자를 읽어 모듈을 부른다.

    python tools/slope_test.py
    python tools/slope_test.py --속도 0.3 0.5 0.9 --폭 9
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("GO1_CEIL", "1")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--방식", nargs="*",
                    default=["등반", "횡단·오른쪽오르막", "횡단·왼쪽오르막"],
                    help="등반 · 횡단·오른쪽오르막 · 횡단·왼쪽오르막")
    ap.add_argument("--속도", type=float, nargs="*", default=[0.3, 0.5, 0.9])
    ap.add_argument("--요각", type=float, default=0.0)
    ap.add_argument("--폭", type=int, default=7)
    ap.add_argument("--스텝", type=int, default=400)
    args = ap.parse_args()

    from go1_nav.hlc import measure

    rows = measure.ramp_test(args.방식, args.속도, width=args.폭,
                             nsteps=args.스텝, yaw_cmd=args.요각)
    print(measure.ramp_table(rows))


if __name__ == "__main__":
    main()
