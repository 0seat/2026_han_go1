"""장애물 한 칸을 LLC 단독으로 통과시켜 한계를 본다. 인자를 읽어 모듈을 부른다.

    python tools/obstacle_test.py
    python tools/obstacle_test.py --랜드 ROCK ROUGH --속도 0.4 0.8
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("GO1_CEIL", "1")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--랜드", nargs="*", default=["ROCK", "ROUGH", "FLAT"])
    ap.add_argument("--속도", type=float, nargs="*", default=[0.4, 0.6, 0.8])
    ap.add_argument("--폭", type=int, default=3)
    ap.add_argument("--스텝", type=int, default=400)
    args = ap.parse_args()

    from go1_nav.hlc import measure

    rows = measure.obstacle_test(args.랜드, args.속도, width=args.폭,
                                 nsteps=args.스텝)
    print(measure.obstacle_table(rows))


if __name__ == "__main__":
    main()
