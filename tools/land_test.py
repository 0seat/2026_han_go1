"""랜드 하나짜리 복도를 잰다. **옛 판을 지금 파라미터로 다시 재는 데 쓴다.**

    python tools/land_test.py --체크포인트 hlc5/21_가중/params_latest.pkl --랜드 경사 --단 1
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("GO1_CEIL", "1")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--체크포인트", required=True)
    ap.add_argument("--랜드", nargs="*", default=["경사"])
    ap.add_argument("--단", type=int, default=1)
    ap.add_argument("--판수", type=int, default=128)
    ap.add_argument("--스텝", type=int, default=None)
    args = ap.parse_args()

    from go1_nav import paths
    from go1_nav.hlc import maze, measure, stage1, train

    by_name = {v: k for k, v in maze.NAMES.items()}
    params = train.load(paths.params_file(args.체크포인트))
    for name in args.랜드:
        kind = by_name[name]
        lv = args.단 if kind == maze.RAMP else 0
        task = stage1.Task(kind, level_after=lv)
        print(f"\n=== {name}  단 {lv} ===", flush=True)
        measure.lane_report(task, train.policy(params, task),
                            n=args.판수, nsteps=args.스텝)


if __name__ == "__main__":
    main()
