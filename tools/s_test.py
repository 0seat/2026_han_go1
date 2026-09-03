"""S자 복도를 **로컬에서** 재고 찍는다.

콜랩 드라이브 마운트가 끊기면 여기로 온다. 인자를 읽어 모듈을 부르는 것이
전부다 -- 판단이 들어가는 코드는 `lands` · `measure` · `stage1` 에 있다.

    python tools/s_test.py --판수 16 --스텝 500
    python tools/s_test.py --배치 돌고넘기 --영상만
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("GO1_CEIL", "1")

BATCHES = {
    "넘고돌기": False,                      # 장애물이 꺾임 앞
    "돌고넘기": True,                       # 장애물이 꺾임 뒤
    "섞음": [False, True, False, True],
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--체크포인트", default="hlc5/17_꺾임뒤/params_latest.pkl")
    ap.add_argument("--배치", nargs="*", default=list(BATCHES))
    ap.add_argument("--판수", type=int, default=16)
    ap.add_argument("--스텝", type=int, default=500)
    ap.add_argument("--구간", type=int, default=3)
    ap.add_argument("--씨앗", type=int, default=0)
    ap.add_argument("--출력", default=None)
    ap.add_argument("--영상만", action="store_true")
    ap.add_argument("--표만", action="store_true")
    args = ap.parse_args()

    from go1_nav import paths
    from go1_nav.hlc import lands, maze, measure, stage1, train

    params = train.load(paths.params_file(args.체크포인트))
    out = paths.outputs("S자") if args.출력 is None else Path(args.출력)
    out.mkdir(parents=True, exist_ok=True)
    kinds = [maze.TUNNEL, maze.BRIDGE, maze.ROCK, maze.RAMP]

    for name in args.배치:
        h, c, p = lands.snake_corridor(kinds, run=args.구간,
                                       after_turn=BATCHES[name],
                                       level_after=[0, 0, 0, 1])
        task = stage1.Task({"height": h, "ceiling": c, "plan": p})
        pol = train.policy(params, task)
        route = p["lane_route"][0]
        length = sum(float(((b - a) ** 2).sum() ** 0.5)
                     for a, b in zip(route[:-1], route[1:]))
        print(f"\n=== {name} ===  꺾임 {len(kinds) - 1}번  경로 {length:.0f} m",
              flush=True)

        if not args.영상만:
            t0 = time.time()
            measure.lane_report(task, pol, n=args.판수, seed=args.씨앗,
                                nsteps=args.스텝)
            print(f"  측정 {time.time() - t0:.0f}초", flush=True)
        if not args.표만:
            t0 = time.time()
            print(stage1.debug_video(task, pol, out / f"S_{name}.mp4",
                                     nsteps=args.스텝, tries=8, stride=4))
            print(f"  영상 {time.time() - t0:.0f}초", flush=True)


if __name__ == "__main__":
    main()
