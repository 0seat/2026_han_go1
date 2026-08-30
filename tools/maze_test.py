"""미로 구간 판을 **로컬에서** 재고 찍는다.

인자를 읽어 모듈을 부르는 것이 전부다 -- 판단이 들어가는 코드는 `lands` ·
`measure` · `stage1` 에 있다.

    python tools/maze_test.py --체크포인트 hlc6/01_밀집/params_latest.pkl
    python tools/maze_test.py --씨앗 0 --모양 8 16 --밀도 0.7 --턱빼기 \
        --차선 21 --고를것 시간초과
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("GO1_CEIL", "1")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--체크포인트", required=True)
    ap.add_argument("--씨앗", type=int, default=3)
    ap.add_argument("--모양", type=int, nargs=2, default=(4, 10))
    ap.add_argument("--구간", type=int, default=6)
    ap.add_argument("--밀도", type=float, default=None)
    ap.add_argument("--턱빼기", action="store_true")
    ap.add_argument("--역방향", action="store_true",
                    help="정답지를 뒤로도 훑어 차선을 두 배로")
    ap.add_argument("--판수", type=int, default=128)
    ap.add_argument("--스텝", type=int, default=None)
    ap.add_argument("--차선", type=int, nargs="*", default=None,
                    help="영상을 찍을 차선 번호. 안 주면 안 찍는다")
    ap.add_argument("--고를것", default="fail",
                    help="fail · 성공 · 시간초과 · 넘어짐")
    ap.add_argument("--표없이", action="store_true", help="측정을 건너뛴다")
    ap.add_argument("--출력", default=None)
    args = ap.parse_args()

    from go1_nav import paths
    from go1_nav.hlc import lands, maze, measure, stage1, train

    kinds = (tuple(k for k in maze.PLACED if k != maze.STEP)
             if args.턱빼기 else None)
    mz = maze.generate(args.씨앗, shape=tuple(args.모양), kinds=kinds,
                       density=args.밀도)
    h, c, p = lands.maze_segments(mz, span=args.구간,
                                  reverse=args.역방향)
    params = train.load(paths.walking() / args.체크포인트)
    out = (paths.outputs(f"미로{args.씨앗}") if args.출력 is None
           else Path(args.출력))
    out.mkdir(parents=True, exist_ok=True)

    if not args.표없이:
        task = stage1.Task({"height": h, "ceiling": c, "plan": p})
        print(f"씨앗 {args.씨앗}  {args.모양[0]}x{args.모양[1]}  "
              f"구간 {args.구간}칸  차선 {task.n_lanes}", flush=True)
        t0 = time.time()
        measure.lane_report(task, train.policy(params, task),
                            n=args.판수, nsteps=args.스텝)
        print(f"  측정 {time.time() - t0:.0f}초", flush=True)

    # **판을 하나만 세운다.** 차선은 seed 로 고른다 (`Task.seeds_for_lane`).
    # 전에는 차선마다 한 차선짜리 Task 를 새로 만들었는데, 그때마다 jit 이 다시
    # 컴파일돼 차선당 268 초를 냈다. 영상 여덟 편에 36 분이 걸린 것이 그것이고,
    # 프로세스를 넷으로 나눠도 각자 컴파일을 내므로 병렬 이득이 0 이었다.
    if args.차선:
        if args.표없이:
            task = stage1.Task({"height": h, "ceiling": c, "plan": p})
        policy = train.policy(params, task)
        for lane in args.차선:
            print(f"\n차선 {lane}  ({measure._lane_name(p, lane)})", flush=True)
            print(stage1.debug_video(task, policy,
                                     out / f"lane{lane}_{args.고를것}.mp4",
                                     nsteps=args.스텝 or stage1.MAX_STEPS,
                                     tries=12, stride=2, prefer=args.고를것,
                                     lane=lane))


if __name__ == "__main__":
    main()
