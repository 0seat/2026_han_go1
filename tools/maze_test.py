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
    ap.add_argument("--경사", type=float, default=None,
                    help="경사 각도를 이 판에서만 갈아끼운다 (기본은 maze.SLOPE_DEG). "
                         "**다시 컴파일한다** -- 지형이 HLO 상수라 캐시 열쇠가 바뀐다")
    ap.add_argument("--역방향", action="store_true",
                    help="정답지를 뒤로도 훑어 차선을 두 배로")
    ap.add_argument("--판수", type=int, default=128)
    ap.add_argument("--차선만", type=int, nargs="*", default=None,
                    help="이 차선들에만 판을 몰아준다. **다시 컴파일한다** -- "
                         "가중치가 reset 의 상수라 캐시 열쇠가 바뀐다")
    ap.add_argument("--판씨앗", type=int, default=0,
                    help="판 뽑는 씨앗. 프로세스를 나눌 때 서로 다르게 준다")
    ap.add_argument("--배치", type=int, default=128,
                    help="한 번에 굴리는 판 수. **캐시 열쇠라 바꾸면 다시 컴파일한다**")
    ap.add_argument("--스텝", type=int, default=None)
    ap.add_argument("--차선", type=int, nargs="*", default=None,
                    help="영상을 찍을 차선 번호. 안 주면 안 찍는다")
    ap.add_argument("--시도", type=int, default=12,
                    help="원하는 결과를 찾느라 굴려 보는 횟수. 완주 영상처럼 "
                         "한 판이 비싸면 줄인다")
    ap.add_argument("--경로표시", action="store_true",
                    help="정답 경로 칸을 반투명 파란 판으로 깐다")
    ap.add_argument("--시점", default="track",
                    help="track (따라가기) 또는 탑뷰")
    ap.add_argument("--고를것", default="fail",
                    help="fail · 성공 · 시간초과 · 넘어짐")
    ap.add_argument("--표없이", action="store_true", help="측정을 건너뛴다")
    ap.add_argument("--묶음", action="store_true",
                    help="차선 표 대신 랜드 구성별 요약. 차선이 많을 때 쓴다")
    ap.add_argument("--출력", default=None)
    ap.add_argument("--행저장", default=None,
                    help="차선 표를 json 으로도 남긴다. 나중에 교차하려면 필요")
    args = ap.parse_args()

    from go1_nav import paths
    from go1_nav.hlc import lands, maze, measure, stage1, train

    if args.경사 is not None:
        # **`HIGH` 만 갈아끼운다.** `ELEVATION` 은 고정 상수라 안 따라오고, 그래서
        # `SPAN` 이 그대로다 -- 관측 서명이 안 바뀐다는 뜻이다 (`maze.ELEVATION`
        # 주석 참고). 각도를 재보려고 파일을 고쳤다 되돌리는 일을 없애려고 둔다.
        import math
        maze.HIGH = ((maze.CELLS_PER_TILE - 1) * maze.CELL
                     * math.tan(math.radians(args.경사)))
        assert maze.ELEVATION >= maze.LEVEL_MAX * maze.HIGH + maze.WALL_HEIGHT
        print(f"  경사 {args.경사}도  단 높이 {maze.HIGH:.4f} m", flush=True)

    kinds = (tuple(k for k in maze.PLACED if k != maze.STEP)
             if args.턱빼기 else None)
    mz = maze.generate(args.씨앗, shape=tuple(args.모양), kinds=kinds,
                       density=args.밀도)
    h, c, p = lands.maze_segments(mz, span=args.구간,
                                  reverse=args.역방향)
    params = train.load(paths.params_file(args.체크포인트))
    out = (paths.outputs(f"미로{args.씨앗}") if args.출력 is None
           else Path(args.출력))
    out.mkdir(parents=True, exist_ok=True)

    if not args.표없이:
        task = stage1.Task({"height": h, "ceiling": c, "plan": p})
        if args.차선만:
            # **판을 몰아준다.** 차선당 판이 두셋이면 도달 0.000 이 "못 간다"인지
            # "운이 나빴다"인지 안 갈린다 -- 진짜 승률 0.5 도 3판 연속 실패가
            # 12.5% 다. 의심 차선만 남기고 다시 재면 그 구분이 선다.
            import numpy as _np
            w = _np.zeros(task.n_lanes)
            w[_np.asarray(args.차선만)] = 1.0
            task.set_lane_weight(w)
            print(f"  차선 {len(args.차선만)}개에만 판을 준다", flush=True)
        print(f"씨앗 {args.씨앗}  {args.모양[0]}x{args.모양[1]}  "
              f"구간 {args.구간}칸  차선 {task.n_lanes}", flush=True)
        t0 = time.time()
        rows = measure.lane_report(task, train.policy(params, task),
                                   n=args.판수, seed=args.판씨앗,
                                   nsteps=args.스텝,
                                   표=not args.묶음, batch=args.배치)
        if args.행저장:
            import json
            Path(args.행저장).write_text(
                json.dumps(rows, ensure_ascii=False), encoding="utf-8")
        if args.묶음:
            print(measure.lane_spread(rows))
            print(measure.direction_split(rows))
            print(measure.lane_groups(rows))
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
                                     tries=args.시도, stride=1, prefer=args.고를것,
                                     lane=lane, camera=args.시점,
                                     route=(p["lane_route"][lane]
                                            if args.경로표시 else None)))


if __name__ == "__main__":
    main()
