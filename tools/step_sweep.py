"""턱 한계를 **학습된 축 조합**으로 잰다. 인자를 읽어 모듈을 부른다.

    python tools/step_sweep.py
    python tools/step_sweep.py --체크포인트 \
        phase19A_cmd_footswing_native_gait_pilot3M/20260824_094058/final_brax_checkpoint
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("GO1_CEIL", "1")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--체크포인트", default=None,
                    help="walking() 아래 상대 경로. 안 주면 지금 쓰는 LLC")
    ap.add_argument("--높이", type=float, nargs="*",
                    default=[0.04, 0.06, 0.08, 0.10, 0.12])
    ap.add_argument("--vx", type=float, nargs="*", default=[0.4, 0.6, 0.9])
    ap.add_argument("--pitch", type=float, nargs="*", default=[-0.3, 0.0, 0.3])
    ap.add_argument("--height", type=float, nargs="*", default=[0.22, 0.27, 0.32])
    ap.add_argument("--stance", type=float, nargs="*", default=None,
                    help="미학습으로 표시된 축. 응답이 있는지 같이 본다")
    ap.add_argument("--씨앗", type=int, default=16)
    ap.add_argument("--스텝", type=int, default=None)
    args = ap.parse_args()

    from go1_nav import paths
    from go1_nav.hlc import measure

    ckpt = None if args.체크포인트 is None else paths.walking() / args.체크포인트
    grid = {"vx": args.vx, "pitch": args.pitch, "height": args.height}
    if args.stance:
        grid["stance_width"] = args.stance
    print(f"체크포인트 {ckpt or '(기본 phase18)'}", flush=True)
    rows = measure.step_sweep(args.높이, grid, seeds=args.씨앗,
                              nsteps=args.스텝, checkpoint=ckpt)
    print(measure.step_table(rows))
    for ax in grid:
        print(measure.step_table(rows, axis=ax))


if __name__ == "__main__":
    main()
