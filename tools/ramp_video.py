"""경사 좌우 비교 영상을 뽑는다. 인자를 읽어 `measure.ramp_video` 를 부른다.

`ramp_test` 가 낸 표와 **같은 지형·같은 명령**이라 표와 나란히 놓을 수 있다.

    python tools/ramp_video.py                       20도와 10도, 좌우 네 편
    python tools/ramp_video.py --각도 20 --속도 0.6   한 각도만
"""

from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("GO1_CEIL", "1")

# `python tools/ramp_video.py` 로 부르면 sys.path[0] 이 tools/ 라 저장소가 안 잡힌다.
# `render_maze.py` 와 같은 처리다.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--각도", type=float, nargs="*", default=[20.0, 10.0],
                    help="경사 각도. **각도마다 다시 컴파일한다** -- 지형이 "
                         "HLO 상수라 캐시 열쇠가 바뀐다")
    ap.add_argument("--방식", nargs="*",
                    default=["횡단·왼쪽오르막", "횡단·오른쪽오르막"])
    ap.add_argument("--속도", type=float, default=0.6,
                    help="vx 명령. 0.6 은 20도 왼쪽에서 넘어지는 값이다")
    ap.add_argument("--스텝", type=int, default=400)
    ap.add_argument("--간격", type=int, default=2, help="프레임 솎기")
    ap.add_argument("--크기", type=int, nargs=2, default=(640, 480))
    ap.add_argument("--출력", default=None)
    args = ap.parse_args()

    from go1_nav.hlc import measure

    made = []
    for deg in args.각도:
        made += measure.ramp_video(args.출력, modes=args.방식, speed=args.속도,
                                   deg=deg, nsteps=args.스텝, stride=args.간격,
                                   px=tuple(args.크기))
    print()
    for p in made:
        print(f"  {p}")


if __name__ == "__main__":
    main()
