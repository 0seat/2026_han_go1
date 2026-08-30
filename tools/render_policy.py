"""학습된 정책을 영상으로. **콜랩 말고 여기서 돌린다.**

    python tools/render_policy.py 경로/params_latest.pkl
    python tools/render_policy.py 경로/params_latest.pkl --랜드 터널
    python tools/render_policy.py 경로/params_latest.pkl --성공판 --판수 5

왜 로컬인가
-----------

콜랩의 `MUJOCO_GL=egl` 이 하드웨어 가속을 못 잡고 소프트웨어로 그린다. 실측이
프레임당 1 초를 넘는다. GPU 없는 로컬 윈도우가 0.161 초라 **6배 빠르다.**
그래서 `train(render=False)` 가 기본이고, 파라미터만 드라이브에 남긴 뒤 영상은
여기서 뽑는다.

200 스텝 한 편이 약 32 초다.

여기에는 판단도 계산도 없다
---------------------------

정책을 되세우는 것은 `train.policy`, 굴리고 녹화하는 것은 `stage1.debug_video`,
프레임을 만드는 것은 `measure.render_frames` 다. 이 파일은 인자를 읽고 그것들을
부를 뿐이다.

천장 스위치를 자동으로 맞춘다
-----------------------------

`obs.CEIL_ON` 은 `GO1_CEIL` 환경변수로 정해지고 **임포트 시점에 굳는다.** 그런데
어떤 값으로 학습했는지는 pkl 안에 적혀 있다. 그래서 `go1_nav` 를 부르기 전에
파일을 날로 열어 관측 크기를 읽고 스위치를 맞춘다. 안 그러면 `train.load` 가
서명이 다르다며 거부하고, 사용자가 그 이유를 스스로 알아내야 한다.
"""
import argparse
import os
import pickle
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

#: 랜드 이름 -> `maze` 상수 이름. 사용자가 숫자를 외우지 않게 한다.
LANDS = {
    "평지": "FLAT", "경사": "RAMP", "턱": "STEP", "도랑": "GAP",
    "돌": "ROCK", "벽": "WALL", "거침": "ROUGH", "다리": "BRIDGE",
    "터널": "TUNNEL", "절벽": "PIT",
}


def _peek_ceiling(path):
    """`go1_nav` 를 임포트하기 전에 pkl 에서 천장 설정만 훔쳐본다.

    서명 항목에 `천장켜짐` 이 들어 있다. 옛 파일이면 액터 크기로 유추한다 --
    255 면 천장이 있고 164 면 없다.
    """
    try:
        with open(path, "rb") as f:
            blob = pickle.load(f)
    except Exception:
        return None
    if not isinstance(blob, dict):
        return None
    fields = blob.get("fields") or {}
    if "천장켜짐" in fields:
        return bool(fields["천장켜짐"])
    size = fields.get("액터크기")
    return None if size is None else size > 200


def main():
    ap = argparse.ArgumentParser(description="학습된 정책을 영상으로 뽑는다")
    ap.add_argument("params", help="train.save 가 쓴 pkl 경로")
    ap.add_argument("--랜드", default="평지", choices=sorted(LANDS),
                    help="어느 랜드에서 굴릴까 (기본 평지)")
    ap.add_argument("--차선", nargs="+", default=None, choices=sorted(LANDS),
                    help="차선 복도로 굴린다. 랜드를 나열하면 그 순서로 차선이 "
                         "된다. 주면 --랜드 는 무시한다")
    ap.add_argument("--출력", default=None, help="mp4 경로 (기본 outputs/ 아래)")
    ap.add_argument("--스텝", type=int, default=None, help="에피소드 상한")
    ap.add_argument("--판수", type=int, default=3,
                    help="원하는 판을 찾느라 돌려 볼 횟수")
    ap.add_argument("--성공판", action="store_true",
                    help="실패판 대신 성공판을 녹화한다")
    ap.add_argument("--간격", type=int, default=2, help="프레임 솎기 (2 면 5 Hz)")
    ap.add_argument("--요각지터", type=float, default=None,
                    help="시작 요각 흔들기 (도). 기본은 stage1 의 값")
    ap.add_argument("--씨앗", type=int, default=0,
                    help="시작 seed. 차선 복도에서 원하는 차선이 안 나오면 "
                         "이걸 옮긴다 -- seed 가 차선을 정한다")
    ap.add_argument("--밀기", type=float, default=0.0,
                    help="출발 지점을 진행 방향으로 미는 거리 (m). 경사 커리큘럼")
    ap.add_argument("--단", type=int, default=0,
                    help="장애물 뒤쪽 높이 단. 경사는 1 로 둘 것")
    args = ap.parse_args()

    ceiling = _peek_ceiling(args.params)
    if ceiling is not None:
        os.environ["GO1_CEIL"] = "1" if ceiling else "0"
        print(f"  pkl 이 말하는 천장 설정: {'켜짐' if ceiling else '꺼짐'}")

    import math
    from go1_nav.hlc import maze, obs, stage1, train      # noqa: E402

    if args.차선:
        kind = [getattr(maze, LANDS[n]) for n in args.차선]
        name = "+".join(args.차선)
    else:
        kind = getattr(maze, LANDS[args.랜드])
        name = args.랜드
    kw = {"level_after": args.단, "start_shift": args.밀기}
    if args.요각지터 is not None:
        kw["yaw_jitter"] = math.radians(args.요각지터)

    task = stage1.Task(kind, **kw)
    print(f"  관측 {obs.SIZE}  서명 {obs.SIGNATURE}  랜드 {name}"
          f"  차선 {task.n_lanes}")

    params = train.load(args.params)
    policy = train.policy(params, task)

    out = args.출력 or os.path.join(ROOT, "outputs", f"정책_{name}.mp4")
    import jax
    summary = stage1.debug_video(
        task, policy, out, rng=jax.random.PRNGKey(args.씨앗),
        nsteps=args.스텝 or stage1.MAX_STEPS,
        tries=args.판수, stride=args.간격,
        prefer="성공" if args.성공판 else "fail")
    print(f"  {summary}")
    print(f"  저장 {out}")


if __name__ == "__main__":
    main()
