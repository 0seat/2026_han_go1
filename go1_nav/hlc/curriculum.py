"""랜드를 이어서 학습한다. **밤새 사람 없이 도는 것을 전제로 짰다.**

한 판씩 손으로 돌리면 판 사이마다 사람이 필요하다. 터널이 40분에 끝나는데
다음 판이 아침에 시작되면 밤이 통째로 논다. 그래서 순서를 여기 적어 두고
한 번에 건다.

사람이 없을 때 지켜야 하는 것 셋
--------------------------------

    파라미터를 판마다 따로 남긴다   뒤 판이 망쳐도 앞 판이 살아 있다
    한 판이 죽어도 다음으로 간다    NaN 하나에 밤이 통째로 날아가지 않게
    되돌아가서 다시 잰다            **잊었는지를 재는 유일한 방법**

마지막 것이 이 모듈의 진짜 이유다. 이어서 학습하면 앞 랜드를 잊는다 --
파국적 망각이라 부르는 것인데, 이 프로젝트는 **한 번도 재 본 적이 없다.**
판마다 끝나고 지금까지 지나온 랜드를 전부 다시 굴려서 도달률을 남긴다.
표가 오른쪽 아래로 내려가면 잊고 있는 것이고, 그러면 섞어서 학습해야 한다.

왜 이 순서인가
--------------

    터널   숙이기.   평지 파라미터에서 이어 간다.  천장 0.32 는 안 숙이면 못 지난다
    돌     비켜가기. 조준 대조군이 0.828 이라 원래 쉽다.  회복 판에 가깝다
    거침   버티기.   지형이 흔들려도 명령을 유지하는 것
    경사   오르기.   `level_after=1` 이라야 진짜 오르막이다.  **이게 제일 어렵다**

턱은 없다. LLC 의 `footswing` 축이 아직 안 열려서 0.06 m 가 그 전의 한계고,
열리면 높이를 다시 재야 한다. 다리 · 도랑 · 벽 · 절벽은 통과 자체가 막혀 있거나
폭이 실측되지 않았다.
"""

from __future__ import annotations

import dataclasses
import time
import traceback
from pathlib import Path

from . import baseline, maze, obs, stage1, train


@dataclasses.dataclass(frozen=True)
class Stage:
    """판 하나. **폴더 이름이 곧 신원이다** -- 숫자를 앞에 붙여 순서를 남긴다."""

    이름: str
    kind: int
    steps: int
    stop_at: float
    level_after: int = 0
    num_envs: int = 8192


#: 밤새 도는 기본 순서. 바꾸려면 여기를 고치고 노트북은 안 건드린다.
#:
#: `stop_at` 이 판마다 다르다. 터널은 천장을 낮춰서 지난 판(0.36, 0.953)보다
#: 어렵고, 돌은 대조군이 이미 0.828 이라 그보다 확실히 위여야 배운 것이 된다.
#: 경사는 오르막을 한 번도 안 돌려 봤으므로 낮게 걸어 둔다 -- 조기 종료가
#: 안 걸리면 그냥 `steps` 를 다 쓴다. 손해는 시간뿐이고 밤은 길다.
STAGES = (
    Stage("3_터널", maze.TUNNEL, 10_000_000, 0.85),
    Stage("4_돌", maze.ROCK, 6_000_000, 0.92),
    Stage("5_거침", maze.ROUGH, 6_000_000, 0.90),
    Stage("6_경사", maze.RAMP, 10_000_000, 0.80, level_after=1),
)


def run(stages=STAGES, *, restore, root, n=64, seed=0, baselines=True):
    """`stages` 를 차례로 돌린다. 앞 판의 파라미터를 뒤 판이 이어받는다.

    `restore` -- 첫 판이 이어받을 파라미터. 경로면 `train.load` 로 읽는다.
    `root`    -- 판별 폴더가 생길 자리. `root/<판 이름>/params_latest.pkl`.

    되돌려주는 것은 표를 만들 행들이다. 배열이 아니라 dict 목록이라 노트북에서
    그대로 찍어도 화면이 안 터진다.
    """
    root = Path(root)
    if isinstance(restore, (str, Path)):
        restore = train.load(restore)

    params = restore
    rows, done = [], []
    t0 = time.perf_counter()
    print(f"관측 {obs.SIZE}  서명 {obs.SIGNATURE}  판 {len(stages)}개", flush=True)

    for st in stages:
        name = baseline._NAMES[int(st.kind)]
        print(f"\n=== {st.이름}  ({name})  {st.steps:,} 스텝  "
              f"목표 도달 {st.stop_at}", flush=True)
        try:
            task = stage1.Task(int(st.kind), level_after=st.level_after)
        except Exception:
            traceback.print_exc()
            continue

        if baselines:
            for tag, pol in (("고정", baseline.fixed()), ("조준", baseline.aim())):
                r = baseline.evaluate(task, pol, n=n, seed=seed)
                print(f"  대조군 {tag}  도달 {r['도달률']:.3f}  "
                      f"넘어짐 {r['넘어짐률']:.3f}  "
                      f"시간초과 {r['시간초과률']:.3f}", flush=True)
                rows.append({"판": st.이름, "랜드": name, "무엇": f"대조군 {tag}",
                             "도달": r["도달률"]})

        try:
            _, params, _, _ = train.train(
                task, num_timesteps=st.steps, num_envs=st.num_envs,
                num_evals=10, seed=seed, restore=params,
                video_dir=root / st.이름,
                stop_at=st.stop_at, stop_patience=2)
        except Exception:
            # 한 판이 죽어도 밤은 계속된다. 파라미터는 앞 판 것을 그대로 들고 간다.
            traceback.print_exc()
            print(f"  {st.이름} 실패 -- 앞 판 파라미터로 다음 판에 간다", flush=True)
            del task
            continue

        done.append((st, task))
        rows.extend(_recall(params, done, n=n, seed=seed))

    print(f"\n전체 {time.perf_counter() - t0:.0f}초", flush=True)
    print(table(rows), flush=True)
    return rows


def _recall(params, done, *, n, seed):
    """지금 파라미터로 **지나온 랜드를 전부 다시** 굴린다.

    이것이 파국적 망각의 측정이다. 마지막 줄만 높고 앞줄이 내려가 있으면
    이어서 학습한 것이 앞을 덮어쓴 것이고, 그러면 판을 섞어야 한다.
    """
    out = []
    for st, task in done:
        name = baseline._NAMES[int(st.kind)]
        r = baseline.evaluate(task, train.policy(params, task), n=n, seed=seed)
        out.append({"판": done[-1][0].이름, "랜드": name, "무엇": "복습",
                    "도달": r["도달률"]})
        print(f"  복습 {name:4s}  도달 {r['도달률']:.3f}  "
              f"넘어짐 {r['넘어짐률']:.3f}", flush=True)
    return out


def table(rows) -> str:
    """행들을 사람이 읽는 표로. 노트북이 배열을 되돌려받지 않게 문자열로 낸다."""
    out = ["", "밤샘 학습 결과", "",
           f"  {'판':10s} {'랜드':6s} {'무엇':10s} {'도달':>6s}", "  " + "-" * 36]
    for r in rows:
        out.append(f"  {r['판']:10s} {r['랜드']:6s} {r['무엇']:10s} "
                   f"{r['도달']:6.3f}")
    out.append("")
    out.append("  복습 줄이 앞 랜드에서 내려가 있으면 잊은 것이다.")
    return "\n".join(out)
