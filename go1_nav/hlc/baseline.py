"""0단계 대조군 -- **학습이 아무것도 안 했을 때의 성적.**

PPO 곡선을 해석하려면 이것이 먼저 있어야 한다. 기준선이 없으면 보상이 올라가도
그것이 학습인지 과제가 원래 쉬운 것인지 구분할 방법이 없다.

두 대조군을 잰다
----------------

    고정 명령   `vx` 만 상수. 조향이 없다
                제일 낮은 바닥. 요각 지터 때문에 대개 옆으로 흘러 나간다

    목표 조준   목표 방향으로 `yaw` 를 P 제어하고 정렬된 만큼 `vx` 를 낸다
                사람이 손으로 짤 만한 가장 단순한 제어기.  **이것이 진짜 바닥이다**

PPO 가 "목표 조준"을 못 이기면 배운 것이 없다. 고정 명령만 이겨 놓고 학습됐다고
말하면 안 된다.

조준 정책은 관측만 본다
-----------------------

특권 정보를 쓰지 않는다. `obs.guide` 가 관측 꼬리에 넣어 둔 **로봇 프레임 방향
단위벡터**를 그대로 읽는다. 그래야 PPO 와 같은 정보로 겨루는 셈이 된다.

지도 좌표나 목표의 절대 위치를 쓰면 대조군이 정책보다 더 많이 아는 상태가 되어
비교가 무의미해진다.
"""

from __future__ import annotations

import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from ..common import path as path_enc
from ..llc import spec
from . import action, obs, skills, stage1

#: 관측에서 길잡이가 시작하는 자리. `obs` 의 배치가 바뀌면 여기도 바뀐다 --
#: 숫자를 옮겨 적지 않고 `obs` 의 상수에서 유도한다.
#:
#: **주의 —** 유도해 두면 자동으로 따라오는 것이 아니라, 덩어리가 새로 끼면
#: 여기에도 더해야 한다. 2026-08-21 에 천장 스캔이 들어오면서 실제로 그랬다.
GUIDE_AT = (obs.PROPRIO_SIZE * obs.STACK + obs.SCAN_SIZE + obs.CEIL_SIZE)

#: 조준 제어기의 이득. 튜닝하지 않았다 -- 대조군은 잘 만들 대상이 아니라
#: **넘어야 할 바닥**이고, 손으로 짠 티가 나는 편이 정직하다.
AIM_YAW_GAIN = 1.5
AIM_SPEED = 0.6


def fixed(vx=0.6):
    """고정 명령. 조향이 없다."""
    raw = np.zeros(skills.ACTION_SIZE, dtype=np.float32)
    raw[spec.index("vx")] = vx / action.SCALE["vx"]
    raw = jnp.asarray(raw)

    def policy(observation, key):
        del observation, key
        return raw
    return policy


def aim(speed=AIM_SPEED, yaw_gain=AIM_YAW_GAIN):
    """목표 조준. 관측의 길잡이만 읽는다.

    `obs.guide` 의 앞 세 숫자가 가장 가까운 경유점의 `[방향x, 방향y, 압축거리]`
    이고, 방향은 이미 **로봇 프레임 단위벡터**다. 그래서 `arctan2` 한 번으로
    방위각이 나온다 -- `common/path.py` 가 각도 대신 단위벡터를 내는 이유가
    여기서 값을 한다. `+pi` 와 `-pi` 의 이음매가 없다.

    정렬된 만큼만 전진하는 것(`cos`)이 요점이다. 안 그러면 목표가 옆에 있는데
    앞으로 달려 나간다.
    """
    ivx, iyaw = spec.index("vx"), spec.index("yaw")

    def policy(observation, key):
        del key
        o = observation["state"] if isinstance(observation, dict) else observation
        g = o[GUIDE_AT:GUIDE_AT + 3]
        bearing = jnp.arctan2(g[1], g[0])
        raw = jnp.zeros(skills.ACTION_SIZE)
        forward = speed * jnp.clip(jnp.cos(bearing), 0.0, 1.0)
        raw = raw.at[ivx].set(forward / action.SCALE["vx"])
        raw = raw.at[iyaw].set(jnp.clip(yaw_gain * bearing, -1.0, 1.0))
        return raw
    return policy


def evaluate(task, policy, n=32, seed=0, nsteps=stage1.MAX_STEPS):
    """에피소드 `n` 개를 **한꺼번에** 돌리고 통계를 낸다.

    하나씩 돌리면 로컬에서 못 쓸 만큼 느리다. `vmap` 으로 배치를 만든다 --
    학습이 어차피 요구하는 경로라 여기서 미리 확인하는 셈도 된다.

    조기 종료를 하지 않는다. 배치마다 끝나는 시점이 다른데 `jit` 안에서는 자를
    수 없으므로, 끝난 환경은 `done` 을 세워 두고 계속 돌린다. 지표는 **처음
    끝난 시점의 값**을 쓴다.
    """
    reset = jax.jit(jax.vmap(task.reset))
    step = jax.jit(jax.vmap(task.step))
    vpolicy = jax.jit(jax.vmap(policy))

    keys = jax.random.split(jax.random.PRNGKey(int(seed)), int(n))
    st = reset(keys)
    live = np.ones(n, dtype=bool)
    reached = np.zeros(n, dtype=bool)
    fell = np.zeros(n, dtype=bool)
    steps = np.zeros(n, dtype=np.int32)
    dist = np.asarray(st.metrics["목표거리"]).copy()
    total = np.zeros(n, dtype=np.float64)

    t0 = time.perf_counter()
    for i in range(nsteps):
        keys = jax.random.split(keys[0], n)
        st = step(st, vpolicy(st.obs, keys))
        d = np.asarray(st.done) > 0
        total += np.where(live, np.asarray(st.reward), 0.0)
        newly = live & d
        reached |= newly & (np.asarray(st.metrics["도달"]) > 0)
        fell |= newly & (np.asarray(st.metrics["넘어짐"]) > 0)
        steps = np.where(newly, i + 1, steps)
        dist = np.where(live, np.asarray(st.metrics["목표거리"]), dist)
        live &= ~d
        if not live.any():
            break
    steps = np.where(steps == 0, i + 1, steps)

    return {
        "표본": int(n),
        "도달률": round(float(reached.mean()), 3),
        "넘어짐률": round(float(fell.mean()), 3),
        "시간초과률": round(float((~reached & ~fell).mean()), 3),
        "보상평균": round(float(total.mean()), 3),
        "남은거리_평균": round(float(dist.mean()), 3),
        "도달까지_스텝": int(np.median(steps[reached])) if reached.any() else -1,
        "초": round(time.perf_counter() - t0, 1),
    }


def compare(kinds, video_dir, *, n=32, seed=0, level_after=None, task_kw=None,
            verbose=True):
    """랜드 종류마다 두 대조군을 재고 표로 낸다. 영상도 남긴다.

    `video_dir` 에 기본값이 없다. 숫자만 보고 판정하면 틀린 이유로 통과한 것을
    못 잡는다 -- 이 프로젝트에서 이미 두 번 당했다 (`hlc/measure.py`).
    """
    from . import maze
    from .measure import save_video

    video_dir = Path(video_dir)
    video_dir.mkdir(parents=True, exist_ok=True)
    level_after = level_after or {}
    task_kw = task_kw or {}
    rows = []

    for kind in kinds:
        name = _NAMES[int(kind)]
        task = stage1.Task(int(kind), level_after=level_after.get(int(kind), 0),
                           **task_kw)
        for tag, policy in (("고정", fixed()), ("조준", aim())):
            r = evaluate(task, policy, n=n, seed=seed)
            rows.append({"랜드": name, "대조군": tag, **r})
            if verbose:
                print(f"  {name:4s} {tag}  도달 {r['도달률']:.3f}  "
                      f"넘어짐 {r['넘어짐률']:.3f}  시간초과 {r['시간초과률']:.3f}  "
                      f"남은거리 {r['남은거리_평균']:.2f} m  ({r['초']:.0f}초)",
                      flush=True)
        # 영상은 조준 대조군만. 고정은 옆으로 흘러 나가는 것을 볼 뿐이다.
        summary, frames = stage1.rollout(
            task, aim(), jax.random.PRNGKey(seed), record=True)
        # stride 2 -> 5 Hz 영상. 렌더가 평가보다 비싸서 절반으로 줄인다.
        # 무엇이 일어났는지 보는 데는 5 Hz 로 충분하다.
        save_video(task.env, frames, video_dir / f"baseline_{name}.mp4",
                   fps=10, stride=2)
        if verbose:
            print(f"       영상 {summary}", flush=True)
        del task
    return rows


_NAMES = {0: "평지", 1: "경사", 2: "턱", 3: "도랑", 4: "돌",
          5: "벽", 6: "거침", 7: "다리", 8: "터널", 9: "절벽"}


def table(rows) -> str:
    """행들을 사람이 읽는 표로. 노트북이 배열을 되돌려받지 않게 문자열로 낸다."""
    out = ["", "0단계 대조군  (학습 없음)", "",
           "  랜드  대조군   도달    넘어짐  시간초과  남은거리"]
    for r in rows:
        out.append(f"  {r['랜드']:4s} {r['대조군']:4s}  {r['도달률']:6.3f}  "
                   f"{r['넘어짐률']:6.3f}  {r['시간초과률']:7.3f}  "
                   f"{r['남은거리_평균']:7.2f} m")
    aim_rows = [r for r in rows if r["대조군"] == "조준"]
    if aim_rows:
        best = max(r["도달률"] for r in aim_rows)
        out += ["", f"  PPO 가 넘어야 할 바닥 = 조준 대조군의 도달률",
                f"  가장 높은 것 {best:.3f}"]
    return "\n".join(out)
