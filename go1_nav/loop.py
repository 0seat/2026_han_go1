"""뼈대 — 끝에서 끝까지 한 번 돌린다.

    지도 -> nav -> 경로변환 -> HLC -> LLC -> mujoco -> 반복

이 파일이 하는 일은 **잇는 것뿐이다.** 판단도 계산도 여기 두지 않는다.
각 조각은 인자로 받는다. 그래야 nav 담당과 HLC 담당이 이 파일을 안 건드리고
자기 것만 바꿔 끼울 수 있다.

지금 진짜인 것은 LLC와 경로변환뿐이고 nav와 HLC는 더미다.
그래서 **로봇이 목표에 도착하지 않는 것이 정상이다.** 여기서 보려는 것은
행동이 아니라 배선이다.

차원을 이 파일에 적지 않는다. 관측 크기는 env에서, 가중치는 체크포인트에서
나오고 둘이 다르면 첫 행렬곱에서 죽는다. 검사를 따로 두면 같은 일을 두 번 한다.
"""

from __future__ import annotations

import jax
import numpy as np

import jax.numpy as jnp

from .common import path as path_enc
from .hlc import env as hlc_env
from .hlc import stub as hlc_stub
from .llc import loader, spec
from .nav import stub as nav_stub

#: 학습된 6축을 11축 명령으로 펼치는 자리. `spec.expand`는 dict를 받고
#: KeyError를 던져 jit 안에서 못 쓰므로, 숫자만 spec에서 받아 여기서 조립한다.
_BASE = jnp.asarray(spec.BASE_VECTOR, dtype=jnp.float32)
_SLOT = jnp.asarray(spec.TRAINED_INDEX)


def run(checkpoint, *, goal=(4.0, 0.0), nsteps=1000, seed=0,
        nav_path=nav_stub.path, hlc_act=hlc_stub.act,
        world=None, record=False, verbose=True):
    """뼈대를 nsteps 만큼 돌리고 요약을 낸다.

    반환값에 큰 배열을 담지 않는다. 궤적은 (nsteps, 2)라 작지만, 영상 프레임은
    `record=True`일 때만 만들고 여기서 바로 파일로 떨군다.
    """
    env = hlc_env.make(noise_level=0.0)
    obs_size = loader.env_observation_size(env)
    policy_fn = loader.load_policy(checkpoint, obs_size)

    reset = jax.jit(env.reset)
    step = jax.jit(env.step)
    with_command = jax.jit(env.with_command)
    infer = jax.jit(policy_fn)

    key = jax.random.PRNGKey(seed)
    key, sub = jax.random.split(key)
    state = reset(sub)

    goal = np.asarray(goal, dtype=np.float64).reshape(2)
    track = np.empty((nsteps, 2))
    frames = [] if record else None
    fell = -1

    for i in range(nsteps):
        qpos = np.asarray(state.data.qpos)
        robot_xy = qpos[0:2]
        robot_yaw = path_enc.yaw_from_quat(qpos[3:7])
        track[i] = robot_xy

        points = nav_path(world, robot_xy, goal)
        features = path_enc.encode(points, robot_xy, robot_yaw)
        command = _BASE.at[_SLOT].set(hlc_act(features))

        state = with_command(state, command)
        key, sub = jax.random.split(key)
        action, _ = infer(state.obs, sub)
        state = step(state, action)

        if record:
            frames.append(state)
        if bool(state.done) and fell < 0:
            fell = i

    travelled = float(np.linalg.norm(track[-1] - track[0]))
    remaining = float(np.linalg.norm(goal - track[-1]))
    summary = {
        "스텝": nsteps,
        "이동거리_m": round(travelled, 3),
        "목표까지_남은거리_m": round(remaining, 3),
        "넘어진_스텝": fell,
        "경로특징_크기": path_enc.SIZE,
        "관측_크기": obs_size,
        "액션_크기": int(env.action_size),
    }

    if verbose:
        print("=" * 60)
        print("뼈대 1회 주행")
        for k, v in summary.items():
            print(f"  {k:20s} {v}")
        print("=" * 60)
        print("통과 기준: 안 넘어진다 + 앞으로 간다")
        print("목표 도달은 기준이 아니다 -- HLC가 아직 관측을 안 본다")

    if record:
        _save_video(env, frames)
    return summary


def _save_video(env, states, filename="skeleton.mp4", fps=50, stride=2,
                camera="track"):
    """영상 저장. 눈으로 보는 것이 통과 기준의 마지막 한 줄이다.

    ffmpeg이 없는 환경(윈도우 로컬 등)이 흔해서 GIF로 물러난다. 여기서 죽으면
    롤아웃을 다시 돌려야 하는데, 영상은 부수적이라 그럴 값어치가 없다.
    """
    # camera="track"이 아니면 로봇이 화면 밖으로 걸어 나가 아무것도 안 보인다.
    rendered = env.render(states[::stride], camera=camera)
    out_fps = max(fps // stride, 1)
    try:
        import mediapy
        mediapy.write_video(filename, rendered, fps=out_fps)
    except Exception as exc:
        from PIL import Image
        filename = str(filename).rsplit(".", 1)[0] + ".gif"
        print(f"[loop] mp4 실패({type(exc).__name__}), GIF로 저장합니다.")
        images = [Image.fromarray(f) for f in rendered]
        images[0].save(filename, save_all=True, append_images=images[1:],
                       duration=int(1000 / out_fps), loop=0)
    print(f"영상 저장: {filename}")
