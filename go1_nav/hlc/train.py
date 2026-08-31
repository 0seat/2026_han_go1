"""HLC 학습 -- brax PPO 배선. **판단은 없고 잇기만 한다.**

`stage1.Task` 가 playground 의 `MjxEnv` 라서 이 파일은 얇다. 신경망 모양과
하이퍼파라미터를 LLC 쪽과 맞춰 두는 것, 그리고 **영상을 자동으로 남기는 것**이
여기서 하는 일의 전부다.

LLC 와 같은 모양을 쓰는 이유
----------------------------

phase18 이 512/256/128, silu, tanh_normal, 비대칭 actor-critic 으로 돌았고 그게
이 로봇에서 실제로 수렴한 유일한 조합이다. HLC 가 다른 문제이긴 하지만, 굳이
다르게 시작할 근거가 없다. 다르게 할 이유가 생기면 그때 바꾼다.

`full_reset=True` 가 중요하다
-----------------------------

brax 의 자동 리셋은 기본적으로 **첫 상태를 캐시해 두고 되돌린다.** 그러면
`stage1` 이 애써 흔든 진입 조건(위치 · 요각 · 초기 속도 · LLC 이득)이 에피소드
전체에서 하나로 고정된다. 1단계 설계의 핵심이 통째로 무력해지므로 반드시
`full_reset=True` 다. 벽시계 시간이 조금 늘지만 그 값을 한다.

영상은 선택이 아니다
--------------------

`policy_params_fn` 에 영상 콜백을 건다. 평가 때마다 한 편씩 남는다. 이 프로젝트
에서 표만 보고 틀린 판정을 두 번 냈고(`hlc/measure.py`), 보상 곡선은 무엇이
일어났는지 말해 주지 않는다.
"""

from __future__ import annotations

import functools
import pickle
import time
from pathlib import Path

import jax
import numpy as np

# ---------- jax 0.11 에서 flax 를 살린다 ----------
#
# flax 가 `jax.core.get_opaque_trace_state(convention="flax")` 를 부르는데
# (`flax/core/tracers.py`, `flax/nnx/tracers.py`), 그 이름이 jax 0.11 에서
# 지워졌다. 콜랩이 jax 를 올리면서 **학습이 시작되자마자** 이렇게 죽는다.
#
#     AttributeError: jax.core.get_opaque_trace_state was deprecated in JAX
#     v0.10.0 and removed in JAX v0.11.0.
#     Use jax.extend.core.get_opaque_trace_state.
#
# 임포트가 아니라 **첫 트레이싱에서** 터진다 -- `train()` 을 부른 뒤라 로그에는
# 학습이 시작된 것처럼 보인다.
#
# 옮겨간 것이 이름뿐이라(경고문이 새 자리를 직접 알려 준다) 여기서 이어 준다.
# 판을 흔드는 대신 이 방법을 고른 이유
#
#     flax 를 올린다     brax 와의 궁합을 다시 맞춰야 한다. 콜랩에서 확인 불가
#     jax 를 내린다      jaxlib · CUDA 플러그인까지 따라 내려야 한다
#     이름을 잇는다      같은 함수다. jax 가 이름을 되살리면 저절로 무해해진다
#
# **주의 —** flax 가 이 줄보다 먼저 그 이름을 읽으면 늦는다. 실제 호출이
# 트레이싱 시점이라 지금까지는 늦은 적이 없다.
if not hasattr(jax.core, "get_opaque_trace_state"):
    try:
        import jax.extend.core as _jec

        jax.core.get_opaque_trace_state = _jec.get_opaque_trace_state
    except (ImportError, AttributeError):
        # 못 이었으면 그냥 둔다. 여기서 죽이면 원인이 더 안 보인다.
        pass

from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from mujoco_playground import wrapper

from . import obs, stage1

#: 정규화 표준편차에 더하는 바닥값. **없으면 학습이 발산한다.**
#:
#: `std = sqrt(분산 + std_eps)` 다. 상수 칸은 분산이 0 이라 `sqrt(std_eps)` 가
#: 그대로 std 가 된다. brax 기본값이 `0.0` 이고, 그러면 std 가 하한 `1e-6` 에
#: 붙는다. 그래서 **오래 상수였다가 드물게 바뀌는 칸**이 있으면 그 순간 정규화
#: 출력이 10^5 규모로 튄다.
#:
#: 우리 관측이 정확히 그렇다. 평지에서 액터 관측 164칸 중 109칸이 상수다.
#:
#:     높이 스캔 91칸          평지라 전부 0
#:     미학습 명령 5축 x 3틱    UNTRAINED_HOLD 고정
#:     특권 14칸               랜드 one-hot · 단 · 마찰 · 이득/편향
#:     길잡이 압축거리 2칸      0.292 / 0.661 고정      <- 지뢰
#:
#: 마지막이 치명적이다. 전방 0.5 m 와 1.5 m 점의 거리는 목표가 그보다 멀면 정확히
#: 상수인데, 로봇이 목표에 가까워지면 바뀐다. `(0.66-0.29)/1e-6 = 370,000` 이다.
#: **성공에 가까워지는 바로 그 순간 입력이 폭발한다.**
#:
#: 증거 (2026-08-21, 평지 직진, 천장 없음)
#:
#:     스텝        v_loss        kl_mean
#:     491,520     9,655,768     1412.58
#:     983,040    15,034,950        0.50
#:
#: 수익이 O(10) 인 과제에서 가치 손실이 천만이고 KL 이 1412 다. `total_loss` 가
#: `v_loss` 와 같아 정책 손실(-0.011)은 묻혔다. 정책이 학습된 것이 아니라 가치망
#: 폭주에 끌려다녔다.
#:
#: 값을 1e-2 로 잡은 근거 -- std 바닥이 `sqrt(1e-2) = 0.1` 이 된다. 상수 칸에서
#: 일어나는 가장 큰 도약이 천장의 `1.0 -> 0.09`(0.91)와 길잡이의
#: `0.292 -> 0.999`(0.71)라, 둘 다 정규화 뒤 10 이하로 들어온다.
#:
#: **공짜가 아니다.** 이건 전 채널에 걸리는 바닥이라, 진짜로 분산이 작은 칸도
#: 같이 눌린다. 실측된 std 가 이렇다.
#:
#:     중력 z          0.0137     넘어짐 지표.  약 7배 압축된다
#:     지면 대비 높이   0.0205     약 5배
#:     길잡이 거리      0.0075     약 13배
#:     자이로 y        0.0561     약 2배
#:
#: 압축은 선형이라 신경망이 가중치로 되살릴 수 있다. 폭발은 되살릴 수 없다.
#: 그래서 이쪽을 택한다. 작은 분산 칸이 실제로 문제가 되면 그때 `max_abs_value`
#: 로 바꾼다 -- 그쪽이 더 수술적이지만 `ppo.train` 이 안 열어 줘서 전처리기를
#: 가로채야 한다. **열려 있는 손잡이를 먼저 쓴다.**
OBS_STD_EPS = 1e-2

#: 신경망. phase18 과 같은 모양이다.
NETWORK = dict(
    policy_hidden_layer_sizes=(512, 256, 128),
    value_hidden_layer_sizes=(512, 256, 128),
    policy_obs_key="state",
    value_obs_key="privileged_state",
)

#: PPO 하이퍼파라미터. phase18 의 값에서 셋만 다르다.
#:
#:     discounting     0.97 -> 0.99   HLC 는 10 Hz 라 같은 실시간이 스텝 수로는
#:                                    1/5 다. 지평선을 맞추려면 감마를 올려야 한다.
#:                                    `stage1.DISCOUNT` 와 **같아야 한다** --
#:                                    퍼텐셜 성형이 그 감마를 쓴다
#:     entropy_cost    0.01 -> 0.005  행동이 11축 + 게이트라 LLC(12관절)보다
#:                                    탐색이 싸다. 과하면 명령이 떨려 LLC 가
#:                                    학습 분포를 벗어난다
#:     episode_length  1000 -> 200    HLC 스텝 기준. `stage1.MAX_STEPS`
PPO = dict(
    learning_rate=3e-4,
    entropy_cost=5e-3,
    discounting=stage1.DISCOUNT,
    clipping_epsilon=0.2,
    gae_lambda=0.95,
    unroll_length=20,
    batch_size=256,
    num_minibatches=32,
    num_updates_per_batch=4,
    normalize_observations=True,
    normalize_observations_std_eps=OBS_STD_EPS,
    # **학습 로그와 나중 측정이 같은 자를 쓰게 한다.** brax 기본값은 거짓이라
    # 평가가 확률적 행동으로 돌고, 그러면 로그의 도달률이 `train.policy`
    # (결정적)로 잰 값보다 낮게 나온다. 꺾인 복도에서 0.609 대 1.000 이 나와
    # 판단을 두 번 그르쳤다. 학습 자체는 안 바뀌고 조기 종료만 엄격해진다.
    deterministic_eval=True,
)



def _network_factory():
    return functools.partial(ppo_networks.make_ppo_networks, **NETWORK)


class _Stop(Exception):
    """조기 종료 신호. `ppo.train` 이 멈출 길을 안 열어 줘서 예외로 뚫는다."""


def _check_brax_jax() -> None:
    """brax 의 학습 경로가 이 jax 에서 도는가. **판을 세우기 전에 묻는다.**

    jax 0.11 이 pmap 계열을 걷어내면서 brax 가 쓰는 `jax.device_put_replicated`
    가 사라졌다. 그런데 터지는 자리가 `ppo.train` 안이라, 미로를 굽고 Task 를
    세우고 체크포인트까지 읽은 **뒤에야** 죽는다. 64x64 판이면 그게 몇 분이고,
    로그에는 학습이 시작된 것처럼 보인다.

    이름 하나가 아니라 아키텍처가 바뀐 것이라 이어 붙일 수 없다. 고치는 길은
    jax 를 0.11 미만으로 되돌리는 것뿐이라, 여기서는 **그 사실만 즉시 알린다.**

        pip install -q "jax[cuda12]<0.11" "jaxlib<0.11"     그 뒤 런타임 재시작
    """
    missing = [n for n in ("device_put_replicated", "pmap") if not hasattr(jax, n)]
    if missing:
        raise RuntimeError(
            f"jax {jax.__version__} 에는 brax 가 쓰는 {', '.join(missing)} 가 "
            f"없습니다. jax 를 0.11 미만으로 내리고 런타임을 재시작하세요 -- "
            f'pip install -q "jax[cuda12]<0.11" "jaxlib<0.11"')


def train(task, *, num_timesteps=20_000_000, num_envs=1024, num_evals=10,
          seed=0, save_dir=None, video_dir=None, render=False,
          video_steps=stage1.MAX_STEPS,
          video_every=3, video_tries=3, video_stride=2,
          restore=None, progress=None,
          stop_at=1.0, stop_patience=2, stop_metric="eval/episode_도달",
          **overrides):
    """1단계 학습 한 판.

    **저장 자리와 녹화 자리를 나눈다.** `save_dir` 이 체크포인트, `video_dir` 이
    영상이다. 전에는 `video_dir` 하나가 둘을 겸했고, 렌더가 비싸서 그걸 비웠더니
    **체크포인트까지 같이 꺼져 콜랩에서 9시간을 날렸다.** 이름이 영상인데 저장을
    쥐고 있으면 그 함정을 피할 방법이 없다.

    `save_dir` 을 안 주면 `video_dir` 을 쓴다 -- 옛 호출이 그대로 돈다.
    둘 다 없으면 디스크에 아무것도 안 남는다. 그건 배선 시험용이다.

    `num_timesteps` 는 **HLC 스텝** 수다. LLC 스텝으로는 5배다 -- 2천만이면
    LLC 1억 스텝이고, phase18 의 한 스테이지(800만)보다 훨씬 크다. 처음에는
    100만 정도로 곡선 모양만 보고 늘리는 편이 낫다.

    영상 비용을 줄이는 손잡이가 넷이다. 렌더는 프레임당 1.3 초로 **물리보다
    비싸다** -- 지형이 150 x 250 격자라 매 프레임 수만 개의 삼각형을 CPU 가
    올린다. 평가마다 4분씩 쓰면 학습보다 녹화가 오래 걸린다.

        video_every   몇 번째 평가마다 찍나. 처음과 마지막은 항상 찍는다
        video_steps   에피소드 상한. 기본은 학습과 같은 200 스텝
        video_stride  프레임 솎기. 2 면 5 Hz
        video_tries   실패판을 찾느라 돌려 보는 횟수

    **`video_steps` 를 학습보다 짧게 두지 말 것.** 120 으로 줄였더니 영상이
    12 초에서 잘려서, 아직 걸어가는 중인 로봇이 "중간에 멈춘" 것처럼 보였다.
    영상이 답해야 하는 것은 왜 실패했는가인데 잘리면 그걸 못 본다. 렌더 비용은
    `video_every` 와 `video_stride` 로 줄이는 편이 오해를 안 만든다.

    조기 종료
    ---------

    `stop_metric` 이 `stop_at` 이상으로 `stop_patience` 번 연속 나오면 멈춘다.
    기본은 도달률 1.0 이 두 번이다.

    필요한 이유 -- 평지 판이 2,621,440 스텝에서 도달 1.000 을 찍고 4,587,520
    까지 그대로였다. **2백만 스텝, 약 47분을 천장에서 낭비했다.**

    `ppo.train` 에는 멈추는 인자가 없어서 예외로 빠져나온다. 파라미터는
    `policy_params_fn` 이 먼저 불리면서 챙겨 두므로 잃지 않는다 -- brax 는
    `policy_params_fn` -> 평가 -> `progress_fn` 순서라, 판정이 나올 때 그
    판정에 쓰인 파라미터가 이미 손에 있다.

    끄려면 `stop_at=None`.

    이어 돌리기
    -----------

    `restore` 에 앞 판의 `params` 를 넘기면 거기서 이어 간다. 보행 학습은 곡선이
    가파른 구간에서 끊기기 쉬운데, 매번 처음부터 돌리면 그 구간을 다시 산다.

        mk, params, _, _ = train.train(task, num_timesteps=1_000_000, ...)
        mk, params, _, _ = train.train(task, num_timesteps=3_000_000,
                                       restore=params, ...)

    **주의 —** 정규화 통계까지 같이 복원된다(`params[0]`). 그래서 이어 돌릴 때
    관측 배치가 바뀌어 있으면 조용히 틀린다. 그것을 막으려고 `load` 가 배치
    서명을 대조한다 -- `obs.py` 의 서명 절 참고.
    """
    cfg = dict(PPO)
    cfg.update(overrides)
    frames = []
    held = {"params": None, "make_policy": None, "hits": 0}

    def on_progress(step, metrics):
        frames.append((step, metrics))
        if progress is not None:
            progress(step, metrics)
        else:
            # **v_loss 와 kl 을 같이 찍는다.** 이것을 안 보고 도달률만 보다가
            # 발산을 탐색 실패로 오진했다. 정상 범위는 v_loss 가 O(100) 이하,
            # kl 이 0.01~0.05 다. v_loss 가 10^4 를 넘으면 학습이 아니라 폭주다.
            r = metrics.get("eval/episode_reward", float("nan"))
            g = metrics.get("eval/episode_도달", float("nan"))
            v = metrics.get("training/v_loss", float("nan"))
            k = metrics.get("training/kl_mean", float("nan"))
            print(f"  {step:>10,}  보상 {r:8.3f}  도달 {g:5.3f}"
                  f"  v_loss {float(v):12,.1f}  kl {float(k):8.4f}", flush=True)

        if stop_at is None or held["params"] is None:
            return
        value = metrics.get(stop_metric)
        if value is None:
            return
        held["hits"] = held["hits"] + 1 if float(value) >= float(stop_at) else 0
        if held["hits"] >= int(stop_patience):
            print(f"  조기 종료 -- {stop_metric} 가 {stop_at} 이상으로 "
                  f"{stop_patience}번 연속", flush=True)
            raise _Stop

    seen = {"n": 0}
    # 저장 자리. 안 주면 옛 호출과 같게 `video_dir` 로 떨어진다.
    out_dir = save_dir if save_dir is not None else video_dir
    # **둘 다 없으면 시끄럽게 알린다.** 조용히 넘어가면 몇 시간 뒤에야 안다.
    # 실측 -- `video_dir` 을 주석 처리하고 27.5M 스텝을 저장 없이 돌렸다.
    # 그 전에는 `video_dir=None` 이 저장까지 끄는 바람에 9시간을 날렸다.
    # 저장을 끄는 것이 정당한 경우(배선 시험)가 있으므로 막지는 않는다.
    if out_dir is None:
        print("\n" + "!" * 66, flush=True)
        print("  주의 -- 체크포인트를 저장하지 않습니다.", flush=True)
        print("  save_dir 도 video_dir 도 없습니다. 런타임이 끊기면 전부 잃습니다.",
              flush=True)
        print("  저장하려면  train(..., save_dir=str(paths.walking() / 'hlc8'))",
              flush=True)
        print("!" * 66 + "\n", flush=True)

    def on_params(step, make_policy, params):
        """**평가마다 파라미터를 저장하고**, 가끔 영상을 남긴다.

        저장과 녹화를 분리한 이유가 있다. 저장은 2 MB 에 1 초 미만이고, 녹화는
        1 분이다. 비싼 쪽에 맞춰 둘 다 성기게 하면 끊겼을 때 잃는 것이 커진다.

        저장을 매번 하는 이유 -- 콜랩은 끊긴다. 그리고 코드를 고치면 모듈 캐시
        때문에 런타임을 재시작해야 하는데, 그러면 메모리의 파라미터가 사라진다.
        **이걸로 1.47백만 스텝을 한 번 날렸다.** 그때는 저장이 아예 없었다.
        """
        # 조기 종료가 이걸 되돌려 주므로 자리 설정과 무관하게 챙긴다.
        held["params"], held["make_policy"] = params, make_policy
        i, seen["n"] = seen["n"], seen["n"] + 1
        last = step >= int(num_timesteps)

        # 최신본은 늘 같은 이름으로 덮어쓴다. 되읽을 때 스텝 수를 몰라도 된다.
        if out_dir is not None:
            save(params, Path(out_dir) / "params_latest.pkl", quiet=True)
            save(params, Path(out_dir) / f"params_{step:010d}.pkl", quiet=True)

        # **저장과 녹화를 분리한다.** 렌더는 프레임당 1 초로 물리보다 비싸고,
        # GPU 를 놀린다. 파라미터가 드라이브에 남으므로 영상은 나중에 로컬 CPU 에서
        # `stage1.debug_video` 로 뽑는 편이 낫다 -- 보고 싶을 때만 돌리면 된다.
        if not render or video_dir is None:
            return
        if i % max(1, int(video_every)) and not last:
            return
        policy = make_policy(params, deterministic=True)
        path = Path(video_dir) / f"stage1_{step:010d}.mp4"
        summary = stage1.debug_video(
            task, lambda o, k: policy(o, k)[0], path, nsteps=video_steps,
            tries=video_tries, stride=video_stride)
        print(f"    영상 {path.name}  {summary}", flush=True)

    _check_brax_jax()

    t0 = time.perf_counter()
    try:
        make_policy, params, metrics = ppo.train(
            environment=task,
            num_timesteps=int(num_timesteps),
            num_envs=int(num_envs),
            num_evals=int(num_evals),
            episode_length=stage1.MAX_STEPS,
            action_repeat=1,
            seed=int(seed),
            network_factory=_network_factory(),
            # 진입 조건 무작위화를 살리는 유일한 방법이다. 위 docstring 참고.
            wrap_env_fn=functools.partial(wrapper.wrap_for_brax_training,
                                          full_reset=True),
            progress_fn=on_progress,
            policy_params_fn=on_params,
            **({"restore_params": restore} if restore is not None else {}),
            **cfg,
        )
    except _Stop:
        make_policy, params = held["make_policy"], held["params"]
        metrics = frames[-1][1] if frames else {}
    print(f"학습 {time.perf_counter() - t0:.0f}초", flush=True)
    if out_dir is not None:
        save(params, Path(out_dir) / "params_latest.pkl")
    return make_policy, params, metrics, frames


def save(params, path, quiet=False):
    """정책 파라미터를 파일로. **콜랩에서 이것 없이 몇 시간 돌리지 말 것.**

    런타임이 끊기거나 모듈을 다시 읽으려고 재시작하면 메모리의 `params` 가
    통째로 날아간다. 실제로 한 번 당했다 -- `train.py` 에 `restore` 를 추가한 뒤
    콜랩이 캐시된 옛 모듈을 들고 있어서, 고친 코드를 쓰려면 재시작해야 했고
    재시작하면 1.5백만 스텝짜리 결과가 사라지는 상황이 됐다.

    orbax 를 안 쓰고 pickle 을 쓰는 이유 -- 이건 **우리가 만든 파라미터**라
    체크포인트 형식을 남과 맞출 필요가 없다. `llc/loader.py` 가 orbax 를 읽는 것은
    남이 준 체크포인트를 읽기 때문이다.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = {
        "params": jax.tree_util.tree_map(np.asarray, params),
        "signature": obs.SIGNATURE,
        "fields": obs._signature_fields(),
        "soft": obs._soft_fields(),
    }
    with open(path, "wb") as f:
        pickle.dump(blob, f)
    if not quiet:
        print(f"    파라미터 {path.name}  {path.stat().st_size / 1e6:.1f} MB"
              f"  서명 {obs.SIGNATURE}", flush=True)
    return path


def policy(params, task, *, deterministic=True):
    """저장된 파라미터 -> 굴릴 수 있는 정책. **손으로 조립하지 말 것.**

    `train` 이 돌려주는 `make_policy` 는 그 세션에만 있다. 나중에 pkl 만 들고
    영상을 뽑으려면 신경망을 같은 모양으로 다시 세워야 하는데, 거기서 틀리기 쉬운
    자리가 둘이다.

        전처리기   `running_statistics.normalize` 여야 한다. 빼먹으면 정규화 없이
                   돌아가고, 관측이 정규화 전 크기라 정책이 엉뚱하게 움직인다.
                   `std_eps` 는 저장된 통계 안에 들어 있으므로 따로 안 넘긴다
        관측 크기   `task.observation_size` 에서 받는다. 숫자를 적지 않는다

    쓰는 법

        p = train.load(경로)
        pol = train.policy(p, task)
        stage1.debug_video(task, pol, '평지.mp4')
    """
    from brax.training.acme import running_statistics

    net = _network_factory()(
        task.observation_size, task.action_size,
        preprocess_observations_fn=running_statistics.normalize)
    inner = ppo_networks.make_inference_fn(net)(params,
                                                deterministic=deterministic)
    return lambda o, k: inner(o, k)[0]


def load(path, *, strict=True):
    """`save` 가 쓴 파일을 되읽는다. `train(restore=...)` 에 그대로 넣는다.

    **배치 서명을 대조한다.** 어긋나면 읽지 않고 세운다 -- 크기가 같은 채로 뜻만
    바뀐 변경(`action.SCALE`, `maze.SPAN`, 스캔 격자 등)은 `restore` 가 성공해
    버려서, 로봇이 이상해질 뿐 아무도 원인을 못 찾는다. `obs.py` 의 서명 절 참고.

    `strict=False` 로 억지로 읽을 수 있다. 무엇이 다른지 알고 그래도 되는 경우만
    쓸 것 -- 예를 들어 보상 계수만 바꿨을 때다.

    서명이 없는 옛 파일도 읽는다. 2026-08-21 이전 것이고, 그때는 관측이 164 라
    지금(255)과 안 맞아 어차피 브랙스가 세운다.
    """
    with open(Path(path), "rb") as f:
        blob = pickle.load(f)

    if not (isinstance(blob, dict) and "params" in blob):
        print("  주의 — 서명 없는 옛 파라미터다. 대조를 건너뛴다.", flush=True)
        return blob

    diff = obs.check_signature(blob.get("signature", ""), blob.get("fields"))
    if diff:
        lines = "\n    ".join(diff)
        msg = (f"배치 서명이 다르다. 이 파라미터는 지금 코드와 안 맞는다.\n"
               f"    {lines}\n"
               f"  정말 읽으려면 load(..., strict=False).")
        if strict:
            raise ValueError(msg)
        print(f"  주의 — {msg}", flush=True)

    for line in obs.soft_diff(blob.get("soft")):
        print(f"  주의 — 무른 항목이 바뀌었다: {line}", flush=True)
        print("         미세조정 없이 그대로 쓰면 안 된다.", flush=True)

    return blob["params"]
