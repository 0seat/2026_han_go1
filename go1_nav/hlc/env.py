"""11차원 명령 Go1 환경 — playground `Go1JoystickFlatTerrain`의 최소 파생.

설계 원칙: **`_get_obs`도 `_get_reward`도 건드리지 않는다.**

playground의 Go1 joystick은 관측을 이렇게 조립한다 (명령이 맨 뒤).

    state = jp.hstack([linvel, gyro, gravity, qpos-default, qvel, last_act,
                       info["command"]])

즉 `info["command"]`의 길이만 3에서 11로 바꾸면 관측이 48에서 56으로 **저절로**
늘어난다. `privileged_state`도 `state`를 앞에 물고 있으므로 123에서 131이 된다.
체크포인트의 실제 shape가 정확히 56/131이므로, phase12가 한 일도 이것일 가능성이
매우 높다. 관측 조립 코드를 베껴 쓰지 않는 편이 훨씬 안전하다 -- WTW 이식에서
관측 순서를 손으로 옮기다 DOF 순서를 틀려 며칠을 날렸다.

이 가설이 틀렸다면 `gate()`가 첫 줄에서 잡는다. 통과하지 못하면 스윕을 돌리지 말 것.

주의 사항 두 가지

    1. PD를 거는 자리가 WTW 이식과 반대다. playground는 Kd를 `dof_damping[6:]`에
       넣고 `actuator_biasprm[:, 2]`는 건드리지 않는다. `docs/01_llc.md` 8.2절은
       정반대로 적혀 있으니 습관대로 옮기면 감쇠가 이중으로 걸린다.
       이 모듈은 playground의 기본 모델을 그대로 쓰므로 문제되지 않지만,
       나중에 지형을 붙일 때 다시 만난다.

    2. 부모는 `steps_until_next_cmd`가 0이 되면 명령을 스스로 재샘플한다.
       스윕은 명령을 고정해야 하므로 `hold()`가 매 스텝 되돌린다.
"""

from __future__ import annotations

import jax
import jax.numpy as jp
import numpy as np

from ..llc import spec


def _import_playground():
    """무거운 import를 함수 안으로. 로컬(JAX 없음)에서 spec만 읽을 수 있게."""
    from mujoco_playground import registry
    from mujoco_playground._src.locomotion.go1 import go1_constants as go1_consts
    from mujoco_playground._src.locomotion.go1.joystick import Joystick
    return registry, go1_consts, Joystick


def make(noise_level: float = 0.0, command_dim: int = spec.DIM):
    """11차원 명령 env를 만든다.

    noise_level=0.0이 기본인 이유 -- 스윕은 명령 하나당 응답 하나를 재는 것이고,
    관측 노이즈는 그 응답에 분산만 더한다. 시드 간 편차를 노이즈가 아니라 정책의
    상태 의존성으로 읽으려면 여기서는 꺼야 한다. 노이즈 강건성은 별도 측정이다.
    """
    registry, go1_consts, Joystick = _import_playground()

    class Go1Command11(Joystick):
        """명령 슬롯만 3 -> command_dim으로 늘린 joystick."""

        def __init__(self, config):
            self._command_dim = int(command_dim)
            self._default_command = jp.asarray(
                spec.baseline_vector()[:self._command_dim], dtype=jp.float32
            )
            super().__init__(task="flat_terrain", config=config)

            # 접촉 센서 id. 부모 버전에 따라 이름이 달라 직접 만든다.
            self._floor_found_sensor = [
                self._mj_model.sensor(f"{geom}_floor_found").id
                for geom in go1_consts.FEET_GEOMS
            ]
            self._floor_found_adr = np.asarray(
                [self._mj_model.sensor_adr[i] for i in self._floor_found_sensor]
            )

        # 부모의 자동 재샘플을 무력화.
        # playground 0.2.0의 시그니처는 `sample_command(rng, x_k)`이고 본체가
        # `x_k - w*(x_k - y*z)` (y,z,w는 shape (3,))라 11차원 x_k를 넣으면
        # 브로드캐스트에서 터진다. 그대로 돌려주는 것이 유일하게 안전하다.
        def sample_command(self, rng, *args, **kwargs):
            del rng, kwargs
            if args and args[0] is not None:
                return args[0]
            return self._default_command

        def reset(self, rng):
            """부모 reset 뒤 명령을 11차원으로 승격한다.

            playground의 `reset`은 `sample_command`를 부르지 않고 `shape=(3,)`로
            명령을 직접 만든다(joystick.py:208). 그래서 `sample_command`만
            덮어써서는 관측이 48D로 나온다 -- 실제로 여기서 한 번 걸렸다.
            """
            return self.with_command(super().reset(rng), self._default_command)

        # ---------- 명령 주입 ----------
        def with_command(self, state, command):
            """명령을 바꾸고 **관측을 다시 조립한다.**

            관측 꼬리를 `.at[-11:].set()`으로 덮어쓰는 방법도 있지만 쓰지 않는다.
            그러려면 "명령이 꼬리에 있다"와 "privileged 앞부분이 state와 같다"를
            둘 다 참으로 가정해야 하는데, 여기서는 부모의 `_get_obs`를 그대로
            불러 **가정 없이** 만든다. noise_level=0에서는 재조립이 원본과 같다.
            """
            info = dict(state.info)
            info["command"] = jp.asarray(command, dtype=state.data.qpos.dtype)
            info["steps_until_next_cmd"] = jp.asarray(1_000_000_000, dtype=jp.int32)
            return state.replace(info=info, obs=self._get_obs(state.data, info))

        def hold(self, state, command):
            """스텝 뒤에 명령을 되돌린다. `with_command`와 같지만 의도가 다르다."""
            return self.with_command(state, command)

        # ---------- 측정 ----------
        def probe(self, state):
            """한 스텝의 관측 가능량. 스칼라만 모아 (23,) 벡터로 낸다.

            배열을 그대로 쌓으면 스캔 출력이 수백 MB가 된다. 여기서 압축한다.
            """
            d = state.data
            linvel = self.get_local_linvel(d)          # 몸통 좌표계 3
            gyro = self.get_gyro(d)                    # 몸통 좌표계 3
            gravity = self.get_gravity(d)              # 몸통 좌표계 중력 단위벡터 3

            # 몸통 좌표계 중력에서 피치·롤. 요에 불변이라 이 경로를 쓴다.
            #   g_b = [sin(pitch), -cos(p)sin(r), -cos(p)cos(r)]
            pitch = jp.arcsin(jp.clip(gravity[0], -1.0, 1.0))
            roll = jp.arctan2(-gravity[1], -gravity[2])

            contact = (d.sensordata[self._floor_found_adr] > 0).astype(jp.float32)

            feet_w = d.site_xpos[self._feet_site_id]           # (4,3) 월드
            rot = d.xmat[self._torso_body_id].reshape(3, 3)
            feet_b = (feet_w - d.qpos[0:3]) @ rot              # 몸통 좌표계

            return jp.concatenate([
                linvel,                 # 0:3
                gyro,                   # 3:6
                jp.array([pitch, roll, d.qpos[2]]),   # 6:9
                contact,                # 9:13
                feet_w[:, 2],           # 13:17  발 높이 (평지 floor z=0)
                feet_b[:, 1],           # 17:21  발 좌우 위치
                jp.array([state.done, state.reward]),  # 21:23
            ])

    config = registry.get_default_config("Go1JoystickFlatTerrain")
    config.impl = "jax"
    config.noise_config.level = float(noise_level)
    if hasattr(config, "pert_config"):
        config.pert_config.enable = False   # 외란은 스윕의 신호를 가린다

    return Go1Command11(config=config)


#: `probe()` 출력의 열 이름. 정의는 `spec`이 소유한다 (JAX 없이 읽을 수 있어야 한다).
PROBE_COLUMNS = spec.PROBE_COLUMNS


def gate(env, verbose: bool = True) -> bool:
    """스윕을 돌리기 전에 통과해야 하는 구조 검사.

    여기서 걸리면 `spec.py`의 관측 배치 가설이 틀린 것이다. 스윕을 강행하면
    "정책이 명령에 반응하지 않는다"는 결론이 나오는데, 그것은 슬롯이 죽은 게
    아니라 우리가 엉뚱한 자리에 넣고 있다는 뜻이다. 구별이 불가능하므로 멈춘다.
    """
    state = jax.jit(env.reset)(jax.random.PRNGKey(0))
    obs = state.obs
    checks: list[tuple[str, bool, str]] = []

    s = tuple(obs["state"].shape)
    p = tuple(obs["privileged_state"].shape)
    checks.append(("state 56D", s == (56,), f"{s}"))
    checks.append(("privileged_state 131D", p == (131,), f"{p}"))
    checks.append(("command 11D", tuple(state.info["command"].shape) == (spec.DIM,),
                   f"{tuple(state.info['command'].shape)}"))
    checks.append(("action 12D", env.action_size == 12, f"{env.action_size}"))

    # 명령이 관측 꼬리에 있는가 -- 슬롯을 하나씩 흔들어 직접 확인한다.
    probe_cmd = jp.asarray(spec.baseline_vector()) + 1.0
    moved = jax.jit(env.with_command)(state, probe_cmd)
    delta = np.asarray(moved.obs["state"] - obs["state"])
    changed = np.flatnonzero(np.abs(delta) > 1e-6)
    tail = set(range(56 - spec.DIM, 56))
    checks.append(("명령이 state[45:56]에만 실린다", set(changed.tolist()) <= tail,
                   f"바뀐 인덱스 {changed.tolist()}"))

    mirror = np.allclose(np.asarray(obs["privileged_state"][:56]),
                         np.asarray(obs["state"]), atol=1e-6)
    checks.append(("privileged[:56] == state", mirror, "미러 아님" if not mirror else ""))

    dt = float(env.dt)
    checks.append(("제어 주기 0.02 s (50 Hz)", abs(dt - 0.02) < 1e-9, f"{dt}"))

    # probe가 도는지. 여기서 걸리면 playground 버전이 달라 접근자 이름이 바뀐 것이다.
    try:
        p = jax.jit(env.probe)(state)
        okp = tuple(p.shape) == (len(spec.PROBE_COLUMNS),)
        checks.append((f"probe {len(spec.PROBE_COLUMNS)}D", okp, f"{tuple(p.shape)}"))
    except Exception as exc:
        checks.append(("probe 실행", False, f"{type(exc).__name__}: {exc}"))

    ok = all(c[1] for c in checks)
    if verbose:
        print("=" * 66)
        print("구조 게이트")
        for name, passed, detail in checks:
            mark = "통과" if passed else "실패"
            print(f"  [{mark}] {name}" + (f"   {detail}" if detail else ""))
        print("=" * 66)
        if not ok:
            print("게이트 미통과. spec.py의 관측 배치 가설이 틀렸습니다.")
            print("스윕을 돌리면 '슬롯이 죽었다'와 '자리를 잘못 짚었다'를 구별할 수 없습니다.")
    return ok
