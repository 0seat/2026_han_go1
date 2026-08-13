"""11차원 명령 사양 — 이 파일이 단일 출처다.

근거의 우선순위 (WTW 이식에서 배운 것을 그대로 적용한다)

    1  체크포인트 `_METADATA`의 텐서 shape          절대
    2  `results/final_run_config.json`               절대 (이 가중치를 만든 config)
    3  학습 노트북 소스                              높음
    4  mujoco_playground 원본 소스                   높음 (env 골격)
    5  추론·유추                                     쓰지 말 것 -- 스윕으로 재고 나서 쓴다

아래 `RANGES`는 근거 2다. `SLOT_MEANING`은 **아직 근거 5**다.
`sweep.axis_scan()`이 실측으로 채워 넣기 전까지는 가설로 취급할 것.

관측 배치 (근거 1+3+4)

    state[56] = [ linvel 3 | gyro 3 | gravity 3 | qpos-default 12 | qvel 12
                | last_act 12 | command 11 ]
                 <------------------ 45 -----------------><- 45:56 ->
    privileged_state[131] = [ state 56 | ... 75 ]

    45+11 = 56, playground Go1 joystick privileged 123+8 = 131 -- 체크포인트와 일치.
    명령이 **맨 뒤**라는 것은 phase8 노트북의 `obs["state"].at[-4:].set(command)`가
    직접 보여준다. 11차원판도 같은 자리라는 것은 가설이며 `env.gate()`가 검사한다.

WTW와 다른 점 중 사고를 부르는 것 (`docs/01_llc.md`와 대조하며 읽을 것)

    - 명령 스케일 벡터가 **없다.** 물리 단위를 그대로 넣는다.
      WTW의 `COMMANDS_SCALE`에 해당하는 것이 이 스택에는 없다.
    - `height`가 **절대 높이 [0.22, 0.32] m**다. WTW는 기본자세 대비 상대값이었다.
    - `vx`에 **음수가 없다.** 후진 명령은 학습 분포 밖이다.
    - 관측 정규화가 **있다.** 체크포인트의 normalizer(params[0])를 반드시 통과시킨다.
      brax 추론 경로를 쓰면 자동이고, 직접 MLP를 굴리면 빠뜨린다.
"""

from __future__ import annotations

#: 명령 벡터의 순서. `final_run_config.json`의 `command_order` 그대로.
COMMAND_ORDER: tuple[str, ...] = (
    "vx",             # 0
    "vy",             # 1
    "yaw",            # 2
    "height",         # 3
    "pitch",          # 4
    "roll",           # 5
    "stance_width",   # 6
    "step_freq",      # 7
    "gait_phase",     # 8
    "gait_duration",  # 9
    "footswing",      # 10
)

DIM = len(COMMAND_ORDER)

#: 학습 시 샘플링 범위 (phase12 `command_ranges`). phase13/14도 이 범위를 이어받았다.
RANGES: dict[str, tuple[float, float]] = {
    "vx":            (0.0,  0.8),
    "vy":            (-0.15, 0.15),
    "yaw":           (-0.35, 0.35),
    "height":        (0.22, 0.32),
    "pitch":         (-0.3,  0.3),
    "roll":          (-0.2,  0.2),
    "stance_width":  (0.15, 0.25),
    "step_freq":     (1.5,  4.5),
    "gait_phase":    (0.0,  1.0),
    "gait_duration": (0.3,  0.7),
    "footswing":     (0.06, 0.15),
}

#: 스윕의 기준점. 나머지 차원을 여기 고정한 채 한 축씩 흔든다.
#: vx만 0이 아닌 이유 -- 정지 상태에서는 보행 관련 차원(step_freq, footswing,
#: gait_*)이 원리적으로 반응할 수 없어 "죽은 슬롯"과 구별되지 않는다.
BASELINE: dict[str, float] = {
    "vx":            0.40,
    "vy":            0.00,
    "yaw":           0.00,
    "height":        0.30,   # 노트북의 DEFAULT_BODY_HEIGHT_COMMAND
    "pitch":         0.00,
    "roll":          0.00,
    "stance_width":  0.20,
    "step_freq":     3.00,
    "gait_phase":    0.50,
    "gait_duration": 0.50,
    "footswing":     0.10,
}

#: 실측으로 확정한 학습 여부. 근거는 `sweep.axis_scan()` 2026-08-11
#: (phase14 체크포인트, 평지, 노이즈 0, 8 s, 11축 x 7점 x 3시드).
#:
#: 인터페이스에 슬롯이 있다 != 그 슬롯이 학습됐다. WTW에서도 같은 구조였고
#: (`docs/01_llc.md` 0.3절), 이 스택도 마찬가지다. 다만 성격이 다르다 --
#: WTW의 duty/roll은 학습 내내 **상수**만 관측해 채널이 죽은 것이고,
#: 여기 5개는 학습 중 **범위 전체에서 균등 샘플**됐지만(phase12 command_ranges)
#: 추종 보상이 없어서 정책이 무시하도록 수렴한 것이다.
#:
#: 이 차이가 실무에서 중요하다. 아직 학습되지 않았을 뿐 **분포 안에 있으므로**
#: 어떤 값을 넣어도 안전하다. WTW처럼 "건드리면 분포 이탈"이 아니다.
#: 나중 단계에서 보상을 걸어 재학습하면 그대로 열린다.
TRAINED: dict[str, bool] = {
    "vx": True, "vy": True, "yaw": True,
    "height": True, "pitch": True, "roll": True,
    # 아래 5개는 미학습. 명령을 흔들어도 응답이 없다.
    "stance_width": False, "step_freq": False,
    "gait_phase": False, "gait_duration": False, "footswing": False,
}

#: 학습된 축만. HLC 액션 스페이스는 이 6개 위에서 설계한다.
TRAINED_DIMS: tuple[str, ...] = tuple(n for n in COMMAND_ORDER if TRAINED[n])

#: 미학습 축을 고정할 값. 범위 중앙 -- 학습 중 균등 샘플된 구간의 한가운데다.
#: 응답이 없으므로 값 자체는 중요하지 않지만, 재학습으로 열렸을 때 기준이 되도록
#: 한 곳에 적어 둔다.
UNTRAINED_HOLD: dict[str, float] = {
    n: (RANGES[n][0] + RANGES[n][1]) / 2 for n in COMMAND_ORDER if not TRAINED[n]
}

#: 실측 추종 특성. 실측 = 이득 x 명령 + 편향. 근거는 위와 같은 스윕.
#: `(이득, 편향, R^2)`. 미학습 축은 이득이 0에 가까워 의미가 없어 싣지 않는다.
TRACKING: dict[str, tuple[float, float, float]] = {
    "vx":     (0.921, -0.0126, 0.998),
    "vy":     (0.589, +0.0123, 0.998),
    "yaw":    (0.774, -0.0038, 1.000),
    "height": (0.455, +0.1182, 0.996),
    "pitch":  (0.344, +0.0154, 0.995),
    "roll":   (0.415, +0.0014, 0.997),
}

#: 명령 범위를 `TRACKING`으로 통과시킨 **실제 도달 범위**.
#: 명령 범위와 혼동하지 말 것 -- HLC가 쓸 수 있는 실제 제어권한은 이쪽이다.
#: 예: pitch를 +0.3으로 명령해도 몸통은 +0.12 rad밖에 안 기운다.
def actual_range(name: str) -> tuple[float, float]:
    if name not in TRACKING:
        raise KeyError(f"{name}은 미학습 축이라 도달 범위가 없습니다.")
    gain, bias, _ = TRACKING[name]
    lo, hi = RANGES[name]
    a, b = gain * lo + bias, gain * hi + bias
    return (a, b) if a <= b else (b, a)


#: 명령과 무관하게 정책이 스스로 정한 보행 특성. 미학습 축을 흔들어도 안 바뀐다.
#: sync_diag 0.94 = 대각 두 발이 94% 같은 접촉 상태 -> **트로트로 고정**이다.
#: (sync_02 0.25 = 같은 쪽 앞뒤는 역위상. 전형적인 트로트 패턴.)
FIXED_GAIT = {
    "step_hz": 2.63,        # 명령 1.5~4.5 전 구간에서 2.53~2.68
    "duty": 0.66,           # 명령 0.3~0.7 전 구간에서 0.658~0.664
    "swing_m": 0.027,       # 명령 0.06~0.15 전 구간에서 0.0269~0.0273
    "stance_width_m": 0.202,  # 명령 0.15~0.25 전 구간에서 0.2016~0.2033
    "gait": "trot",
}

#: 가설이었던 슬롯 의미. 학습된 6개는 실측으로 확인됐고(1위 지표가 예상과 일치),
#: 미학습 5개는 확인할 방법이 없어 이름 그대로 둔다.
SLOT_MEANING: dict[str, str] = {
    "vx":            "몸통 좌표계 전후 속도 m/s (확인)",
    "vy":            "몸통 좌표계 좌우 속도 m/s (확인)",
    "yaw":           "요 각속도 rad/s (확인)",
    "height":        "지면 법선 기준 몸통 절대 높이 m (확인)",
    "pitch":         "몸통 피치 rad (확인)",
    "roll":          "몸통 롤 rad (확인)",
    "stance_width":  "좌우 발 간격 m (미학습)",
    "step_freq":     "발걸음 빈도 Hz (미학습)",
    "gait_phase":    "걸음 위상차 0~1 (미학습)",
    "gait_duration": "지지 비율 duty (미학습)",
    "footswing":     "발 들어올림 높이 m (미학습)",
}


#: `env.probe()`가 매 스텝 내는 원시 열. 후처리가 인덱스를 하드코딩하지 않게 한다.
PROBE_COLUMNS: tuple[str, ...] = (
    "vx_local", "vy_local", "vz_local",
    "wx", "wy", "wz",
    "pitch", "roll", "torso_z",
    "contact0", "contact1", "contact2", "contact3",
    "footz0", "footz1", "footz2", "footz3",
    "footy0", "footy1", "footy2", "footy3",
    "done", "reward",
)

#: 원시 열에서 뽑아낸 롤아웃 단위 지표. 슬롯의 생사를 이 목록 위에서 판정한다.
RESPONSE_METRICS: tuple[str, ...] = (
    "vx", "vy", "yaw_rate", "pitch", "roll", "torso_z",
    "duty", "step_hz", "swing", "stance_w",
    "sync_01", "sync_02", "sync_diag",
)

#: 슬롯 이름 -> 그 슬롯이 조작할 것으로 기대되는 지표. 검증 대상이지 전제가 아니다.
#: 1위로 올라온 지표가 이것과 다르면 `SLOT_MEANING`이 틀린 것이다 -- WTW에서
#: `gait_offset`이 bound를, `gait_bound`가 pace를 만든 전례가 있다.
EXPECTED_METRIC: dict[str, str] = {
    "vx": "vx", "vy": "vy", "yaw": "yaw_rate",
    "height": "torso_z", "pitch": "pitch", "roll": "roll",
    "stance_width": "stance_w", "step_freq": "step_hz",
    "gait_duration": "duty", "footswing": "swing",
    "gait_phase": "sync_diag",
}


def baseline_vector() -> list[float]:
    """`BASELINE`을 `COMMAND_ORDER` 순서의 리스트로."""
    return [BASELINE[name] for name in COMMAND_ORDER]


def expand(action: dict[str, float] | list[float]) -> list[float]:
    """학습된 6축 -> 전체 11축 명령 벡터.

    HLC는 6차원만 낸다. 나머지 5개는 `UNTRAINED_HOLD`로 채운다.
    미학습 축을 HLC 액션에 열어두면 정책이 아무 영향 없는 축을 탐색하느라
    표본을 낭비한다 -- 열려면 LLC를 먼저 재학습해야 한다.
    """
    if not isinstance(action, dict):
        if len(action) != len(TRAINED_DIMS):
            raise ValueError(
                f"학습된 축은 {len(TRAINED_DIMS)}개입니다: {TRAINED_DIMS}"
            )
        action = dict(zip(TRAINED_DIMS, action))
    unknown = set(action) - set(TRAINED_DIMS)
    if unknown:
        raise KeyError(f"미학습 축은 명령할 수 없습니다: {sorted(unknown)}")
    merged = {**UNTRAINED_HOLD, **{n: BASELINE[n] for n in TRAINED_DIMS}, **action}
    return [merged[n] for n in COMMAND_ORDER]


#: `expand`의 jit 안 판. HLC 학습은 brax PPO가 환경을 jit + vmap으로 돌리는데
#: `expand`는 dict를 받고 KeyError를 던지고 파이썬 리스트를 내므로 그 안에서
#: 못 쓴다. 그렇다고 이 파일에 jax를 들이면 selftest가 로컬에서 못 돈다.
#: 그래서 **숫자만 내주고 조립은 부르는 쪽이 한 줄로** 한다.
#:
#:     import jax.numpy as jnp
#:     base = jnp.asarray(spec.BASE_VECTOR)
#:     command = base.at[jnp.asarray(spec.TRAINED_INDEX)].set(action6)
#:
#: 숫자는 여전히 이 파일에서만 나온다. 옮겨 적는 곳이 없다.
BASE_VECTOR: tuple[float, ...] = tuple(
    BASELINE[n] if TRAINED[n] else UNTRAINED_HOLD[n] for n in COMMAND_ORDER
)

#: 학습된 축이 11칸 중 몇 번째인가. `TRAINED_DIMS`와 순서가 같다.
TRAINED_INDEX: tuple[int, ...] = tuple(COMMAND_ORDER.index(n) for n in TRAINED_DIMS)


def index(name: str) -> int:
    """명령 이름 -> 슬롯 번호. 인덱스를 하드코딩하지 말고 이것을 쓸 것."""
    return COMMAND_ORDER.index(name)


def sweep_values(name: str, n: int = 7) -> list[float]:
    """한 축의 스윕 점들. 범위 전체를 균등 분할한다.

    n의 기본값이 7인 이유 -- 3점으로 추세를 읽었다가 틀린 적이 있다
    (`wtw_nav/configs/default.py`의 pitch 주석). 평탄역과 단조 증가를 구별하려면
    최소 5점이 필요하고, 비단조까지 보려면 7점이다.
    """
    lo, hi = RANGES[name]
    if n < 2:
        raise ValueError(f"스윕 점이 {n}개면 기울기를 낼 수 없습니다.")
    step = (hi - lo) / (n - 1)
    return [lo + step * i for i in range(n)]
