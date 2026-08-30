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
    - `vx`에 **음수가 있다.** phase18부터 후진이 열렸다. 2026-08-15 이전에 쓰인
      문서와 주석은 "후진은 학습 분포 밖"이라고 적고 있으니 그쪽을 믿지 말 것.
    - 관측 정규화가 **있다.** 체크포인트의 normalizer(params[0])를 반드시 통과시킨다.
      brax 추론 경로를 쓰면 자동이고, 직접 MLP를 굴리면 빠뜨린다.

동결 (2026-08-17)

    이 파일의 **순서와 단위**를 얼린다. 값은 실측으로 바뀌지만 형식은 안 바뀐다.

        얼린 것     11축의 순서 (`COMMAND_ORDER`)
                    물리 단위. 정규화하지 않는다
                    HLC는 11축을 다 내고 미학습 축은 마스크로 닫는다

        안 얼린 것   `RANGES`의 수치      LLC 능력이 늘면 넓힌다
                    `TRAINED`의 참거짓   보상이 붙으면 열린다
                    `TRACKING`의 이득    체크포인트마다 다시 잰다

    얼리는 이유는 **LLC가 계속 갱신되기 때문이다.** 형식이 고정이면 LLC 변경이
    HLC 입장에서 동역학 섭동이라 미세조정으로 흡수된다. 형식이 바뀌면 처음부터다.

    정규화 대신 물리 단위를 쓰는 결정이 여기서 값을 했다 -- phase18에서 vx
    상한이 0.8에서 넓어졌는데, `[-1,1]` 정규화였다면 같은 출력의 의미가 바뀌어
    기존 정책이 통째로 어긋난다. 물리 단위면 범위 확장은 선택지가 늘어나는
    것뿐이라 옛 정책이 그대로 유효하다.

    **HLC 정책의 출력 폭은 `DIM`(11)으로 둔다.** `len(TRAINED_DIMS)`로 두면 축이
    열릴 때마다 출력층 모양이 바뀌어 재학습이 된다. 11로 두고 `TRAINED_INDEX`
    밖의 축을 샘플링에서 빼면(로그확률에 안 넣는다) 축 열기가 마스크 한 비트다.
"""

from __future__ import annotations

#: 이 사양이 기술하는 체크포인트. 값을 갱신할 때 여기부터 고친다.
SOURCE = "phase18_speed_fwd/20260815_115644"

#: 계보. 드라이브에서 실제로 확인한 것만 적는다.
#:
#:     phase11 -> phase12 (11축 도입) -> phase13 (추종 보상) -> phase14 (heightfield)
#:            -> ??? -> phase_reverse_v2 (후진) -> phase18 (속도 확장)
#:
#: phase15~17과 phase_reverse_v2가 드라이브에 없다. 그래서 phase18이 phase14의
#: heightfield 능력을 물려받았는지 **문서로는 확인할 수 없다.**
#:
#: 다만 **실측으로는 확인됐다.** phase18을 LLC로 쓴 HLC가 8x16 미로(경사 · 턱 ·
#: 돌 · 외나무다리 · 터널)에서 차선별 도달률 0.948을 냈다 (2026-08-27,
#: `hlc6/02_목표고침`). 지형 위 보행은 된다. 물려받았다는 문서가 없을 뿐이다.

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

#: 명령 범위. `vx`만 phase18 기준이고 나머지 열 축은 phase12 `command_ranges` 그대로다.
#:
#: vx를 (-1.0, 1.0)으로 두는 근거와 그 한계
#:
#:     상한   phase18 config의 목표는 1.5지만 stage7(1.5)이 `FORCED`로 끝났다.
#:            quick_check가 fwd_0.9 488/500, fwd_1.2 366/500, fwd_1.5 299/500이다.
#:            488 이상 살아남는 구간이 0.9까지라 1.0에서 자른다.
#:     하한   quick_check의 reverse_1.0이 434/500이다. -1.0까지는 있다.
#:            **정확한 학습 하한은 모른다** -- 부모인 phase_reverse_v2가 드라이브에
#:            없고 phase18 config에 `command_ranges`가 안 실려 있다.
#:
#: **주의 — 후진 -1.0은 여유가 없다.** 434/500 = 86.8%로, 전진 0.9의 97.6%보다
#: 한참 낮다. reverse_0.5는 500/500이다. 그런데 HLC가 실제로 후진을 쓴다 --
#: 2026-08-21 터널 판에서 정책이 입구에서 물러섰다 다시 진입하는 것을 영상으로
#: 확인했다. 즉 **13%로 넘어지는 동작에 기대고 있다.**
#:
#: 좁히지 않고 두기로 했다(사용자 결정, 2026-08-21). 후진 중 넘어짐이 늘면
#: 이 숫자를 제일 먼저 볼 것.
#:
#: 상한을 1.5로 올리자는 이야기가 나왔을 때 거절한 근거도 같은 표다.
#: fwd_1.5가 299/500(59.8%)이고 stage7이 28M 스텝을 쓰고도 FORCED로 끝났다.
#: 게다가 config의 `yaw_drift_factor = 1.0 + 4.0*clip(speed/1.5,0,1)`이라
#: **고속에서 직진하도록 5배로 압박받으며 학습했다.** HLC는 반대로 움직이면서
#: 계속 돌아야 하므로, 고속 + 회전은 학습 분포 밖이다.
#:
#: 좁게 잡아도 손해가 없다. 물리 단위라 나중에 넓히는 것은 숫자 하나이고,
#: 그때 옛 정책은 그대로 유효하다.
RANGES: dict[str, tuple[float, float]] = {
    "vx":            (-1.0, 1.0),
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
#: **주의 — footswing은 2026-08-17 기준 LLC 담당이 재학습 중이다.** 열리면 여기를
#: True로 바꾸는 것만으로 `TRAINED_DIMS` · `BASE_VECTOR` · `TRAINED_INDEX`가 따라온다.
#: 열렸다고 판단하는 근거는 스윕이지 학습이 돌았다는 사실이 아니다.
#:
#: phase18의 `reward_parameters`는 phase14와 **완전히 같다.** 새 추종 보상이 하나도
#: 안 붙었으므로 phase18에서 이 표가 달라졌을 근거가 없다. 그래서 갱신하지 않는다.
#: 한 가지 단서 -- `footswing_tracking_scale`은 있는데 `footswing_tracking_sigma`가
#: 없다. pitch · roll · stance_width는 전부 sigma가 짝으로 있다. 배선 누락이면
#: footswing이 phase13부터 안 살아난 이유가 이것일 수 있다.
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

#: 실측 추종 특성. 실측 = 이득 x 명령 + 편향. `(이득, 편향, R^2)`.
#: 미학습 축은 이득이 0에 가까워 의미가 없어 싣지 않는다.
#:
#: **주의 — 이 표는 phase14에서 잰 것이고 phase18로 갱신되지 않았다.**
#: 특히 `vx`는 명령 0.0~0.8 구간에서만 맞춘 직선인데 지금 범위는 -1.0~1.0이라
#: 양 끝이 외삽이다. 후진은 애초에 존재하지도 않던 구간이다.
#: `sweep.axis_scan()`을 phase18로 다시 돌려 채운다. 그 전까지 이 값은 참고용이다.
#:
#: **주의 — `sweep` 모듈은 이 저장소에 없다.** 여러 주석이 참조하지만 찾으면
#: 없으므로, 다시 재려면 먼저 그것부터 가져오거나 새로 써야 한다.
TRACKING: dict[str, tuple[float, float, float]] = {
    "vx":     (0.921, -0.0126, 0.998),
    "vy":     (0.589, +0.0123, 0.998),
    "yaw":    (0.774, -0.0038, 1.000),
    "height": (0.455, +0.1182, 0.996),
    "pitch":  (0.344, +0.0154, 0.995),
    "roll":   (0.415, +0.0014, 0.997),
}

#: `TRACKING`의 직선을 실제로 맞춘 명령 구간. 이 밖은 외삽이다.
#: `RANGES`가 이 구간을 벗어나면 `actual_range`가 거부한다 -- 근거 5로 계산한
#: 숫자가 근거 2처럼 보이는 것을 막기 위해서다.
TRACKING_MEASURED_OVER: dict[str, tuple[float, float]] = {
    "vx":     (0.0,   0.8),
    "vy":     (-0.15, 0.15),
    "yaw":    (-0.35, 0.35),
    "height": (0.22,  0.32),
    "pitch":  (-0.3,  0.3),
    "roll":   (-0.2,  0.2),
}


#: 명령 범위를 `TRACKING`으로 통과시킨 **실제 도달 범위**.
#: 명령 범위와 혼동하지 말 것 -- HLC가 쓸 수 있는 실제 제어권한은 이쪽이다.
#: 예: pitch를 +0.3으로 명령해도 몸통은 +0.12 rad밖에 안 기운다.
def actual_range(name: str) -> tuple[float, float]:
    """명령 범위 -> 실제 도달 범위. 측정 구간 밖이면 거부한다.

    거부하는 이유 -- phase18에서 `vx` 범위가 -1.0~1.0으로 넓어졌는데 이득을 잰
    구간은 0.0~0.8이다. 직선을 그대로 늘리면 "후진 1.0을 명령하면 -0.93 m/s가
    나온다"는 숫자가 나오는데, 그 구간은 잰 적이 없다. 조용히 틀린 숫자를 내느니
    멈추는 편이 낫다. 스윕으로 `TRACKING`을 갱신하면 풀린다.
    """
    if name not in TRACKING:
        raise KeyError(f"{name}은 미학습 축이라 도달 범위가 없습니다.")
    lo, hi = RANGES[name]
    mlo, mhi = TRACKING_MEASURED_OVER[name]
    if lo < mlo or hi > mhi:
        raise ValueError(
            f"{name}의 명령 범위 ({lo}, {hi})가 이득을 잰 구간 ({mlo}, {mhi})을 "
            f"벗어납니다. {SOURCE}로 sweep.axis_scan()을 다시 돌려 TRACKING과 "
            f"TRACKING_MEASURED_OVER를 갱신할 것."
        )
    gain, bias, _ = TRACKING[name]
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
