"""HLC 행동 -> 11축 명령. **범위가 넓어져도 뜻이 안 바뀌게** 만든다.

정책 신경망의 원시 출력은 `[-1, 1]`이다(brax의 `NormalTanhDistribution`). 그것을
명령으로 바꾸는 방법이 두 가지인데, 하나는 나중에 재학습을 부른다.

    나쁜 것   command = lerp(RANGES[축], (raw + 1) / 2)
              vx 범위가 (0, 0.8) -> (-1, 1) 로 넓어지는 순간 **같은 출력의 뜻이
              바뀐다.** 학습된 정책이 통째로 어긋난다. phase18 에서 실제로
              범위가 넓어졌으니 가정이 아니다.

    쓰는 것   command = clip(BASELINE + raw * SCALE, RANGES)
              `SCALE`은 `RANGES`와 무관한 고정 상수다. 범위가 넓어지면 도달할
              수 있는 영역만 늘고, 옛 정책의 출력은 뜻이 그대로다.

`llc/spec.py`의 동결 절에 적은 "물리 단위로 낸다"가 이것이다.

출력 폭은 항상 `spec.DIM`(11)이다. 미학습 축은 `TRAINED` 마스크로 닫혀 있고,
닫힌 축의 원시 출력은 **버린다.** 열리는 날 마스크만 바꾸면 되고 신경망 모양은
그대로다.
"""

from __future__ import annotations

import jax.numpy as jnp

from ..llc import spec

#: 축마다 원시 출력 1.0 이 몇 단위인가. **`RANGES`에서 유도하지 않는다.**
#:
#: 값의 근거 -- 각 축이 한 번에 낼 수 있는 변화폭을 물리적으로 잡았다.
#: 실측 도달 범위(`spec.TRACKING`)보다 넉넉히 두고 클립에 맡긴다. 좁게 잡으면
#: 정책이 포화 근처에서만 놀고, 넓게 잡으면 탐색이 거칠어지는데 후자가 낫다.
SCALE: dict[str, float] = {
    "vx":     1.0,      # +-1.0 m/s.  phase18 의 검증 구간 전체
    "vy":     0.15,     # 실측 이득 0.589 라 실제로는 +-0.09 m/s
    "yaw":    0.35,     # 실측 이득 0.774
    "height": 0.05,     # BASELINE 0.30 에서 +-0.05 -> 범위 (0.22, 0.32) 안
    "pitch":  0.30,
    "roll":   0.20,
    # 미학습 축은 마스크가 닫혀 있어 안 쓰이지만, 열릴 때를 대비해 적어 둔다.
    "stance_width": 0.05,
    "step_freq":    1.5,
    "gait_phase":   0.5,
    "gait_duration": 0.2,
    "footswing":    0.045,
}

#: 원시 출력 0 이 뜻하는 명령. **`spec.BASELINE`을 그대로 쓰지 않는다.**
#:
#: `BASELINE`은 축 스윕용이고 `vx = 0.40`이다. 정지 상태에서는 보행 관련 축이
#: 원리적으로 반응할 수 없어 죽은 슬롯과 구별이 안 되기 때문이다. 그런데 HLC의
#: 중심으로 쓰면 `vx` 가 -0.6 ~ +1.0 으로 비대칭이 되어 후진이 덜 탐색된다.
#:
#: 그래서 **움직임 축은 0, 자세 축은 중립**을 중심으로 둔다. 원시 출력 0 = 제자리.
#: 미학습 축은 `spec.UNTRAINED_HOLD` 그대로다 -- 마스크가 닫혀 있어 어차피 그 값이
#: 남지만, 열렸을 때 기준이 되도록 여기 적는다.
CENTRE: dict[str, float] = {
    "vx": 0.0, "vy": 0.0, "yaw": 0.0,
    "height": 0.30,          # 기본 자세. 실측 도달 0.455*0.30+0.118 = 0.255 m
    "pitch": 0.0, "roll": 0.0,
    **spec.UNTRAINED_HOLD,
}

_BASE = jnp.asarray([CENTRE[n] for n in spec.COMMAND_ORDER], dtype=jnp.float32)
_SCALE = jnp.asarray([SCALE[n] for n in spec.COMMAND_ORDER], dtype=jnp.float32)
_LO = jnp.asarray([spec.RANGES[n][0] for n in spec.COMMAND_ORDER], dtype=jnp.float32)
_HI = jnp.asarray([spec.RANGES[n][1] for n in spec.COMMAND_ORDER], dtype=jnp.float32)
#: 학습된 축만 1. 닫힌 축의 원시 출력은 여기서 0 이 되어 BASELINE 이 남는다.
_MASK = jnp.asarray([1.0 if spec.TRAINED[n] else 0.0
                     for n in spec.COMMAND_ORDER], dtype=jnp.float32)

#: HLC 정책의 출력 폭. 마스크가 닫혀 있어도 11을 유지한다.
SIZE = spec.DIM


def to_command(raw) -> jnp.ndarray:
    """정책 원시 출력 (SIZE,) -> 11축 명령. jit 안에서 돈다."""
    raw = jnp.asarray(raw, jnp.float32).reshape(SIZE)
    return jnp.clip(_BASE + raw * _SCALE * _MASK, _LO, _HI)


#: 이득 · 편향 무작위화를 적용할 축. **속도 축만이다.**
#:
#: 이 무작위화가 흉내내려는 것은 "LLC 가 속도 명령을 얼마나 따르는가"이고, 그건
#: `spec.TRACKING` 의 이득이 재는 것과 같다. 그런데 명령 전체에 곱하면
#: **절대값 축까지 흔들린다** -- height 0.300 -> 0.290, step_freq 3.0 -> 2.69,
#: footswing 0.105 -> 0.117. 앞은 기본 자세가 바뀌는 것이고, 뒤 둘은 미학습 축을
#: `UNTRAINED_HOLD` 에서 밀어내는 것이라 LLC 의 학습 분포를 벗어난다.
#:
#: 그래서 vx · vy · yaw 에만 건다. pitch · roll 은 절대 각도라 뺀다.
PERTURB = ("vx", "vy", "yaw")

_PERTURB = jnp.asarray([1.0 if n in PERTURB else 0.0
                        for n in spec.COMMAND_ORDER], dtype=jnp.float32)


def perturb(command, gain, bias) -> jnp.ndarray:
    """LLC 에 **실제로 보낼** 명령. 속도 축만 이득 · 편향으로 흔든다.

    관측에 실리는 것은 흔들기 **전** 값이다 -- 실기에서 HLC 는 자기가 뭘
    시켰는지만 알지, LLC 가 그것을 얼마나 따르는지는 모른다.
    """
    command = jnp.asarray(command, jnp.float32).reshape(SIZE)
    scaled = command * gain + bias
    return jnp.clip(jnp.where(_PERTURB > 0, scaled, command), _LO, _HI)

