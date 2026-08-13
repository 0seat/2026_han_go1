"""더미 HLC — 경로를 무시하고 앞으로만 간다.

HLC 담당은 **이 파일의 안만** 바꾼다. `act`의 이름과 입출력은 그대로 두고
고정값을 학습된 정책으로 갈아끼운다.

난수가 아니라 고정 전진인 이유 -- 영상에서 배선이 맞는지 눈으로 보이기
때문이다. 로봇이 앞으로 가면 명령이 통한 것이다. 난수는 봐도 모른다.

내는 것은 **학습된 축의 값만** `spec.TRAINED_DIMS` 순서로 담은 (6,)이다.
나머지 5축은 부르는 쪽이 `spec.BASE_VECTOR`로 채운다. 미학습 축을 열어두면
정책이 아무 영향 없는 축을 탐색하느라 표본을 낭비한다.
"""

from __future__ import annotations

import jax.numpy as jnp

from ..llc import spec

#: 학습된 6축의 고정값. vx만 올리고 나머지는 기준값 그대로.
FORWARD = jnp.asarray(
    [0.4 if n == "vx" else spec.BASELINE[n] for n in spec.TRAINED_DIMS],
    dtype=jnp.float32,
)


def act(features):
    """경로 특징 (path.SIZE,) -> 학습된 명령 축 (len(spec.TRAINED_DIMS),).

    값은 `spec.RANGES` 안이어야 한다. 더미 구현은 인자를 보지 않으므로
    목표에 도착하지 않는 것이 정상이다.
    """
    del features
    return FORWARD
