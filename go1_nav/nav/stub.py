"""더미 nav — 현재 위치와 목표를 잇는 직선.

nav 담당은 **이 파일의 안만** 바꾼다. `path`의 이름과 입출력은 그대로 두고
직선을 D*로 갈아끼운다. 그러면 나머지 전부가 그대로 돈다.

지도를 안 쓰는데도 인자로 받는 이유 -- 나중에 쓸 것이기 때문이다. 지금 빼면
진짜 nav를 넣을 때 부르는 쪽을 전부 고쳐야 한다.

내는 점 개수는 **고정**이어야 한다. `common/path.py`가 jit 안에서 도는데
배열 크기가 바뀌면 매번 다시 컴파일된다. 진짜 nav도 점 개수를 정해두고
모자라면 마지막 점을 반복해 채운다.
"""

from __future__ import annotations

import jax.numpy as jnp


def path(world, robot_xy, goal_xy):
    """지도 좌표 점 목록 (N, 2). 첫 점이 현재 위치, 마지막 점이 목표.

    더미 구현은 두 점짜리 직선이다. 벽도 지형도 보지 않는다.
    """
    del world
    return jnp.stack([jnp.asarray(robot_xy, dtype=jnp.float32).reshape(2),
                      jnp.asarray(goal_xy, dtype=jnp.float32).reshape(2)])
