"""경로를 로봇이 볼 숫자로 바꾼다 — nav와 HLC가 **둘 다 이 함수를 부른다.**

이 파일이 이 프로젝트에서 유일하게 두 사람이 공유하는 형식이다.
nav가 낸 것을 여기 넣고, HLC가 여기서 나온 것을 받는다. 양쪽이 각자 변환을
만들면 어긋나고, 어긋나도 크기는 같아서 아무도 모른 채 학습이 망한다.
그래서 변환을 한 곳에만 둔다.

형식
    입력   지도 좌표 점 목록 (N, 2)  +  로봇 위치 (2,)  +  로봇 요각 (rad)
    출력   (9,) = 3점 x [방향x, 방향y, 압축거리]

**jax로 쓴다.** HLC 학습은 brax PPO가 환경을 jit + vmap 안에서 돌리는데,
그 안에 numpy가 있으면 추적된 배열이 들어가면서 터진다. jax.numpy는 jit
바깥에서도 그냥 돌아가므로 nav 쪽에서 쓰는 데도 문제가 없다.

여기서 제약이 하나 나온다 -- **경로 점 개수가 고정이어야 한다.** jit은 배열
크기가 안 바뀌어야 돈다. nav가 낼 점 개수를 정해두고 모자라면 마지막 점을
반복해 채운다. 그리고 D*는 jit 안에 못 들어가므로, 학습 중에는 nav를 부르는
것이 아니라 경로가 이미 만들어진 채로 상태에 들어와 있어야 한다.

왜 방향과 거리로 쪼개는가
    좌표 차이 (dx, dy)를 그대로 주면 두 정보가 한 숫자에 섞인다. 3 m 앞과
    30 m 앞이 같은 방향인데 크기만 10배 다른 벡터가 되어, 가까울 때의
    미세한 방향 차이가 먼 점의 크기에 묻힌다. 방향은 단위벡터로 크기를
    없애고, 거리는 로그로 눌러서 둘을 분리한다.

왜 각도가 아니라 단위벡터인가
    각도는 +pi와 -pi가 같은 방향인데 숫자로는 정반대다. 로봇이 뒤를 볼 때마다
    입력이 튄다. 단위벡터에는 그 이음매가 없다.

왜 3점인가
    1점이면 벽 뒤로 곧장 가려 한다. 많이 줄수록 좋지만 먼 점은 재계획으로
    금방 바뀌어 학습에 도움이 안 된다. 가까이 · 중간 · 멀리 셋으로 잡았다.
"""

from __future__ import annotations

import jax.numpy as jnp

#: 경로를 따라 이 거리에 있는 점을 뽑는다 (m). 경로가 짧으면 끝점으로 잘린다.
LOOKAHEAD = (0.5, 1.5, 3.0)

#: 한 점을 몇 숫자로 표현하는가. [방향x, 방향y, 압축거리]
FEATURES = 3

#: 출력 크기. HLC는 이 값을 읽어 쓴다. **숫자를 옮겨 적지 말 것.**
SIZE = len(LOOKAHEAD) * FEATURES

#: 거리를 log(1+d)/LOG_SCALE 로 누른다. 가장 먼 관심 거리에서 1이 되게 잡았다.
LOG_SCALE = float(jnp.log1p(jnp.asarray(LOOKAHEAD[-1])))

_EPS = 1e-9


def resample(points, distances=LOOKAHEAD):
    """경로를 따라 주어진 거리만큼 간 지점들. 경로 좌표 그대로 (K, 2)로 낸다.

    직선거리가 아니라 **경로를 따라간 거리**다. 경로가 굽어 있으면 3 m를
    따라가도 직선으로는 2 m일 수 있고, 그게 맞다 -- 로봇은 그 길로 간다.

    경로가 요청한 거리보다 짧으면 마지막 점을 반복한다. 목표에 다 와서
    경로가 짧아졌을 때 자연스럽게 "세 점이 다 목표"가 된다.

    `points`의 개수는 **정적**이어야 한다(jit 제약). 값은 추적돼도 된다.
    """
    pts = jnp.asarray(points, dtype=jnp.float32).reshape(-1, 2)
    want = jnp.asarray(distances, dtype=jnp.float32)
    if pts.shape[0] == 1:
        return jnp.repeat(pts, want.shape[0], axis=0)

    seg = jnp.linalg.norm(jnp.diff(pts, axis=0), axis=1)
    along = jnp.concatenate([jnp.zeros(1, pts.dtype), jnp.cumsum(seg)])

    d = jnp.minimum(want, along[-1])                       # 경로 끝을 넘지 않는다
    j = jnp.clip(jnp.searchsorted(along, d, side="right") - 1, 0, seg.shape[0] - 1)

    # 길이 0인 구간에서 0으로 나누지 않는다. 그 구간의 보간 비율은 의미가 없다.
    denom = jnp.where(seg[j] > _EPS, seg[j], 1.0)
    t = jnp.where(seg[j] > _EPS, (d - along[j]) / denom, 0.0)
    return pts[j] + t[:, None] * (pts[j + 1] - pts[j])


def encode(points, robot_xy, robot_yaw, distances=LOOKAHEAD):
    """경로 -> (SIZE,). nav도 HLC도 이 함수를 통해서만 오간다.

        feat = encode(경로, robot_xy, robot_yaw)
        action = hlc.act(feat)

    출력은 전부 로봇 기준이다. 지도 좌표가 새어 나가지 않는다 -- 로봇은
    자기가 지도 어디에 있는지 알 필요가 없고, 알면 지도를 외워버린다.
    """
    robot_xy = jnp.asarray(robot_xy, dtype=jnp.float32).reshape(2)
    targets = resample(points, distances)

    delta = targets - robot_xy                      # 지도 좌표 기준 상대 위치
    dist = jnp.linalg.norm(delta, axis=1)

    # 지도 좌표 -> 로봇 좌표. 로봇이 바라보는 쪽이 +x가 되도록 -yaw 만큼 돌린다.
    c, s = jnp.cos(-robot_yaw), jnp.sin(-robot_yaw)
    local = jnp.stack([c * delta[:, 0] - s * delta[:, 1],
                       s * delta[:, 0] + c * delta[:, 1]], axis=1)

    # 방향만 남긴다. 겹친 점(거리 0)은 정면으로 둔다 -- 0으로 나누지 않기 위해서다.
    safe = jnp.where(dist > _EPS, dist, 1.0)[:, None]
    unit = jnp.where(dist[:, None] > _EPS, local / safe,
                     jnp.asarray([1.0, 0.0], dtype=local.dtype))

    return jnp.concatenate([unit, (jnp.log1p(dist) / LOG_SCALE)[:, None]],
                           axis=1).reshape(-1)


def yaw_from_quat(quat):
    """MuJoCo 쿼터니언 (w, x, y, z) -> 요각 rad. 로봇이 지도에서 어느 쪽을 보는가."""
    q = jnp.asarray(quat)
    w, x, y, z = q[0], q[1], q[2], q[3]
    return jnp.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
