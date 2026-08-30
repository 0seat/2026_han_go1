"""HLC 관측의 단일 출처 -- 배치와 크기를 여기서만 정한다.

`llc/spec.py`가 **명령** 쪽 인터페이스를 얼린 것처럼, 이 파일은 **관측** 쪽을
얼린다. 이유가 같다. 관측 배치가 바뀌면 정책의 입력층 모양이 바뀌고, 그것은
미세조정이 아니라 처음부터 학습이다.

    얼린 것     세 덩어리의 순서와 각각의 크기
                높이 스캔의 격자 모양과 상대화 기준
                크리틱만 보는 것과 액터도 보는 것의 구분

    안 얼린 것   스캔의 잡음 · 시야 제한 정도 (커리큘럼으로 흔든다)
                비용장을 무엇이 만드는가 (학습은 가치 반복, 배포는 nav의 D*)

배치
----

    액터 (state)                                                    크기
        자기 상태 x STACK      최근 3틱. 부분관측을 메우는 최소한       63
        높이 스캔              로봇 프레임 격자. 지면 기준              91
        천장 스캔              같은 격자. **몸통 기준 여유고**          91
        길잡이                 nav 경로 + 목표거리. 현재 틱만           10
                                                                    ---
                                                                    255

    크리틱 (privileged_state)
        액터가 보는 것 전부                                          255
        특권 정보              실기에 없는 것만                        21
                                                                    ---
                                                                    276

천장을 뒤늦게 넣었다
--------------------

2026-08-21에 추가했다. 그전까지 **천장이 관측에 없었다.** 터널 랜드가 이미 있었고
`height` 축이 요점이었는데도 빠져 있었다.

그래서 터널을 푼 정책(도달 0.953)은 **몸을 낮춰서 푼 것이 아니다.** 성공판의
몸통 최저 z가 0.250으로 평지와 같았다. 옆벽은 hfield라 지형 스캔에 보였고,
정책은 입구에 정렬해 들어가는 법을 배웠다. `TUNNEL_CLEAR`가 0.36이라 기본
자세로 85 mm가 남아서 그것으로 충분했다.

숙이는 법을 못 배운 것이 아니라 **배울 수 없었다.** 천장이 geom이라 높이 격자에
없고, 로봇은 그것을 볼 방법이 없었다. 부딪혀 봐야 자기수용감각으로 뒤늦게 아는데
그건 회피가 아니라 사고다.

이 누락이 관측 크기를 164에서 255로 바꾸므로 그때까지 학습한 것을 전부 버렸다.
평지 · 회전 · 터널 약 5시간이다. `SIGNATURE`가 이 파일에 생긴 이유다.

왜 자기 상태만 쌓는가
    LLC는 보행 위상과 행동 이력을 내부에 갖는데 HLC는 스냅샷만 본다. 그래서
    HLC 관점의 전이가 마르코프가 아니다. RNN 대신 최근 몇 틱을 쌓아 메운다.
    지형과 길잡이는 한 틱 사이에 거의 안 변하므로 쌓아봐야 같은 숫자만 3배가
    된다 -- 쌓는 것은 **빨리 변하는 것**뿐이다.

왜 명령 11축을 전부 넣는가
    지금 5축은 상수라 죽은 입력이다. 그래도 넣는다. `spec.py`에서 출력 폭을
    11로 얼린 것과 같은 이유로, footswing이 열리는 날 입력층을 안 고치기
    위해서다. 상수 입력은 정규화가 흡수한다.

**`maze.path`와 `maze.route()`는 여기 절대 들어오지 않는다.** 앞은 생성기가 판
자국이고 뒤는 최단 하나다 (`docs/contracts.md` C5). 길잡이가 쓰는 것은 통과
규칙으로 만든 **비용장**이며, 그 위의 어느 내리막을 타든 틀린 길이 아니다.
"""

from __future__ import annotations

import os as _os

import jax.numpy as jnp
from jax.scipy.ndimage import map_coordinates

from ..common import path as path_mod
from ..llc import spec
from . import maze


# ---------- 높이 스캔 ----------

#: 로봇 프레임 스캔 격자 (m). 앞이 +x, 왼쪽이 +y.
#:
#: 뒤로 0.4 m까지 보는 이유 -- phase18에서 후진이 열렸다. 방금 내려온 단차를
#: 못 보면 후진이 자살 행동이 된다.
#:
#: 앞으로 2.0 m는 랜드 한 칸(`TILE` = 2.0)이다. 다음 랜드의 경계까지 본다.
SCAN_FWD = (-0.4, 2.0)

#: 좌우 +-0.6 m. 외나무다리 폭이 0.40이라 양쪽 벼랑이 시야에 들어온다.
SCAN_LAT = (-0.6, 0.6)

#: 격자 간격 (m). 돌 지름이 2 * ROCK_RADIUS = 0.44라 두 점 이상이 돌에 걸린다.
#: 틈(0.5)과 다리(0.40)도 마찬가지다.
#:
#: **주의 —** 거침(요철 0.25 m)은 이 간격으로 표현이 안 된다. 계단현상으로
#: 잡음처럼 들어온다. 그것으로 충분하다 -- HLC가 요철 하나하나를 피할 이유는
#: 없고 "이 랜드가 거칠다"만 알면 된다. 발을 어디 딛을지는 LLC의 몫이다.
SCAN_STEP = 0.2

#: 스캔 값을 이 범위(m)로 자른다. 절벽(-1.0)과 벽(+0.8)이 안 잘리는 폭이다.
#: 2단 언덕(1.43 m)은 포화되는데, 포화돼도 "못 올라간다"는 결론이 같아 괜찮다.
SCAN_CLIP = 1.0

SCAN_NX = int(round((SCAN_FWD[1] - SCAN_FWD[0]) / SCAN_STEP)) + 1      #: 13
SCAN_NY = int(round((SCAN_LAT[1] - SCAN_LAT[0]) / SCAN_STEP)) + 1      #: 7
SCAN_SIZE = SCAN_NX * SCAN_NY                                          #: 91


# ---------- 천장 스캔 ----------

#: 천장 스캔은 지형 스캔과 **같은 격자**를 쓴다. `scan_offsets()`를 그대로
#: 재사용한다. 격자를 따로 정하면 "왜 13 x 3 인가"를 또 설명해야 하고, 근거 없이
#: 고른 상수가 이 파일에 하나 더 생긴다.
#: 천장 블록을 끌 수 있다. **대조군 전용이다.**
#:
#: 2026-08-21에 천장을 넣은 뒤 평지 직진이 학습되지 않았다. 이득 · 저크 · 회전을
#: 차례로 의심해 전부 반증했고, 남은 구조적 차이가 천장 블록뿐이었다. 그런데
#: 기전이 약하다 -- 평지에서 이 91칸은 상수 1.0 이고 정규화를 거치면 정확히 0 이라
#: 순전파에 아무것도 안 더한다. 그래서 **추측 대신 대조군을 만든다.**
#:
#: 켜고 끄는 것 말고는 모든 코드가 같다. 환경변수로 두는 이유는 모듈 상수라
#: 임포트 시점에 정해져야 하기 때문이다.
#:
#:     GO1_CEIL=0 python ...        천장 없음 (SIZE 164)
#:     기본                          천장 있음 (SIZE 255)
#:
#: 판정이 끝나면 이 스위치는 지운다. 영구히 둘 물건이 아니다.
CEIL_ON = _os.environ.get("GO1_CEIL", "1") != "0"

CEIL_SIZE = SCAN_SIZE if CEIL_ON else 0                                #: 91 또는 0

#: 여유고를 이 범위(m)로 자른다. 천장이 없는 자리는 `+CEIL_CLIP` 이다.
#:
#: 음수는 이미 닿았다는 뜻이다. 자르지 않고 남기는 이유 -- 얼마나 깊이 박혔는지가
#: "조금 낮추면 되는가"와 "여기는 못 간다"를 가른다.
CEIL_CLIP = 1.0

#: **기준이 몸통이다.** 지형 스캔은 지면 기준인데 여기만 다르다. 일부러다.
#:
#: 지형은 모양을 알아야 한다 -- 어디를 밟을지, 얼마나 가파른지. 그래서 자세와
#: 섞이면 안 되고 지면이 기준이어야 한다.
#:
#: 천장은 **닿느냐만** 알면 된다. 그리고 닿느냐는 여유고 하나로 정해지는 이진
#: 판정이다. 지면 기준으로 주면 정책이 그 값을 자기 높이(`proprio`에 있다)와
#: 조합해서 여유고를 만들어 내야 한다. 안전에 직결되는 양을 한 번 더 계산시킬
#: 이유가 없다.
#:
#: 기준점은 **몸통 원점**(`qpos[2]`)이다. 몸통 윗면 좌표를 코드에 박지 않는다 --
#: 그러면 그것이 또 하나의 조용한 상수가 된다. 원점에서 윗면까지는 고정 오프셋이라
#: 정책이 문턱을 학습으로 흡수한다.
CEIL_REFERENCE = "몸통 원점 (qpos[2])"


# ---------- 덩어리 크기 ----------

#: 자기 상태 = 선속도 3 + 각속도 3 + 중력 3 + 지면 대비 높이 1 + 직전 명령 11
PROPRIO_SIZE = 3 + 3 + 3 + 1 + spec.DIM

#: 몇 틱을 쌓는가. 3틱이면 HLC 10 Hz에서 0.3초 -- 보폭 하나(2.63 Hz)보다 길다.
STACK = 3

#: 길잡이 = 경로 특징 + [목표까지 직선거리]
#:
#: 비용장(잔여비용·기울기)을 **일부러 뺐다.** 랜드 단위로 계산한 비용장의 기울기는
#: 2 m 주기 구조를 갖고, 그것을 정책 입력에 넣으면 정책이 "세상은 2 m 격자다"를
#: 학습한다. 연속 지형에 놓는 순간 그 신호가 사라져 본 적 없는 입력이 된다.
#:
#: 웨이포인트는 그 문제가 없다 -- "0.5 m 앞은 이쪽"은 격자가 2 m든 0.2 m든
#: 사람 조이스틱이든 같은 모양이다. **정책이 계획기의 해상도를 모르게 한다.**
#:
#: 비용장은 버리는 것이 아니라 **학습기 안으로 옮긴다.** 보상의 퍼텐셜로 쓰고
#: 정책은 못 본다. 보상은 배포에 없으니 전이에 영향이 없다.
#:
#: 목표 거리 하나만 남기는 이유 -- 경로가 3 m 앞에서 잘려 오면 목표가 3 m 앞인지
#: 30 m 앞인지 구분이 안 된다. 이 값은 격자와 무관하다.
GUIDE_SIZE = path_mod.SIZE + 1

#: 액터가 보는 전체 크기. **숫자를 옮겨 적지 말 것.**
SIZE = PROPRIO_SIZE * STACK + SCAN_SIZE + CEIL_SIZE + GUIDE_SIZE

#: 크리틱만 보는 것 = 랜드 종류 one-hot 10 + 단 1 + 마찰 1 + 발 접촉 4
#:                   + LLC 이득/편향 2 + 월드 선속도 3
PRIV_EXTRA = len(maze.IMPLEMENTED) + 1 + 1 + 4 + 2 + 3

#: 크리틱이 보는 전체 크기.
PRIV_SIZE = SIZE + PRIV_EXTRA

def scan_offsets() -> jnp.ndarray:
    """로봇 프레임 스캔 점들 (SCAN_SIZE, 2). 정적이라 한 번만 만들면 된다."""
    fx = jnp.linspace(SCAN_FWD[0], SCAN_FWD[1], SCAN_NX)
    fy = jnp.linspace(SCAN_LAT[0], SCAN_LAT[1], SCAN_NY)
    gx, gy = jnp.meshgrid(fx, fy, indexing="ij")
    return jnp.stack([gx.reshape(-1), gy.reshape(-1)], axis=1)


def terrain_scan(height, robot_xy, robot_yaw, shape) -> jnp.ndarray:
    """높이 격자에서 로봇 주변을 뜬다 -> (SCAN_SIZE,) 미터.

    `height`는 `maze.heightfield`가 낸 [0, 1] 정규화 격자 그대로 넣는다.
    여기서 미터로 되돌리고 **발밑 지면 높이를 뺀다.**

    지면 기준으로 상대화하는 이유 -- 몸통 높이는 HLC가 직접 명령하는 값이라,
    몸통 기준으로 재면 "내가 몸을 낮춘 것"과 "땅이 솟은 것"이 같은 숫자가 된다.
    지형 모양과 자세를 섞지 않으려면 지면이 기준이어야 한다.

    지도 밖은 가장자리 값으로 늘린다. 0으로 채우면 맵 밖이 평지로 보여서
    로봇이 밖으로 나가려 한다.
    """
    ty, tx = shape
    ex, ey = tx * maze.TILE, ty * maze.TILE

    off = scan_offsets()
    c, s = jnp.cos(robot_yaw), jnp.sin(robot_yaw)
    # 로봇 프레임 -> 월드. +x가 로봇이 보는 쪽이다.
    wx = robot_xy[0] + c * off[:, 0] - s * off[:, 1]
    wy = robot_xy[1] + s * off[:, 0] + c * off[:, 1]

    def sample(x, y):
        # 월드 xy -> 격자 좌표. maze.tile_center와 같은 규약(맵 중심이 원점)이다.
        j = (x + ex / 2.0) / maze.CELL - 0.5          # 열 = x
        i = (y + ey / 2.0) / maze.CELL - 0.5          # 행 = y
        v = map_coordinates(height, [i, j], order=1, mode="nearest")
        return v * maze.SPAN - maze.DEPTH

    ground = sample(jnp.asarray([robot_xy[0]]), jnp.asarray([robot_xy[1]]))[0]
    return jnp.clip(sample(wx, wy) - ground, -SCAN_CLIP, SCAN_CLIP)


def ceiling_scan(boxes, robot_xy, robot_yaw, body_z) -> jnp.ndarray:
    """천장까지 남은 여유고 -> (CEIL_SIZE,) 미터. **몸통 원점 기준.**

    `boxes`는 `maze.ceilings`의 (N, 6) = 중심 xyz + 반크기 xyz다. 축정렬이라
    회전을 볼 필요가 없다. 상자의 **바닥면**(cz - hz)이 부딪히는 면이다.

    한 점 위에 상자가 여럿이면 가장 낮은 것을 쓴다. 없으면 `+CEIL_CLIP`이다 --
    0으로 채우면 "천장이 없다"와 "천장이 정확히 머리에 닿았다"가 같은 숫자가 되어
    최악의 자리와 가장 안전한 자리가 구별되지 않는다.

    `N = 0`(터널이 없는 지형)이어도 모양이 안 변한다. `jnp.min`의 `initial`이
    빈 축을 받아 준다 -- 분기를 넣으면 jit 안에서 못 쓴다.
    """
    if not CEIL_ON:                          # 대조군. 블록 자체가 사라진다
        return jnp.zeros(0, jnp.float32)
    boxes = jnp.asarray(boxes, jnp.float32).reshape(-1, 6)
    robot_xy = jnp.asarray(robot_xy, jnp.float32).reshape(2)

    off = scan_offsets()
    c, s = jnp.cos(robot_yaw), jnp.sin(robot_yaw)
    wx = robot_xy[0] + c * off[:, 0] - s * off[:, 1]
    wy = robot_xy[1] + s * off[:, 0] + c * off[:, 1]

    cx, cy, cz = boxes[:, 0], boxes[:, 1], boxes[:, 2]
    hx, hy, hz = boxes[:, 3], boxes[:, 4], boxes[:, 5]

    inside = ((jnp.abs(wx[:, None] - cx[None, :]) <= hx[None, :])
              & (jnp.abs(wy[:, None] - cy[None, :]) <= hy[None, :]))
    clear = (cz - hz)[None, :] - jnp.asarray(body_z, jnp.float32)

    lowest = jnp.min(jnp.where(inside, clear, CEIL_CLIP), axis=1,
                     initial=CEIL_CLIP)
    return jnp.clip(lowest, -CEIL_CLIP, CEIL_CLIP)


def ground_at(height, xy, shape) -> jnp.ndarray:
    """로봇 발밑 지면 높이 (m). 몸통 높이를 지면 기준으로 만들 때 쓴다.

    `terrain_scan`이 안에서 하는 것과 같은 계산이다. 밖에서도 필요해서 꺼냈다 --
    두 곳에서 다른 방식으로 지면을 구하면 스캔과 스칼라가 어긋난다.
    """
    ty, tx = shape
    ex, ey = tx * maze.TILE, ty * maze.TILE
    xy = jnp.asarray(xy, jnp.float32).reshape(2)
    j = (xy[0] + ex / 2.0) / maze.CELL - 0.5
    i = (xy[1] + ey / 2.0) / maze.CELL - 0.5
    v = map_coordinates(height, [i.reshape(1), j.reshape(1)], order=1, mode="nearest")
    return v[0] * maze.SPAN - maze.DEPTH


#: 자기 상태의 물리적 상한. **여기만 안 막혀 있었다.**
#:
#: 지형은 `SCAN_CLIP`, 천장은 `CEIL_CLIP`, 길잡이는 단위벡터 + 로그거리라 전부
#: 유계인데 자기 상태만 원값 그대로였다. 그래서 물리가 수치적으로 터지면 그 값이
#: 신경망까지 그대로 들어갔다.
#:
#: 2026-08-21, 평지 판이 819,200 스텝에서 도달 0.328 까지 갔다가 1,638,400 에서
#: `v_loss` 가 **1,064,283** 으로 튀고 그 뒤 5,700,000 스텝을 도달 0.000 으로
#: 보냈다. `kl` 은 내내 정상(0.008~0.052)이었다 -- 정책 갱신이 아니라 **가치가
#: 오염된 것**이고, 가치가 망가지면 이득이 망가져 정책이 밀려난다.
#:
#: 값의 근거
#:     선속도  LLC 명령 상한이 1.0 m/s 다. 5배 여유
#:     각속도  요각 명령 상한이 0.35 rad/s 다. 넘어질 때 몸통이 도는 것까지 담는다
#:     높이    지형이 담는 범위가 -1.10 ~ 0.71 m 다. 그 밖은 이미 지도 밖이다
#:
#: 정상 주행은 이 한계 근처에도 안 간다. 걸리는 것은 폭주뿐이다.
PROPRIO_CLIP = {"linvel": 5.0, "gyro": 20.0, "height": 1.0}


def proprio(linvel, gyro, gravity, height_above_ground, command) -> jnp.ndarray:
    """자기 상태 (PROPRIO_SIZE,). 전부 몸통 좌표계다.

    `command`는 직전에 HLC가 낸 **11축 전체**다. 6축만 넣으면 축이 열릴 때
    입력층이 바뀐다.

    **NaN 을 0 으로 바꾼다.** 한 번이라도 NaN 이 들어오면 정규화의 러닝 통계가
    통째로 NaN 이 되고, 그 뒤 모든 환경의 관측이 NaN 이 된다. 한 환경의 사고가
    배치 전체를 영구히 죽이는 경로라 반드시 막는다.

    중력 벡터는 안 자른다 -- 단위벡터라 이미 [-1, 1] 이다. 명령도 안 자른다 --
    `action.to_command` 가 `spec.RANGES` 로 이미 clip 한다.
    """
    def _cut(x, n, lim):
        x = jnp.asarray(x, jnp.float32).reshape(n)
        return jnp.clip(jnp.nan_to_num(x, nan=0.0, posinf=lim, neginf=-lim),
                        -lim, lim)

    return jnp.concatenate([
        _cut(linvel, 3, PROPRIO_CLIP["linvel"]),
        _cut(gyro, 3, PROPRIO_CLIP["gyro"]),
        _cut(gravity, 3, 1.0),
        _cut(height_above_ground, 1, PROPRIO_CLIP["height"]),
        jnp.asarray(command, jnp.float32).reshape(spec.DIM),
    ])


def guide(points, robot_xy, robot_yaw, goal_xy):
    """길잡이 (GUIDE_SIZE,). nav가 준 경로 + 목표까지 직선거리.

    `points`는 nav의 출력이다. 1단계에서는 nav가 없으므로 `[로봇, 다음 랜드 중심]`
    두 점을 넣는다 -- 같은 함수, 같은 형식이다. 3단계에서 진짜 D*가 오면 부르는
    쪽만 바뀌고 여기는 안 바뀐다.
    """
    feat = path_mod.encode(points, robot_xy, robot_yaw)
    d = jnp.linalg.norm(jnp.asarray(goal_xy, jnp.float32).reshape(2)
                        - jnp.asarray(robot_xy, jnp.float32).reshape(2))
    return jnp.concatenate([feat, (jnp.log1p(d) / path_mod.LOG_SCALE).reshape(1)])


def route_progress(route, robot_xy):
    """경로상 **남은 거리** (스칼라 m). 보상이 이걸 쓴다.

    `route_polyline` 과 같은 정사영을 쓴다. 다른 값을 두 번 계산하면 관측과
    보상이 어긋날 수 있어서, 계산을 한 곳에 두고 둘 다 여기서 나온다.

    직선 복도는 경로가 목표 한 점이라 **직선거리와 정확히 같다.** 그래서 예전
    판들의 보상이 안 바뀐다.
    """
    pts = jnp.asarray(route, jnp.float32).reshape(-1, 2)
    robot = jnp.asarray(robot_xy, jnp.float32).reshape(2)
    if pts.shape[0] == 1:
        return jnp.linalg.norm(pts[0] - robot)
    _, s_robot, total = _project(pts, robot)
    return jnp.maximum(total - s_robot, 0.0)


def _project(pts, robot):
    """로봇을 경로에 정사영. `(구간 index, 경로상 위치, 경로 전체 길이)`.

    로봇을 경로 앞에 붙여서 재면 **뒤로 가는 구간이 누적 거리에 섞인다.**
    실측으로 그렇게 틀렸다. 누적 거리는 경로만으로 잰다.
    """
    seg = pts[1:] - pts[:-1]
    length = jnp.linalg.norm(seg, axis=1)
    along = jnp.concatenate([jnp.zeros(1, jnp.float32), jnp.cumsum(length)])
    denom = jnp.where(length > 1e-9, length ** 2, 1.0)
    t = jnp.clip(jnp.sum((robot[None, :] - pts[:-1]) * seg, axis=1) / denom,
                 0.0, 1.0)
    foot = pts[:-1] + t[:, None] * seg
    k = jnp.argmin(jnp.linalg.norm(foot - robot[None, :], axis=1))
    return along, along[k] + t[k] * length[k], along[-1]


def route_polyline(route, robot_xy):
    """경로 점들 -> `guide` 에 넣을 폴리라인. **지나온 점을 지운다.**

    `path.resample` 은 폴리라인 **첫 점부터** 경로를 따라 걷는다. 그래서 로봇이
    이미 지나친 경유점이 앞에 남아 있으면 0.5 m 앞이 뒤쪽을 가리킨다.

    지우는 방법 -- 배열 크기가 고정이어야 하므로(jit) 점을 빼지 않고 **뒤쪽 점을
    로봇 위치로 덮는다.** 로봇에서 로봇까지는 길이 0 구간이라 걷는 거리에
    기여하지 않고, `resample` 이 그 구간을 이미 방어하고 있다.

    앞뒤 판정은 경로를 따라간 거리로 한다. 로봇을 각 구간에 정사영해 가장 가까운
    구간을 찾고, 그 지점의 누적 거리보다 뒤에 있는 점을 지운다. 직선거리로
    판정하면 ㄱ 자 모서리에서 다음 구간의 점이 더 멀어 보여 지워진다.

    한 점짜리 경로(`[목표]`)면 `[로봇, 목표]` 가 되어 **예전과 같다.**
    """
    pts = jnp.asarray(route, jnp.float32).reshape(-1, 2)
    robot = jnp.asarray(robot_xy, jnp.float32).reshape(2)
    if pts.shape[0] == 1:
        return jnp.concatenate([robot[None, :], pts], axis=0)

    along, s_robot, _ = _project(pts, robot)
    keep = along > s_robot                      # 로봇보다 뒤에 있는 점을 덮는다
    kept = jnp.where(keep[:, None], pts, robot[None, :])
    return jnp.concatenate([robot[None, :], kept], axis=0)


def assemble(proprio_stack, scan, ceil, guide_vec) -> jnp.ndarray:
    """네 덩어리 -> 액터 관측 (SIZE,). 순서가 여기서 정해진다.

    `proprio_stack`은 (STACK, PROPRIO_SIZE)이고 **0번이 가장 최근**이다.
    천장은 지형 스캔 바로 뒤에 둔다 -- 같은 격자라 붙여 두는 편이 읽기 쉽다.
    """
    p = jnp.asarray(proprio_stack).reshape(STACK, PROPRIO_SIZE)
    out = jnp.concatenate([p.reshape(-1),
                           jnp.asarray(scan).reshape(SCAN_SIZE),
                           jnp.asarray(ceil).reshape(CEIL_SIZE),
                           jnp.asarray(guide_vec).reshape(GUIDE_SIZE)])
    return out


def assemble_privileged(actor_obs, kind_onehot, level, friction,
                        foot_contact, llc_gain_bias, world_linvel):
    """액터 관측 + 특권 정보 -> 크리틱 관측 (PRIV_SIZE,).

    특권으로 넣는 기준은 **실기에 없는 것**이다. 랜드 종류와 마찰은 시뮬만
    알고, LLC 이득/편향은 학습 중 랜덤화한 값 자체다 -- 크리틱이 그것을 알면
    "이 환경이 원래 어려웠다"를 가치에서 빼줄 수 있어 분산이 준다.
    """
    return jnp.concatenate([
        jnp.asarray(actor_obs).reshape(SIZE),
        jnp.asarray(kind_onehot, jnp.float32).reshape(len(maze.IMPLEMENTED)),
        jnp.asarray(level, jnp.float32).reshape(1),
        jnp.asarray(friction, jnp.float32).reshape(1),
        jnp.asarray(foot_contact, jnp.float32).reshape(4),
        jnp.asarray(llc_gain_bias, jnp.float32).reshape(2),
        jnp.asarray(world_linvel, jnp.float32).reshape(3),
    ])


# ---------- 배치 서명 ----------
#
# 왜 있는가
# ---------
#
# 파라미터를 못 이어 쓰게 만드는 변경이 두 종류다.
#
#     크기가 바뀐다      `restore`가 즉시 터진다. 아프지만 **소리가 난다**
#     뜻만 바뀐다        `restore`가 성공하고 로봇만 이상해진다. **소리가 안 난다**
#
# 뒤쪽이 진짜 문제다. `action.SCALE`을 올리면 같은 출력이 다른 명령이 되고,
# `maze.SPAN`을 바꾸면 모든 스캔 값의 환산 계수가 바뀌고, `stage1.REPEAT`를
# 바꾸면 다이나믹스가 바뀐다. 크기는 전부 그대로다.
#
# 서명은 뒤쪽을 앞쪽으로 내린다. 배치를 정하는 상수 전부를 해시해서 파라미터와
# 같이 저장하고, 되읽을 때 대조한다. 다르면 **로드를 거부한다.**
#
# 무른 항목
# ---------
#
# `spec.TRAINED`와 LLC 체크포인트는 **경고만** 낸다. 그것들은 바뀌는 것이
# 정상이고 -- footswing이 열리고 LLC가 갱신되는 것은 예정된 일이다 -- 거부하면
# 미세조정 경로가 막힌다. `llc_gain` 무작위화가 일부러 그것을 대비한 장치다.


def _signature_fields() -> dict:
    """서명에 들어가는 것 전부. **바꾸면 옛 파라미터를 못 읽는다.**"""
    from ..llc import spec as _spec
    from . import action as _action, skills as _skills

    return {
        "명령축": list(_spec.COMMAND_ORDER),
        "명령범위": {k: list(v) for k, v in sorted(_spec.RANGES.items())},
        "행동중심": dict(sorted(_action.CENTRE.items())),
        "행동배율": dict(sorted(_action.SCALE.items())),
        "행동크기": _skills.ACTION_SIZE,
        "게이트": _skills.N_GATES,
        "스캔전후": list(SCAN_FWD),
        "스캔좌우": list(SCAN_LAT),
        "스캔간격": SCAN_STEP,
        "스캔클립": SCAN_CLIP,
        "천장켜짐": CEIL_ON,
        "천장클립": CEIL_CLIP,
        "천장기준": CEIL_REFERENCE,
        "자기상태": PROPRIO_SIZE,
        # 크기는 안 바꾸는데 **값의 뜻을 바꾼다.** 서명이 잡아야 할 딱 그 종류다.
        "자기상태클립": dict(sorted(PROPRIO_CLIP.items())),
        "쌓기": STACK,
        "길잡이": GUIDE_SIZE,
        "전방거리": list(path_mod.LOOKAHEAD),
        "액터크기": SIZE,
        "크리틱크기": PRIV_SIZE,
        "랜드종류수": len(maze.IMPLEMENTED),
        "타일": maze.TILE,
        "격자": maze.CELL,
        "높이범위": maze.SPAN,
        "깊이": maze.DEPTH,
    }


def _soft_fields() -> dict:
    """바뀌어도 거부하지 않는 것. 경고만 낸다."""
    from ..llc import spec as _spec

    return {
        "학습된축": dict(sorted(_spec.TRAINED.items())),
        "LLC출처": _spec.SOURCE,
    }


def _digest(fields: dict) -> str:
    import hashlib
    import json
    blob = json.dumps(fields, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


#: 지금 배치의 서명. `train.save`가 파라미터와 같이 저장한다.
SIGNATURE = _digest(_signature_fields())


def _diff(saved: dict | None, now: dict) -> list[str]:
    """달라진 항목을 사람이 읽는 줄로. **해시만 저장하지 않는 이유가 이것이다.**

    해시가 다르다는 것만 알면 무엇이 왜 다른지 찾느라 파일을 뒤져야 한다.
    항목 전체는 몇백 바이트라 같이 저장하는 편이 싸다.
    """
    if not saved:
        return ["(저장본에 항목이 없다)"]
    out = []
    for k in sorted(set(saved) | set(now)):
        a, b = saved.get(k, "(없음)"), now.get(k, "(없음)")
        if a == b:
            continue
        # 사전은 통째로 찍지 않는다. 열한 축짜리 사전을 두 번 찍으면 어느 축이
        # 다른지 사람이 눈으로 찾아야 하는데, 그러라고 만든 도구가 아니다.
        if isinstance(a, dict) and isinstance(b, dict):
            for kk in sorted(set(a) | set(b)):
                if a.get(kk, "(없음)") != b.get(kk, "(없음)"):
                    out.append(f"{k}.{kk}  저장 {a.get(kk, '(없음)')}"
                               f"  ->  지금 {b.get(kk, '(없음)')}")
        else:
            out.append(f"{k}  저장 {a}  ->  지금 {b}")
    return out


def check_signature(saved: str, saved_fields: dict | None = None) -> list[str]:
    """저장된 서명을 지금 배치와 대조. 빈 목록이면 같다."""
    if saved == SIGNATURE:
        return []
    return _diff(saved_fields, _signature_fields())


def soft_diff(saved_soft: dict | None) -> list[str]:
    """무른 항목 중 달라진 것. 경고용이라 로드를 막지 않는다."""
    if not saved_soft:
        return []
    return _diff(saved_soft, _soft_fields())
