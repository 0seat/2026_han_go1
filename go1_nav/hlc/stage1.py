"""1단계 -- **랜드 하나 건너기.** HLC PPO 의 첫 과제.

과제 정의
---------

    출발 랜드에서 시작해 장애물 랜드를 지나 도착 랜드에 닿는다.

이것이 근사가 아니라 **과제 정의 그대로**인 것이 요점이다. nav 가 계획하고 HLC 가
실행한다는 분업에서 HLC 의 임무가 원래 랜드 A -> B 다. 갈림길에서 어디로 갈지
고르는 것은 nav 몫이라 HLC 가 배울 필요가 없다.

미로 전체로 학습하지 않는 이유

    에피소드가 짧다        크레딧 할당이 쉽다. 훨씬 빨리 배운다
    장애물 표본이 고르다    미로에서는 다리 · 터널이 드물어 표본이 굶는다
    보상이 단순하다        우회할 벽이 없으니 비용장이 필요 없다. 거리가 곧 비용이다

진입 조건 무작위화가 핵심이다
-----------------------------

단일 랜드 학습이 놓치는 유일한 것이 **진입 조건**이다. 매번 랜드 중심에 정지
상태로 서서 시작하면, 미로에서 앞 랜드를 통과해 **속도를 갖고 비스듬히 밀려
들어오는** 상황을 한 번도 안 본다. 이으면 경계마다 분포 이탈이고, 각 랜드가
95%여도 20칸이면 36%가 된다.

**주의 —** 그래서 brax 래퍼를 `full_reset=True` 로 감아야 한다. 기본값은 첫 상태를
캐시해 두고 되돌리는데, 그러면 모든 에피소드가 **같은 진입 조건**이 되어 이
무작위화가 통째로 무력해진다. `train.py` 가 그렇게 감는다.

시간 축
-------

HLC 는 10 Hz 다 (`REPEAT = 5`, LLC 50 Hz). 같은 주기로 두면 안 되는 이유가 둘이다.
LLC 는 준정적 명령으로 학습됐으므로 20 ms 마다 명령이 튀면 학습 분포 밖으로
나가고, 한 스텝이 아무것도 안 바꾸면 크레딧 할당이 지옥이 된다.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from mujoco_playground._src import mjx_env

from ..common import path as path_enc
from ..llc import loader, spec
from . import action, env as hlc_env, lands, maze, obs, skills

#: LLC 스텝 몇 개가 HLC 한 스텝인가. 50 Hz / 5 = 10 Hz.
REPEAT = 5

#: 에피소드 상한 (HLC 스텝). 200 에서 올렸다.
#:
#: **근거 -- 상한이 실제로 걸리고 있었다.** 미로 구간 판(12 m)을 500 스텝으로
#: 다시 재니 한 차선이 0.750 에서 1.000 이 되었다. 평균 168 스텝이라 200 에
#: 붙어 있었고, 그 차선은 학습 내내 **할 수 있는데 못 한 것으로** 신호를 받았다.
#:
#: 회전이 시간 예산의 주범이다. 요각 명령 상한 0.35 에 실측 이득 0.774 이므로
#: 최대 0.271 rad/s 이고, 90도 회전에 5.8 초 = 58 스텝이다. 미로의 정답지는
#: 12 m 구간에 꺾임이 5~7 번이다. 경사 칸에서는 더 느리다.
#:
#: 서명에는 안 들어간다 -- 올려도 옛 파라미터를 그대로 이어 쓴다.
MAX_STEPS = 300

#: 도착 판정 반지름 (m). 랜드가 2 m 니 중심에서 0.5 m 면 확실히 그 랜드 안이다.
GOAL_RADIUS = 0.5

#: 보상 계수.
#:
#: 주 신호는 **전진량**이다. 보상 = (직전 거리 - 지금 거리).
#: Ng 의 정리로 최적 정책이 안 바뀌는 것이 보장되고, 조밀해서 PPO 가 잘 먹는다.
#:
#: **성형에 gamma 를 쓰지 않는다.** 교과서 형태는 `gamma*phi(s') - phi(s)` 이고
#: 그래야 최적 정책이 안 바뀐다는 보장이 붙는다. 그런데 gamma < 1 이면 phi 에
#: 비례하는 **드리프트 항** `phi(gamma-1)` 이 생기고, 그것이 두 번 물었다.
#:
#:     1차   phi = -거리 로 뒀다. 드리프트가 +거리*0.01 이라 **가만히 있으면
#:           스텝당 +0.04 를 번다.** 200 스텝이면 확정 +7. 미학습 정책이 걸으면
#:           자주 넘어져(-10) 기댓값이 +1 쯤이라 정지가 이긴다. 게다가
#:           `action.CENTRE` 의 vx 가 0 이라 정책이 처음부터 그 자리에서 시작한다.
#:           50만 스텝 내내 도달 0.000 이었다.
#:
#:     2차   phi = SPAN - 거리 로 부호를 뒤집었다. 정지는 벌점이 됐는데 이번엔
#:           **목표에 가까울수록 전진 보상이 줄었다** -- 드리프트가 phi 와 함께
#:           커져서 거리 2 m 부터 전진이 음수가 된다.
#:
#: 두 번 다 원인이 같다. 그래서 성형에서 gamma 를 뺀다. `phi(s') - phi(s)` 는
#: 그냥 **전진한 거리**이고 드리프트가 0 이다. 정책 불변 보장은 잃지만, 도달
#: 보너스와 넘어짐 벌점을 얹은 시점에 그 보장은 이미 없었다. 보행 문헌이 쓰는
#: 형태이기도 하다.
#:
#: **주의 — 넘어짐에서 퍼텐셜을 0 으로 놓지 말 것.** phi 가 음수면 성형 항이
#: `0 - phi(s) = +거리` 가 되어 멀리서 넘어질수록 보상이 커진다. 그래서 종단에서
#: 퍼텐셜을 건드리지 않고, 넘어짐은 명시적 음수로만 벌한다.
#:
#: 계수를 만지면 `reward_sanity()` 를 부를 것. 물리 없이 순서와 부호를 본다.
SHAPING = 1.0
GOAL_BONUS = 10.0
FALL_PENALTY = -10.0
#: HLC 스텝당. **에피소드를 다 쓰면 -1.0 이 되게 유지한다.**
#:
#: `MAX_STEPS` 를 200 에서 300 으로 올리면서 같이 낮췄다. 0.005 를 그대로 두면
#: 시간 벌점 총액이 -1.5 가 되어, 상한을 올린 것이 **느린 주행을 더 벌하는**
#: 변경으로 바뀐다. 도달 보너스 10 대비 비율을 건드리지 않으려면 곱이 상수여야 한다.
TIME_COST = 1.0 / MAX_STEPS

#: 빠짐 판정 문턱 (m). 경로 위 지면보다 이만큼 아래면 도랑 · 절벽에 빠진 것이다.
#:
#: **왜 따로 세는가** -- 도랑에 네 발로 내려서면 넘어짐이 안 잡힌다. 실측 --
#: 다리 차선에서 최저 z 가 -0.736 인데 넘어짐 False 였고, 200 스텝을 도랑 안에서
#: 보내고 시간 초과로 기록됐다. 그러면 표에서 "장애물 앞에 서 있다"와 구분이
#: 안 되는데, 둘은 처방이 정반대다.
#:
#: 0.40 인 근거 -- 위험은 도랑 0.5 · 다리 1.0 · 절벽 1.0 이라 전부 걸리고,
#: 정상 요철은 거침 0.06 · 돌 0.14 라 안 걸린다.
#:
#: **기준선이 경로가 아니라 그 타일의 단이다.** 처음에는 경로에 정사영한 발점의
#: 지면과 견줬는데 두 가지로 틀렸다.
#:
#:     경사 열을 세로로 횡단하면 좌우가 곧 최대경사 방향이다. 20도에서 1.1 m 만
#:     옆으로 가면 0.40 m 라 통로(2 m) 안에서 오판이 난다
#:
#:     정답지는 지그재그라 자기 옆을 지난다. `route_foot` 이 2 m 옆의 **다른
#:     구간**을 잡으면 그쪽 단이 1 일 때 0.71 m 차이가 그냥 생긴다
#:
#: 실측 -- 6칸 구간에서는 전 차선 빠짐 0.000 인데 18칸 전체 경로에서 0.312 였다.
#: 지형이 스스로 아는 값을 쓰면 둘 다 사라진다. 도랑 · 다리 도랑 · 절벽은
#: **자기 타일의 단보다 파인 것**이고, 턱과 돌은 단 위로 솟는다. 경사 칸은 자기
#: 단과 이웃 단 사이를 보간하므로 단 아래로 안 내려간다.
PIT_MARGIN = 0.40

#: 몸통이 지면에 붙어 있다고 볼 거리 (m). **빠짐 판정의 둘째 조건이다.**
#:
#: 파인 곳 위를 **뛰어넘는 중**이면 발밑 지면은 파여 있지만 몸통은 멀리 떠 있다.
#: 이 조건이 없으면 점프가 생기는 순간 도랑을 넘을 때마다 빠짐으로 종료된다.
#: 몸통 높이 명령이 0.22~0.32 라 0.45 면 서 있는 자세를 전부 담고도 남는다.
STAND_CLEAR = 0.45

#: 경유점 지터의 기본 폭 (m). 통로 반폭이 터널 0.4 · 다리 0.3 이라, 0.2 면
#: 경유점이 통로 안에는 남으면서 중심은 아니다. 0 이면 예전과 같다.
ROUTE_JITTER = 0.2


def _jitter_route(route, key, width):
    """경유점을 **경로 진행 방향의 수직으로** 흔든다. 첫 점과 끝 점은 안 흔든다.

    첫 점은 출발이라 흔들면 로봇이 경로 밖에서 시작한 것처럼 보인다. 끝 점은
    목표라 흔들면 도달 판정과 어긋난다.
    """
    route = jnp.asarray(route, jnp.float32).reshape(-1, 2)
    n = route.shape[0]
    if n <= 2 or width <= 0.0:
        return route
    d = route[1:] - route[:-1]
    seg = jnp.linalg.norm(d, axis=1, keepdims=True)
    unit = d / jnp.where(seg > 1e-9, seg, 1.0)
    # 각 점의 법선은 그 점으로 **들어오는** 구간 방향에서 얻는다.
    normal = jnp.stack([-unit[:, 1], unit[:, 0]], axis=1)
    amp = jax.random.uniform(key, (n - 2, 1), minval=-width, maxval=width)
    moved = route[1:-1] + normal[:-1] * amp
    return jnp.concatenate([route[:1], moved, route[-1:]], axis=0)


#: 한 스텝 전진량의 상한 (m). **보상에서 유일하게 상한이 없던 항목을 막는다.**
#:
#: 전진량은 `거리(이전) - 거리(지금)` 이고 거리는 `qpos[0:2]` 에서 온다. 물리가
#: 수치적으로 터져 위치가 폭주하면 **그 한 스텝의 보상이 그대로 폭주한다.**
#: 8192 개 환경 중 하나만 그래도 그 배치의 가치 목표가 오염되고, 오염된 목표는
#: `v_loss` 를 백만 단위로 만든다.
#:
#: 실제로 그랬다 (2026-08-21, 평지, 천장 켬)
#:
#:       819,200   보상 +3.423  도달 0.328  v_loss         2.4
#:     1,638,400   보상 +2.582  도달 0.242  v_loss 1,064,283.1
#:     2,457,600   보상 -2.115  도달 0.000  v_loss    16,930.1
#:
#: 그 뒤 5,700,000 스텝을 회복하지 못했다. 819,200 시점 정책은 영상으로 확인해
#: 보면 59 스텝에 목표에 도달한다 -- **못 배운 것이 아니라 배운 것을 잃었다.**
#:
#: 값의 근거 -- `vx` 상한 1.0 m/s 에 HLC 주기 0.1 초면 한 스텝 최대 0.1 m 다.
#: 0.5 는 5배 여유라 정상 학습에는 절대 안 걸린다.
#:
#: **주의 —** 이것은 증상을 막는 것이지 원인을 고치는 것이 아니다. 원인은 MJX 가
#: 왜 터지는가이고 `발산` 지표가 켜진 판을 재현해야 알 수 있다. 다만 막아 두면
#: 그 판만 버리고 학습이 이어진다.
MAX_PROGRESS = 0.5

#: 이 거리(m)를 넘으면 폭주로 본다. 지도 대각선이 약 12 m 다.
#: 보상에 안 쓰고 **지표로만** 쓴다 -- 일어났는지 눈에 보여야 원인을 쫓을 수 있다.
DIVERGE_DIST = 50.0
#: 명령 변화율. LLC 를 학습 분포 안에 붙든다.
#:
#: **0.02 로 되돌렸다 (2026-08-21).** 한때 0.002 로 내렸는데, 그건 진단을 잘못
#: 밀어붙인 것이었다 -- 아래 분해는 사실이지만 **증상의 설명이지 원인의 증거가
#: 아니었다.** 역사적으로 평지 직진이 도달 1.000 까지 간 판은 0.02 로 돌았고,
#: 0.002 로 내려도 여전히 학습이 안 됐다. 알려진 정상 설정에서 변수를 하나 더
#: 늘린 셈이라 되돌린다.
#:
#: 분해 자체는 남겨 둔다. 나중에 형태를 고칠 때 근거가 된다.
#:
#: 평지 직진이 460만 스텝 동안 도달 0.000 으로 죽었을 때 항목별로 재 보니
#: 이렇게 나왔다.
#:
#:                   합계      전진     도달    넘어짐    시간     저크
#:     무작위 정책   -7.379   -0.689   +0.000   +0.000   -1.000   -5.690
#:     고정 vx=0.6  +13.734   +4.121  +10.000   +0.000   -0.375   -0.012
#:
#: 탐색하는 정책이 내는 벌점의 **77%가 저크**였다. 완주해서 버는 전진 보상
#: 전체가 +4.12 인데 저크 하나가 -5.69 다. 그러면 초기 정책이 보상을 올리는 가장
#: 가파른 길은 걷는 법을 찾는 것이 아니라 **명령을 안 바꾸는 것**이 된다. 즉시
#: +5.7 이 회수된다. 그리고 명령을 고정하면 탐색이 죽고, 남은 상수 중 제일 안전한
#: 것이 정지다. 실제로 학습된 정책이 `height` 0.320 에 붙어 제자리에 서 있었다.
#:
#: 단위가 섞인 것이 더 근본이다
#: ----------------------------
#:
#: 저크는 **물리 단위 그대로** 11축을 더한다. 축마다 단위가 달라 숫자가 큰 축이
#: 독식한다. 무작위 정책의 축별 기여를 보면
#:
#:     vx 0.67   yaw 0.23   pitch 0.20   roll 0.13   vy 0.10   height 0.03
#:
#: `vx` 가 m/s 라 숫자가 크다는 이유만으로 절반을 먹는다. 자세를 부드럽게 하려던
#: 벌점이 **전진 명령을 억제하는 데 대부분 쓰였다.**
#:
#: 옳은 형태는 정규화된 행동 공간에서 재는 것(`|raw - raw_prev|` 의 평균)이다.
#: 그러면 축이 고르게 세어지고 계수가 해석 가능해진다. **지금 안 바꾼다** --
#: 이 판에서 한 번에 여러 개를 바꿔 원인을 못 가린 적이 두 번 있었다. 걷기를
#: 배운 뒤에 형태를 고친다. 회전 · LLC 이득과 같은 커리큘럼이다.
JERK_COST = 0.02
DISCOUNT = 0.99            #: 성형에 쓰는 감마. PPO 의 감마와 같아야 한다

#: 시작 요각 범위 (rad). **랜드마다 다르게 잡으라고 `Task` 인자로 뺐다.**
#:
#:     평지        3.14 (+-180도).  평지는 그 자체가 조향 과제다.
#:                 회전을 배울 자리가 여기뿐이라 여기서 크게 준다
#:     장애물 랜드  0.52 (+-30도).  크게 주면 과제가 "돌기"로 바뀌어
#:                 장애물 통과를 못 배운다
#:
#: 목표는 항상 정면 +x 에 있으므로, 시작 요각을 흔드는 것이 곧 목표 방위각을
#: 흔드는 것이다. 지형을 바꿀 필요가 없다.
YAW_JITTER = 0.52

#: 시작 속도 범위 (m/s). 정지 출발과 달려 들어오는 경우를 둘 다 본다.
SPEED_JITTER = (0.0, 0.8)

#: LLC 명령 추종 무작위화. 에피소드마다 하나씩 뽑아 `보내는 명령 = 명령*g + b`.
#:
#: **기본이 꺼짐이다. 그리고 그게 실수가 아니라 결론이다.**
#:
#: 2026-08-21에 기본을 (0.75, 1.35) · (±0.05)로 켜 두었다가 평지 직진이 460만
#: 스텝 동안 도달 0.000 으로 죽었다. 학습된 정책을 뜯어 보니 **제자리에 서서
#: 몸만 최대로 세우고** 있었다 -- `height` 0.320(상한), `vx` 0, 200스텝에 4 cm.
#: 보상이 정확히 -1.038 = `TIME_COST * 200` 이었다.
#:
#: 목적함수는 옳았다. 도달 +13, 정지 -1, 넘어짐 -10 이다. 문제는 **아직 못 걷는
#: 정책에게 무작위화가 넘어질 이유를 하나 더 준다**는 것이다. 움직이면 넘어지고
#: 넘어지면 -10 이니, 정지가 기댓값에서 이기는 국소최적에 눌러앉는다.
#:
#: 돌이켜 보면 처음부터 학습해 성공한 판은 전부 무작위화 없이 돌았고, 무작위화를
#: 켜고 성공한 판은 전부 `restore` 였다. 이어받은 판은 **이미 걷는 정책에서
#: 출발**하므로 이 국소최적을 겪지 않는다. 즉 "처음부터 + 무작위화"는 시험된 적이
#: 없는 조합이었고, 그것을 기본값으로 만든 것이 사고였다.
#:
#: 그래서 무작위화도 커리큘럼이다. 회전에서 배운 것과 같은 모양이다.
#:
#:     1) 처음부터   (1.00, 1.00)  (0, 0)        기술을 먼저 배운다
#:     2) 이어받기   (0.90, 1.15)  (+-0.03)
#:     3) 이어받기   (0.75, 1.35)  (+-0.05)      최종 강건성
#:
#: 3단계의 상한 1.35는 버리지 않는다. `spec.TRACKING["yaw"]`의 실측 이득이
#: 0.774라 LLC 가 개선되어 1.0 이 되면 같은 명령이 **1.29배**로 돌아오고,
#: 1.10 으로는 그 변화를 못 덮는다. 다만 **마지막에** 붙일 것이지 처음에 붙일
#: 것이 아니다.
LLC_GAIN = (1.0, 1.0)
LLC_BIAS = (0.0, 0.0)

#: 무작위화 커리큘럼. `train` 루프가 이 순서로 올린다. 숫자를 셀에 옮겨 적지 말 것.
LLC_CURRICULUM = (
    ((1.00, 1.00), (0.00, 0.00)),
    ((0.90, 1.15), (-0.03, 0.03)),
    ((0.75, 1.35), (-0.05, 0.05)),
)


class Task(mjx_env.MjxEnv):
    """랜드 하나 건너기. playground 의 `MjxEnv` 라 brax 래퍼가 그대로 붙는다.

    LLC 는 **동결**이다. 두 층을 같이 학습시키면 불안정하고, 무엇보다 어느 쪽이
    문제인지 둘 다 못 본다.

    `llc_gain` / `llc_bias` 는 명령이 LLC 에 닿기 전에 곱하고 더할 값의 범위다.
    에피소드마다 뽑는다. 이것이 **LLC 가 밑에서 바뀌는 것에 대한 보험**이다 --
    정책이 특정 체크포인트가 아니라 "이 정도로 명령을 따르는 하위 제어기 집단"에
    대해 학습한다. footswing 학습이 지금 돌고 있으므로 가정이 아니라 사실이다.
    비용은 곱셈 하나다.
    """

    def __init__(self, kind, *, checkpoint=None, level_after=0, axis=None,
                 llc_gain=LLC_GAIN, llc_bias=LLC_BIAS, seed=0,
                 yaw_jitter=YAW_JITTER, speed_jitter=SPEED_JITTER,
                 start_shift=0.0, route_jitter=ROUTE_JITTER, **corridor):
        # `kind` 가 여럿이면 차선 복도다. 판이 배치 안에서 섞이므로 **망각이
        # 일어나지 않는다.** 하나면 예전과 완전히 같은 복도가 나온다.
        if isinstance(kind, dict):          # 이미 만들어진 복도를 그대로 받는다
            self.height_np = kind["height"]
            ceiling = kind["ceiling"]
            self.plan = kind["plan"]
        elif isinstance(kind, (list, tuple)):
            assert axis is None, "차선 복도는 축을 RUN_X 로 고정합니다"
            self.height_np, ceiling, self.plan = lands.mixed_corridor(
                kind, level_after=level_after, seed=seed, **corridor)
        else:
            self.height_np, ceiling, self.plan = lands.obstacle_corridor(
                kind, level_after=level_after, axis=axis, seed=seed,
                **corridor)
        self.env = hlc_env.make(terrain=self.height_np,
                                ceiling=ceiling if len(ceiling) else None)
        # 관측용으로도 들고 있는다. **렌더링용 geom 과 같은 배열이어야 한다** --
        # 두 곳에서 따로 만들면 로봇이 보는 천장과 부딪히는 천장이 어긋난다.
        self.ceiling = jnp.asarray(ceiling, jnp.float32).reshape(-1, 6)
        # `MjxEnv.__init__` 을 부르지 않는다 -- config 를 안 들고 있고, dt 는
        # 안쪽 env 에서 유도하는 편이 어긋날 여지가 없다.
        self._ctrl_dt = self.env.dt * REPEAT
        self._sim_dt = self.env.sim_dt

        if checkpoint is None:
            from .. import paths
            checkpoint = paths.llc()
        self.llc_policy = loader.load_policy(
            checkpoint, loader.env_observation_size(self.env))

        self.height = jnp.asarray(self.height_np)
        self.shape = self.plan["shape"]
        # **차선 배열이 단수형을 대신한다.** 한 랜드짜리 복도는 길이 1 이라
        # 예전과 값이 같고, `mixed_corridor` 면 여러 개다. 목표와 종류가
        # 에피소드마다 달라지므로 `self` 가 아니라 `info` 를 타고 다닌다 --
        # `self` 에 두면 배치 전체가 같은 목표를 보게 된다.
        self.lane_start = jnp.asarray(self.plan["lane_start_xy"], jnp.float32)
        self.lane_goal = jnp.asarray(self.plan["lane_goal_xy"], jnp.float32)
        self.n_lanes = int(self.lane_goal.shape[0])
        self.lane_onehot = jnp.asarray(
            np.eye(len(maze.IMPLEMENTED))[np.asarray(self.plan["lane_kind"])],
            jnp.float32)
        # 단도 차선마다 다르다 -- 경사만 1 이고 나머지는 0 이다. 특권 관측에
        # 들어가는 값이라 `self` 에 스칼라로 두면 배치 전체가 같은 값을 본다.
        self.lane_level = jnp.asarray(self.plan["lane_level"], jnp.float32)
        # 길잡이에 넣을 경로 (차선, 점, 2). 직선 복도는 점이 하나(목표)라
        # `[로봇, 목표]` 가 되어 예전과 값이 같다.
        self.lane_route = jnp.asarray(self.plan["lane_route"], jnp.float32)
        # **출발 방향도 차선마다 다르다.** 복도 판은 전부 +x 로 달리므로 0 이지만
        # 미로의 정답지는 세로로도 간다. 0 으로 두면 로봇이 첫 스텝부터 벽을
        # 보고 서고, 초반 보상이 전부 잡음이 된다.
        self.lane_yaw = jnp.asarray(
            self.plan.get("lane_yaw", np.zeros(self.n_lanes, np.float32)),
            jnp.float32)
        # 단 격자. 빠짐 판정이 쓴다. 없으면 0 (단이 없는 판).
        #
        # **이웃까지 본 최소 단**을 쓴다. 자기 타일의 단으로만 재면 **정상적인
        # 한 단 하강을 추락으로 센다** -- `HIGH` 가 0.713 이라 `PIT_MARGIN` 0.40
        # 보다 크기 때문이다. 미로에서 경사 없이 내려가는 것은 단방향 간선이지
        # 사고가 아니다 (`maze.py` 머리말). 실측 -- 언덕 끝 차선의 빠짐이
        # 1.000 인데 그 판의 최저 z 가 0.566 이었다. 빠진 적이 없다.
        #
        # 이웃 최소로 잡으면 "옆 칸의 단으로 내려간 것"과 "어느 이웃보다도
        # 낮은 곳"이 갈린다. 뒤쪽만 도랑 · 절벽이다.
        lv = np.asarray(self.plan.get("level_grid",
                                      np.zeros(self.shape, np.float32)),
                        dtype=np.float32)
        floor = lv.copy()
        for ax in (0, 1):
            floor = np.minimum(floor, np.roll(lv, 1, axis=ax))
            floor = np.minimum(floor, np.roll(lv, -1, axis=ax))
        self.level_grid = jnp.asarray(floor, jnp.float32)
        # 차선 추출 확률. 기본은 균등이라 예전과 값이 같다.
        self.lane_weight = jnp.full((self.n_lanes,), 1.0 / self.n_lanes,
                                    jnp.float32)
        if "lane_weight" in self.plan:
            self.set_lane_weight(self.plan["lane_weight"])
        # 0 번 차선 값. 예전 이름을 쓰는 곳이 아직 있으므로 남겨 둔다.
        self.goal = self.lane_goal[0]
        self.kind_onehot = self.lane_onehot[0]
        self.level_after = float(self.lane_level[0])
        self.gain_range = tuple(llc_gain)
        self.bias_range = tuple(llc_bias)
        self.yaw_jitter = float(yaw_jitter)
        # 출발 지점을 진행 방향으로 미는 거리 (m). **탐색을 열려고 쓴다.**
        #
        # 경사에서 정책이 시도 자체를 안 하는 것을 실측했다. 경사 앞 x 0.95 까지
        # 걸어와 200 스텝을 서 있는다. 옳은 계산이다 -- 도달 확률이 0 이면
        # GOAL_BONUS(+10)는 안 잡히고 넘어짐(-10)만 잡히는데, 가만히 있으면
        # TIME_COST 로 -1.0 이 전부다. 시도의 기댓값이 늘 음수다.
        #
        # 목표 가까이에서 출발시키면 보너스가 실제로 잡혀 부호가 뒤집힌다.
        # 조합을 한 번 찾은 뒤 이 값을 줄여 뒤로 물린다.
        self.start_shift = float(start_shift)
        # 경유점을 옆으로 흔드는 폭 (m). **경로가 정답을 흘리는 것을 막는다.**
        #
        # 경로는 타일 중심을 잇고 통로도 타일 한가운데에 그려진다. 그래서 경유점이
        # 통로 중심을 **정확히** 가리켰다. 실측 -- 경로 기준 y 오프셋 ±0.3 m 까지
        # 터널 바닥이 0.0 이고 ±0.5 부터 벽 0.42 다. 다리도 같다. 즉 경유점만
        # 따라가면 정렬이 공짜였고, 지형 스캔을 볼 이유가 없었다.
        #
        # 실전에서는 자기 위치 추정에 오차가 있어 로봇이 보는 경유점이 그만큼
        # 흔들린다. 그 조건을 학습에 넣는다.
        #
        # 전진 보상은 거의 안 바뀐다 -- `route_progress` 는 정사영이라 **옆으로
        # 벗어난 양이 애초에 안 들어간다.** 실측 0.2 m 지터에서 직선 구간 한 스텝
        # 전진량 차이가 0.5% 였다. 목표점은 안 흔든다. 흔들면 도달 판정이 흔들린다.
        self.route_jitter = float(route_jitter)
        self.speed_jitter = tuple(speed_jitter)
        self.friction = float(np.asarray(self.env._mj_model.geom_friction)[0, 0])

    # ---------- MjxEnv 가 요구하는 것 ----------

    @property
    def xml_path(self) -> str:
        return self.env.xml_path

    @property
    def mj_model(self):
        return self.env.mj_model

    @property
    def mjx_model(self):
        return self.env.mjx_model

    @property
    def action_size(self) -> int:
        return skills.ACTION_SIZE

    def render(self, trajectory, **kw):
        """`trajectory` 는 안쪽 LLC 상태 목록이다. 그리는 것은 안쪽 env 가 한다."""
        return self.env.render(trajectory, **kw)

    # ---------- 관측 ----------

    def _proprio(self, data, command):
        ground = obs.ground_at(self.height, data.qpos[0:2], self.shape)
        return obs.proprio(self.env.get_local_linvel(data),
                           self.env.get_gyro(data),
                           self.env.get_gravity(data),
                           data.qpos[2] - ground, command)

    def _observe(self, data, history, gain_bias, goal, kind_onehot, level,
                 route):
        xy = data.qpos[0:2]
        yaw = path_enc.yaw_from_quat(data.qpos[3:7])
        scan = obs.terrain_scan(self.height, xy, yaw, self.shape)
        ceil = obs.ceiling_scan(self.ceiling, xy, yaw, data.qpos[2])
        # 경로를 길잡이에 넣는다. 직선 복도면 점이 목표 하나라 예전 `[로봇, 목표]`
        # 와 값이 같고, 꺾인 복도면 경유점이 모서리를 돈다. 3단계에서 D* 가 오면
        # `route` 를 만드는 쪽만 바뀌고 여기는 안 바뀐다.
        actor = obs.assemble(
            history, scan, ceil,
            obs.guide(obs.route_polyline(route, xy), xy, yaw, goal))
        contact = (data.sensordata[self.env._floor_found_adr] > 0
                   ).astype(jnp.float32)
        priv = obs.assemble_privileged(
            actor, kind_onehot, level, self.friction,
            contact, gain_bias, data.qvel[0:3])
        return {"state": actor, "privileged_state": priv}

    # ---------- 에피소드 ----------

    def _tile_level(self, xy):
        """그 자리 랜드의 **높이 단**. 빠짐 판정의 기준선이다.

        단 격자가 없으면 0 을 낸다 -- 단이 없는 판에서는 "지면이 0 보다 0.4 m
        아래인가"가 되어 도랑 · 절벽만 잡는다. 그것이 옳은 동작이다.
        """
        ty, tx = self.shape
        ex, ey = tx * maze.TILE, ty * maze.TILE
        xy = jnp.asarray(xy, jnp.float32).reshape(2)
        j = jnp.clip(((xy[0] + ex / 2.0) / maze.TILE).astype(jnp.int32), 0, tx - 1)
        i = jnp.clip(((xy[1] + ey / 2.0) / maze.TILE).astype(jnp.int32), 0, ty - 1)
        return self.level_grid[i, j]

    def set_lane_weight(self, weights) -> None:
        """차선 추출 확률을 바꾼다. **학습을 시작하기 전에 부른다.**

        왜 필요한가 -- 미로 구간 판 12차선 중 11개가 1.000 인데 균등으로 뽑으면
        배치의 92 % 가 이미 아는 것을 반복한다. 실측 -- 그 상태로 16 M 스텝을
        더 돌렸더니 도달이 0.805~0.852 를 오갈 뿐 추세가 없었다.

        문헌의 Prioritized Level Replay 와 같은 생각이다. 다만 두 함정을 피한다.

            실패율로 뽑지 말 것    정말 안 되는 차선 하나가 배치를 독점한다.
                                   실패율 100 % 인데 배울 것이 0 인 상황이다
            되던 것을 빼지 말 것   실측으로 세 번 잊었다 -- 다리만 학습하니 터널이
                                   0.875 에서 0.000, gap 1 만 학습하니 꺾임 앞
                                   다리가 0.625 에서 0.000

        그래서 **바닥과 천장을 둔다** (`weights_from` 참고). 그리고 학습 중에
        갱신하지 않는다 -- 가중치가 jit 경계를 넘어 환경으로 들어가야 하는데,
        그 배관은 값에 비해 비싸다. 학습 전에 한 번 재서 고정하고 다음 학습에서
        다시 잰다.

        **주의 -- 이미 jit 으로 굳은 `reset` 에는 반영되지 않는다.** 추적 시점에
        상수로 잡히기 때문이다. 측정용으로 만든 `Task` 를 그대로 학습에 넘기려면
        측정보다 먼저 불러야 한다.
        """
        w = np.asarray(weights, dtype=np.float64).reshape(-1)
        assert w.shape[0] == self.n_lanes, (
            f"가중치가 {w.shape[0]}개인데 차선은 {self.n_lanes}개입니다")
        assert (w >= 0).all() and w.sum() > 0, "가중치는 음수가 아니어야 합니다"
        self.lane_weight = jnp.asarray(w / w.sum(), jnp.float32)
        # **굳은 함수를 버린다.** `reset` 이 가중치를 상수로 잡으므로, 안 버리면
        # 이 호출이 조용히 무시된다 (`_jitted` 참고).
        self._jit_fns = None

    def lane_of(self, rng):
        """이 seed 가 어느 차선을 낼지. **물리를 안 굴리고 답한다.**

        왜 있는가
        ---------

        차선 하나를 골라 찍으려면 지금까지 **그 차선만 남긴 Task 를 새로
        만들었다.** Task 가 바뀌면 jit 이 다시 컴파일되는데, 실측으로 그 값이
        268 초다. 차선 여덟 개를 찍으면 컴파일만 36 분이다 -- 프로세스를 넷으로
        나눠 돌려도 각자 컴파일을 내므로 병렬 이득이 0 이었다.

        차선은 `reset` 이 seed 에서 뽑는다. 그러니 **seed 만 보고 미리 알 수
        있다.** 원하는 차선이 나오는 seed 를 찾아 쓰면 Task 는 하나로 끝나고
        컴파일도 한 번이다.

        **`reset` 의 쪼개기와 같아야 한다.** 한쪽만 고치면 조용히 어긋나서
        엉뚱한 차선을 찍는다 -- 아래 한 줄이 위 `reset` 첫 줄과 짝이다.
        """
        _, _, _, _, _, _, k6, _ = jax.random.split(rng, 8)
        return int(jax.random.choice(k6, self.n_lanes, p=self.lane_weight))

    def seeds_for_lane(self, lane: int, count: int, *, base=0, tries=4096):
        """그 차선을 내는 seed 들. 못 채우면 찾은 만큼만 준다.

        `lane_weight` 가 균등이면 차선 하나가 나올 확률이 `1/n_lanes` 다.
        944 차선이면 4096 번 뒤져 평균 4 개가 나온다 -- `tries` 를 넉넉히 둔 이유다.
        """
        out = []
        for i in range(int(tries)):
            k = jax.random.PRNGKey(int(base) + i)
            if self.lane_of(k) == int(lane):
                out.append(k)
                if len(out) >= int(count):
                    break
        return out

    def reset(self, rng) -> mjx_env.State:
        """**진입 조건을 흔들어서** 시작한다. 이 무작위화가 1단계의 핵심이다."""
        rng, k1, k2, k3, k4, k5, k6, k7 = jax.random.split(rng, 8)
        # **차선을 먼저 뽑는다.** 이것이 배치에 판을 섞는 유일한 자리다.
        # 가중치를 안 주면 균등이라 예전과 값이 같다. 근거는 `set_lane_weight` 에.
        lane = jax.random.choice(k6, self.n_lanes, p=self.lane_weight)
        goal = self.lane_goal[lane]
        kind_onehot = self.lane_onehot[lane]
        level = self.lane_level[lane]
        route = self.lane_route[lane]
        route = _jitter_route(route, k7, self.route_jitter)
        j = self.plan["jitter_xy"]
        # 밀어주는 방향은 **그 차선의 진행 방향**이다. 복도 판은 요각이 0 이라
        # 예전과 값이 같고(cos 0 = 1, sin 0 = 0), 미로에서만 달라진다.
        heading = jnp.array([jnp.cos(self.lane_yaw[lane]),
                             jnp.sin(self.lane_yaw[lane])], jnp.float32)
        xy = (self.lane_start[lane] + self.start_shift * heading
              + jax.random.uniform(k1, (2,), minval=-j, maxval=j))
        # **몸통을 그 자리 지형 높이만큼 띄운다.** keyframe 의 z 는 평지 기준이라
        # 경사 중턱에서 출발시키면 땅에 박힌 채로 시작한다. 평지 출발이면 0 이라
        # 다른 랜드는 동작이 안 바뀐다.
        ground = obs.ground_at(self.height, xy, self.shape)
        yaw = self.lane_yaw[lane] + jax.random.uniform(
            k2, (), minval=-self.yaw_jitter, maxval=self.yaw_jitter)
        speed = jax.random.uniform(k3, (), minval=self.speed_jitter[0],
                                   maxval=self.speed_jitter[1])
        gain_bias = jnp.stack([
            jax.random.uniform(k4, (), minval=self.gain_range[0],
                               maxval=self.gain_range[1]),
            jax.random.uniform(k5, (), minval=self.bias_range[0],
                               maxval=self.bias_range[1]),
        ])

        llc = self.env.reset_at(rng, xy=xy, yaw=yaw, speed=speed,
                                z_offset=ground)
        command = action.to_command(jnp.zeros(spec.DIM))
        history = jnp.tile(self._proprio(llc.data, command), (obs.STACK, 1))
        # **경로를 따라 남은 거리.** 직선거리로 재면 벽 쪽으로 사선을 그어도
        # 매 스텝 보상이 오른다 -- 꺾인 복도에서 실제로 그 궤적이 나왔다.
        # 경로가 한 점이면 직선거리와 같아서 직선 복도는 값이 안 바뀐다.
        dist = obs.route_progress(route, llc.data.qpos[0:2])
        info = {
            "llc": llc,
            "key": rng,
            "history": history,
            "command": command,
            "gain_bias": gain_bias,
            "goal": goal,
            # **측정용이다.** 관측에는 안 들어간다 (서명 불변). 도달률은 차선
            # 평균이라 한 차선이 0.91 로 떨어져도 전체는 0.977 로 읽힌다 --
            # 실제로 그렇게 읽고 "경사도 잘 된다"로 잘못 판정했다.
            "lane": lane,
            "kind_onehot": kind_onehot,
            "level": level,
            "route": route,
            "phi": -dist,
            # **이 값은 우리 것이 아니다.** brax 의 `EpisodeWrapper` 가 소유하고
            # 스텝마다 `+ action_repeat` 한다. 리셋에서 자리만 만들어 두고
            # 여기서는 절대 올리지 않는다.
            #
            # 예전에 `step` 에서 같이 +1 했다. 그러면 스텝마다 2씩 올라
            # **에피소드가 상한의 절반에서 잘린다.** 실측 -- episode_length 200
            # 인데 여덟 환경이 전부 100 스텝에서 done 이었다. 판이 짧을 때는
            # 안 드러나다가, 도달까지 109 스텝 걸리는 판에서 도달률이 0.938 대
            # 0.008 로 갈렸다.
            "steps": jnp.zeros((), jnp.int32),
            "skill": skills.initial(),
        }
        return mjx_env.State(
            data=llc.data,
            obs=self._observe(llc.data, history, gain_bias, goal,
                              kind_onehot, level, route),
            reward=jnp.zeros(()), done=jnp.zeros(()),
            # 키 구성이 `step` 과 **같아야 한다.** brax 래퍼가 metrics 를 캐리로
            # 들고 다니므로 하나라도 다르면 pytree 구조가 어긋난다.
            metrics={"도달": jnp.zeros(()), "넘어짐": jnp.zeros(()),
                     "빠짐": jnp.zeros(()),
                     "목표거리": dist, "발산": jnp.zeros(())},
            info=info)

    def step(self, state: mjx_env.State, act) -> mjx_env.State:
        """HLC 한 스텝 = LLC `REPEAT` 스텝. 명령은 그 동안 고정이다."""
        info = dict(state.info)
        gain_bias = info["gain_bias"]
        command = action.to_command(skills.command_of(act))
        # 이득 · 편향은 **속도 축만** 흔들고, LLC 에 닿는 값에만 적용한다.
        # 전체에 곱하면 height 나 step_freq 같은 절대값 축까지 흔들려서
        # 기본 자세가 바뀌고 미학습 축이 UNTRAINED_HOLD 를 벗어난다.
        sent = action.perturb(command, gain_bias[0], gain_bias[1])

        # 특수 동작 자리. 지금은 전부 꺼져 있어 항상 걷기가 나온다. 켜지면 여기서
        # `skill.active` 에 따라 정책을 갈아 끼운다 -- 걷기 · 점프 · 기립 세
        # 체크포인트의 관측 · 행동 인터페이스가 같아 파라미터만 바꾸면 된다.
        skill = skills.update(info["skill"], act,
                              self.env.get_gravity(state.data)[2],
                              jnp.zeros(skills.N_GATES))

        # **명령을 스캔 밖에서 한 번만 넣는다.** 안에서 매번 `with_command` 를
        # 부르면 관측을 두 번 조립하게 된다 -- `with_command` 가 한 번, 바로 뒤
        # `env.step` 이 또 한 번. LLC 서브스텝 5 개면 HLC 스텝당 10 회다.
        #
        # 한 번만 넣어도 되는 이유는 명령이 `info` 에 살아 있기 때문이다.
        # `steps_until_next_cmd` 를 10 억으로 막아 두었고 `sample_command` 도
        # 그대로 돌려주도록 덮어써서, `env.step` 이 만드는 관측 꼬리에 같은 명령이
        # 그대로 실린다 (3 스텝까지 실측 확인).
        first = self.env.with_command(info["llc"], sent)

        def one(carry, _):
            llc, key = carry
            key, sub = jax.random.split(key)
            a, _ = self.llc_policy(llc.obs, sub)
            llc = self.env.step(llc, a)
            return (llc, key), llc.done

        (llc, key), dones = jax.lax.scan(
            one, (first, info["key"]), None, length=REPEAT)
        fell = jnp.max(dones)

        history = jnp.concatenate(
            [self._proprio(llc.data, command)[None], info["history"][:-1]])
        dist = obs.route_progress(info["route"], llc.data.qpos[0:2])
        dist = jnp.nan_to_num(dist, nan=DIVERGE_DIST, posinf=DIVERGE_DIST)
        phi = -dist
        # **도달은 목표까지 직선거리로 판정한다.** 경로상 남은 거리로 판정하면
        # 경로에서 멀리 벗어난 채로 경로 끝 근처에 있어도 도달이 된다.
        straight = jnp.linalg.norm(llc.data.qpos[0:2] - info["goal"])
        straight = jnp.nan_to_num(straight, nan=DIVERGE_DIST,
                                  posinf=DIVERGE_DIST)
        reached = (straight < GOAL_RADIUS).astype(jnp.float32)
        # **전진량을 자른다.** 물리가 터지면 이 항이 그대로 폭주하고 가치 목표를
        # 오염시킨다. 정상 최대가 0.1 m 라 0.5 는 학습에 안 걸린다. 위 상수 참고.
        progress = jnp.clip(phi - info["phi"], -MAX_PROGRESS, MAX_PROGRESS)

        # **빠짐.** 그 타일의 단보다 한참 아래로 내려섰으면 도랑 · 절벽이다.
        # 네 발로 내려서면 넘어짐이 안 잡히고, 거기서 200 스텝을 보낸 뒤
        # 시간 초과로 기록된다 -- 위 `PIT_MARGIN` · `STAND_CLEAR` 주석 참고.
        xy_now = llc.data.qpos[0:2]
        ground = obs.ground_at(self.height, xy_now, self.shape)
        nominal = self._tile_level(xy_now) * maze.HIGH
        sunk = ((ground < nominal - PIT_MARGIN)
                & (llc.data.qpos[2] - ground < STAND_CLEAR))
        sunk = sunk.astype(jnp.float32)
        # **빠짐은 지표일 뿐이다. 종료도 벌점도 아니다.**
        #
        # 처음에는 종료 조건으로 뒀는데 판정을 두 번 틀렸고, 그때마다 **오판이
        # 곧 학습 손상**이었다 -- 경사를 횡단하면 좌우가 최대경사라 걸렸고,
        # 타일 단으로 재니 정상적인 한 단 하강(0.713 > 0.40)이 전부 걸렸다.
        #
        # 국소 기하만으로는 "한 단 내려갔다"와 "구덩이에 빠졌다"가 구별되지
        # 않는다. 둘 다 주변보다 낮은 곳이고, 차이는 거기서 길이 이어지느냐는
        # **의미**에 있다. 보행 RL 이 종료를 몸통 접촉 · 자세 · 몸통 높이 같은
        # **로봇 상태**로만 두는 이유가 이것이다.
        #
        # 그래서 표에만 남긴다. 도랑에서 못 나오면 시간이 흘러 끝나고, 그 비용은
        # `TIME_COST` 가 이미 물린다. 종료로 아끼는 것은 계산량뿐이다.
        failed = fell

        reward = (SHAPING * progress                     # 전진량. gamma 없음
                  + GOAL_BONUS * reached
                  + FALL_PENALTY * failed
                  - TIME_COST
                  - JERK_COST * jnp.sum(jnp.abs(command - info["command"])))

        # `steps` 를 여기서 올리지 않는다. 위 `reset` 의 주석 참고.
        info.update(llc=llc, key=key, history=history, command=command,
                    phi=phi, skill=skill)
        # **metrics 를 새로 만들지 않는다.** brax 의 `EpisodeWrapper` 가 자기 키를
        # 여기 얹어 두는데, 통째로 갈아 끼우면 그 키가 사라져 `lax.scan` 의 캐리
        # 구조가 어긋난다. 들어온 dict 를 갱신한다. `info` 도 같은 이유로 그렇다.
        metrics = dict(state.metrics)
        # `발산` 은 에피소드 동안 합산되므로 0 이 아니면 폭주가 실제로 있었다는
        # 뜻이다. 보상에는 안 쓴다 -- 재는 것과 벌하는 것을 섞지 않는다.
        metrics.update({"도달": reached, "넘어짐": fell, "빠짐": sunk,
                        "목표거리": straight,
                        "발산": (dist > DIVERGE_DIST).astype(jnp.float32)})

        # 시간 초과를 여기서 세지 않는다. `EpisodeWrapper` 가 `episode_length` 로
        # 자르고 그쪽이 부트스트랩을 옳게 처리한다. 여기서 done 을 세우면
        # 시간 초과가 실패처럼 학습된다.
        return state.replace(
            data=llc.data,
            obs=self._observe(llc.data, history, gain_bias, info["goal"],
                              info["kind_onehot"], info["level"],
                              info["route"]),
            reward=reward, done=jnp.clip(failed + reached, 0.0, 1.0),
            metrics=metrics, info=info)


# ---------- 눈으로 보기 ----------

def _jitted(task):
    """`(reset, step)` 을 **task 마다 한 번만** 컴파일한다.

    전에는 `rollout` 이 부를 때마다 `jax.jit(task.step)` 을 새로 만들었다.
    바운드 메서드라 매번 다른 래퍼가 나오고 캐시가 비어 있어, 판마다 컴파일을
    통째로 다시 냈다.

    실측 (8x16, 45차선, 로컬 CPU) -- 판당 330 초. 41 스텝에 죽은 판과 227 스텝을
    완주한 판이 **둘 다 345 초** 였다. 시뮬레이션이 아니라 컴파일이라는 뜻이다.
    같은 영상 작업에서 렌더는 장당 0.09 초로 전체의 0.1 % 도 안 됐다.

    `debug_video` 가 실패한 판을 찾느라 12 판을 굴리므로 여기가 비용의 전부다.

    **주의 --** `reset` 이 `self.lane_weight` 를 추적 시점 상수로 잡는다.
    그래서 `set_lane_weight` 가 이 캐시를 버린다.
    """
    fns = getattr(task, "_jit_fns", None)
    if fns is None:
        fns = (jax.jit(task.reset), jax.jit(task.step))
        try:
            task._jit_fns = fns
        except AttributeError:      # __slots__ 인 판이면 캐시를 포기한다
            pass
    return fns


def rollout(task, policy, rng, nsteps=MAX_STEPS, record=False):
    """한 에피소드. **영상 없이 학습을 판단하지 않는다.**

    이 프로젝트에서 표만 보고 틀린 판정을 두 번 냈다 (`hlc/measure.py` 참고).
    보상 곡선은 무엇이 일어났는지 말해 주지 않는다 -- 목표에 닿았는지, 지형을
    뚫고 미끄러졌는지, 옆으로 굴러서 거리가 줄었는지 구분이 안 된다.

    `policy` 는 `(obs, key) -> (ACTION_SIZE,)` 원시 출력이다. 학습 전에는 상수를
    넣어 배선을 확인하고, 학습 중에는 정책을 넣어 주기적으로 본다.
    """
    reset, step = _jitted(task)
    st = reset(rng)
    # **어느 차선이었는지 남긴다.** 영상 하나는 에피소드 하나라 차선도 하나인데,
    # 요약이 말해 주지 않으면 무엇을 본 것인지 알 수 없다.
    lane = int(np.argmax(np.asarray(st.info["kind_onehot"])))
    frames = [st.info["llc"]] if record else None
    total = 0.0
    track = []
    for i in range(nsteps):
        rng, sub = jax.random.split(rng)
        st = step(st, policy(st.obs, sub))
        total += float(st.reward)
        track.append(np.asarray(st.data.qpos[0:3]))
        if record:
            frames.append(st.info["llc"])
        if bool(st.done):
            break
    track = np.asarray(track)
    return {
        "스텝": i + 1,
        "랜드": maze.NAMES.get(lane, lane),
        "보상합": round(total, 3),
        "도달": bool(st.metrics["도달"]),
        "넘어짐": bool(st.metrics["넘어짐"]),
        "목표까지_m": round(float(st.metrics["목표거리"]), 3),
        "최저z": round(float(track[:, 2].min()), 3),
    }, frames


def debug_video(task, policy, filename, rng=None, nsteps=MAX_STEPS,
                tries=6, prefer="fail", stride=2, lane=None):
    """**실패한 판을 골라** 영상으로 남긴다.

    seed 를 고정해 한 판만 찍으면 대표성이 없다 -- 턱 대조군이 도달률 0.812 인데
    seed 0 이 하필 실패 쪽이라, 영상만 보면 "못 넘는다"로 읽혔다. 반대 경우가 더
    나쁘다. 성공한 판만 보다가 나머지 20%가 왜 실패하는지 영영 모른다.

    성공은 표가 이미 말해 준다. 영상이 답해야 하는 것은 **무엇이 잘못되는가**다.
    그래서 `tries` 판을 찍지 않고 돌려 보고 실패한 첫 판을 골라 그 seed 로만
    다시 돌리며 녹화한다. 결정적이라 잰 것과 찍은 것이 같은 주행이다.

    렌더가 물리보다 훨씬 비싸므로(프레임당 1.3 초) 탐색은 사실상 공짜다.
    전부 성공하면 마지막 판을 남긴다 -- 그때는 볼 것이 없다는 뜻이기도 하다.

    `prefer`
        "fail"      실패한 첫 판. 기본값이고, 학습 중 자동 녹화가 쓴다
        "성공"      성공한 첫 판. 팀에 보여 줄 때나 "되긴 되는가"를 확인할 때
        "시간초과"  안 넘어지고 못 간 판. **흔한 실패를 보려면 이것이다**
        "넘어짐"    실제로 넘어진 판
        그 외        찾지 않고 마지막 seed 를 그냥 쓴다

    "fail" 이 넘어짐과 시간초과를 **구분하지 않는 것이 함정이다.** 실측 -- 어느
    차선이 시간초과 0.875 · 넘어짐 0.125 인데 "fail" 이 골라 준 것이 넘어진
    판이었다. 12.5 % 를 보고 87.5 % 를 설명할 뻔했다. 둘은 처방이 정반대다.

    `lane`
        찍고 싶은 차선. **Task 를 그대로 두고 seed 로 고른다** (`seeds_for_lane`).
        전에는 그 차선만 남긴 Task 를 따로 세웠는데, Task 가 바뀌면 jit 이 다시
        컴파일되고 실측으로 그것이 268 초다. 차선 여덟 개면 컴파일만 36 분이라
        영상 비용의 거의 전부였다 -- 물리도 렌더도 아니었다 (렌더는 64x64 에서도
        프레임당 0.016 초다).
    """
    from .measure import save_video

    if lane is None:
        base = jax.random.PRNGKey(0) if rng is None else rng
        keys = jax.random.split(base, int(tries))
    else:
        keys = task.seeds_for_lane(int(lane), int(tries))
        assert keys, (f"차선 {lane} 을 내는 seed 를 못 찾았습니다. 차선이 "
                      f"{task.n_lanes} 개면 seeds_for_lane 의 tries 를 키우세요")
    pick, found = keys[-1], False

    def matches(s):
        if prefer == "fail":
            return not s["도달"]
        if prefer == "성공":
            return bool(s["도달"])
        if prefer == "넘어짐":
            return bool(s["넘어짐"])
        if prefer == "시간초과":
            # 안 넘어지고 못 갔다. 상한을 다 썼는지는 스텝 수로 본다.
            return not s["도달"] and not s["넘어짐"]
        return None

    if matches({"도달": False, "넘어짐": False}) is not None:
        for k in keys:
            summary, _ = rollout(task, policy, k, nsteps, record=False)
            if matches(summary):
                pick, found = k, True
                break

    summary, frames = rollout(task, policy, pick, nsteps, record=True)
    # 찾던 판을 실제로 찾았는가. 못 찾았으면 마지막 seed 를 그냥 찍은 것이다.
    summary["찾음"] = bool(found)
    summary["고른것"] = prefer if bool(found) else "(못 찾음)"
    # stride 2 -> 5 Hz. 렌더가 프레임당 1.3 초라 여기가 비용의 전부다.
    # 무엇이 잘못됐는지 보는 데는 5 Hz 로 충분하다.
    save_video(task.env, frames, filename, fps=10, stride=stride)
    return summary

def reward_sanity(dist0=4.0, reach_steps=85, stall_steps=200, fall_step=20,
                  verbose=True):
    """세 시나리오의 에피소드 보상을 손으로 더해 본다. **학습 전에 부른다.**

    보려는 것은 딱 하나 -- **순서가 맞는가.**

        도달  >  정지  >  넘어짐

    이 순서가 깨지면 PPO 는 깨진 쪽으로 수렴한다. 첫 학습에서 정지가 도달보다
    쌌던 것은 아니지만 정지가 **양수**였고, 미학습 정책에게는 확정 양수가
    불확실한 큰 양수를 이긴다. 그래서 절대 부호도 같이 본다.

    물리를 안 돌리므로 즉시 끝난다. 보상 계수를 만질 때마다 부르면 된다.
    """
    def episode(phis, terminal_goal=0.0, terminal_fall=0.0):
        r = sum(b - a for a, b in zip(phis[:-1], phis[1:]))
        return (SHAPING * r + GOAL_BONUS * terminal_goal
                + FALL_PENALTY * terminal_fall
                - TIME_COST * (len(phis) - 1))

    def phi(d):
        return -d

    reach = [phi(dist0 * (1 - i / reach_steps) + GOAL_RADIUS * (i / reach_steps))
             for i in range(reach_steps + 1)]
    stall = [phi(dist0)] * (stall_steps + 1)
    fall = [phi(dist0)] * (fall_step + 1)

    out = {
        "도달": round(episode(reach, terminal_goal=1.0), 2),
        "정지": round(episode(stall), 2),
        "넘어짐": round(episode(fall, terminal_fall=1.0), 2),
    }
    out["순서정상"] = out["도달"] > out["정지"] > out["넘어짐"]
    out["정지가_양수"] = out["정지"] > 0
    if verbose:
        print(f"  보상 점검 (거리 {dist0} m 기준)")
        for k in ("도달", "정지", "넘어짐"):
            print(f"    {k:5s} {out[k]:+8.2f}")
        print(f"    순서 도달 > 정지 > 넘어짐 : {out['순서정상']}")
        print(f"    정지가 양수인가 (나쁨)    : {out['정지가_양수']}")
    return out

