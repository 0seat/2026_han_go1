"""특수 동작 층 -- **점프와 복구가 들어올 자리를 지금 비워 둔다.**

둘 다 아직 안 쓴다. 그런데 자리를 지금 만드는 이유는, 나중에 만들면 **행동 공간과
관측 배치가 바뀌어 재학습**이기 때문이다. `spec.TRAINED` 마스크와 같은 논리다.

성격이 다르다
-------------

    복구   방아쇠가 자명하다.  몸통 중력 벡터 하나로 판정된다
           HLC 명령이 필요 없다.  오히려 HLC 를 **선점**해야 한다
           지연이 밀리초 단위여야 한다
           -> 학습된 판단이 아니라 **규칙**이다. 게이트를 쓰지 않는다

    점프   앞에 무엇이 있는지 알아야 한다.  지도 지식이므로 HLC 의 것
           0.5 초를 되돌릴 수 없이 건다.  전략적 결정
           -> HLC 가 **게이트**로 누른다

그래서 복구는 행동 공간을 안 건드리고, 점프만 게이트 한 칸을 쓴다.

권한은 위에서 아래로
--------------------

```
    감독기 (규칙, 학습 안 함)      복구 · 비상정지.  아래를 선점한다
      HLC (PPO, 10 Hz)            연속 11축 + 게이트 N칸
        스킬 (사전학습, 동결)      점프.  발동되면 걸쇠가 걸린다
          LLC (동결)              걷기
```

복구가 "LLC 안"이 아니라 **맨 위**인 것에 주의. 자기수용감각만 쓰고 빠르다는 점은
LLC 를 닮았지만, HLC 의 명령을 무시하고 제어권을 뺏으므로 권한은 최상이다.

인터페이스가 이미 같다
----------------------

드라이브의 세 체크포인트가 **바이트 단위로 같은 인터페이스**다.

```
                  state  privileged  action  net           dist
    phase18 걷기   56     131         12      512/256/128   tanh_normal
    jump 0.5m      56     131         12      512/256/128   tanh_normal
    getup v2       56     131         12      512/256/128   tanh_normal
```

그래서 스킬 전환은 **정책 파라미터를 갈아 끼우는 것**뿐이다. 글루 코드가 없다.
빨간 버튼의 제일 비싼 부분이 이미 값이 치러져 있다.

진짜 어려운 것은 이산 출력이 아니라 **약속**이다
------------------------------------------------

숫자 하나를 이산으로 내는 것은 쉽다. 어려운 것은 이것이다 -- HLC 가 10 Hz 로
판단하는데 점프가 0.6 초 걸리면 비행 중에 판단 지점이 6번 있다. 3번째에서 마음을
바꾸면 넘어진다. 반쯤 하다 마는 점프는 안 하느니만 못하다.

그래서 **걸쇠**를 건다. 발동되면 `max_steps` 동안 HLC 의 게이트 출력을 무시하고
스킬이 LLC 를 직접 몬다. 옵션 프레임워크의 종료 조건이다.

버튼은 잘 죽는다
----------------

예상해야 하는 실패 모드다. 초기화 직후 정책은 게이트를 무작위로 누르고, 무작위
시점의 점프는 거의 다 넘어진다. 그러면 "절대 안 누른다"로 수렴하고 게이트가 영영
안 열린다. 흔한 국소최적이다. 막는 방법 셋을 다 쓴다.

    발동 집합을 규칙으로 좁힌다   "정면 2 m 안에 틈, 정렬 오차 15도 이내"에서만
    스킬을 따로 학습해 동결한다   "어떻게 뛰나"와 "언제 뛰나"를 분리한다
    스킬만이 답인 커리큘럼        우회로가 없는 틈 미로

이산 분포를 쓰지 않는 이유
--------------------------

게이트를 베르누이/범주형으로 두면 brax 의 기본 `tanh_normal` 을 못 쓰고 분포를
직접 만들어야 한다. 대신 **연속 출력 하나를 문턱으로 자른다** (`raw > 0` 이면
발동). 탐색은 가우시안 잡음이 해 주고, 표준 PPO 가 그대로 돈다. 기울기 신호가
베르누이보다 약하지만, 커스텀 분포를 들이는 값보다 그 손해가 싸다.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import struct

from ..llc import spec

#: 몸통 좌표계 중력의 z 성분이 이보다 크면 넘어진 것으로 본다.
#: 똑바로 서 있으면 -1.0, 옆으로 누우면 0 근처, 뒤집히면 +1.0 이다.
FALLEN_GRAVITY_Z = -0.5

#: 넘어짐 판정에 필요한 연속 HLC 스텝. 한 틱의 튐으로 복구가 발동하지 않게 한다.
FALLEN_HOLD = 3


@struct.dataclass
class Skill:
    """특수 동작 하나의 사양. **값을 바꾸는 것만으로 켜진다.**"""

    name: str
    #: 행동 벡터에 **게이트 칸을 잡아먹는가.**
    #:
    #: 방아쇠와 별개다. 이 값이 `N_GATES` 와 `ACTION_SIZE` 를 정하고 그 둘이
    #: 관측 서명에 들어가므로, **한 번 True 로 나간 슬롯은 못 뺀다** -- 빼면
    #: 학습한 파라미터를 전부 못 읽는다. 규칙 발동으로 바꾸더라도 슬롯은 남긴다.
    gated: bool
    #: 무엇이 발동시키는가. `"rule"` 이면 지형 · 자세가, `"gate"` 면 HLC 출력이.
    #:
    #: 감독기는 규칙 기반이다 (이 파일 머리말의 권한 표). 점프도 규칙으로 둔다 --
    #: "언제 뛰나"를 PPO 로 배우려면 틈만 있는 전용 커리큘럼이 필요한데, 지금
    #: 미로의 정답 경로에는 도랑이 한 칸도 없어서 (`maze.PLACED` 에 GAP 이 없다)
    #: 게이트가 영원히 안 눌린다. 배울 신호가 없는 것을 배우게 두지 않는다.
    trigger: str
    enabled: bool       #: 지금 쓰는가. 체크포인트가 있어도 꺼둘 수 있다
    max_steps: int      #: 걸쇠 길이 (HLC 스텝). 이 동안 HLC 출력을 무시한다
    phase: str          #: 드라이브의 체크포인트 폴더. None 이면 아직 없다
    run: str


#: 등록된 특수 동작. **순서가 게이트 슬롯 순서다.**
#:
#: 지금은 둘 다 `enabled=False` 다. 그래서 이 파일은 현재 아무 일도 하지 않는다.
#: 그것이 의도다 -- 켜는 날 바꾸는 것이 `enabled` 한 글자여야 한다.
#:
#: 켜기 전에 해야 할 일
#:     복구   `hlc/measure.py` 로 기립 성공률을 우리 지형에서 재확인.
#:            드라이브 기록은 평지·경사 20도에서 0.80 이다
#:     점프   틈만 있는 전용 환경에서 스킬을 재학습.  지금 체크포인트는
#:            **평지 착지만** 배웠고 올라선 면에 착지해 본 적이 없다
REGISTRY: tuple[Skill, ...] = (
    Skill(name="getup", gated=False, trigger="rule", enabled=True, max_steps=40,
          phase="go1_getup_11d_floorsafe_terrain_friction_curriculum_optuna_100m_v2",
          run="20260814_071905"),
    # 게이트 슬롯은 남기고(서명) 방아쇠만 규칙이다. 위 `trigger` 주석 참고.
    Skill(name="jump", gated=True, trigger="rule", enabled=False, max_steps=6,
          phase="phase_jump_v8", run="20260826_130405"),
)

#: **기립을 켰다. 등록된 08-14 판이 실측에서 유일하게 일어선다.**
#:
#: 최신 계보(`getup_stage1~3L` · `v8` · `v9_1`, 08-24 이후)는 관측을 state 42 ·
#: privileged 91 로 바꿔 갈라져 나갔다. 걷기 · 점프가 56 · 131 이라 파라미터
#: 교체로는 못 들어간다. 그래서 56 · 131 인 여덟 개를 같은 잣대로 쟀다.
#:
#: 평지 복도에서 공중에서 굴려 떨어뜨리고 250 스텝. 기립 = 몸통 중력 z < -0.8
#: 이면서 몸통 z > 0.22 인 상태를 마지막 50 스텝 유지.
#:
#:     정책                    90도 눕힘   180도 눕힘
#:     걷기 (대조)                 0 %         0 %
#:     floorsafe v2 (등록)       100 %       100 %      <- 이것만 된다
#:     anti_penetration            0 %         0 %
#:     anti_pen finetune           0 %         0 %
#:     grounded_multi_pose         0 %         0 %
#:     ppo_optuna v1               0 %         0 %
#:     pose_curriculum v3          0 %         0 %
#:
#: **주의 -- 자세를 만드는 방법이 결과를 정한다.** 처음에는 서 있는 관절 자세
#: 그대로 몸통만 굴렸는데, 그러면 다리가 지형을 파고든 채 시작해 몸통 z 가
#: -0.24 까지 꺼진다. 그 상태로는 전부 0 % 로 나왔다. 공중에서 떨어뜨려
#: 가라앉히는 쪽이 실제 낙상에 가깝다.
#:
#: **점프도 꺼 둔다. 배선은 됐는데 스킬이 안 뛴다.**
#:
#: `phase_jump_v8` (08-26) 은 인터페이스가 맞고 (`hlc_compatible: true`,
#: 56 · 131) 배선도 확인했다 -- 틈 1.8 m 앞에서 규칙이 켜지고 걸쇠가 걸리고
#: 제어권이 넘어간다. 그런데 **정책 자체가 도약을 안 한다.**
#:
#: 평지 단독 실측 (200 스텝, 명령 여섯 가지):
#:
#:     걷기 phase18   z 최고 0.347  최저 0.274  200 스텝 버팀
#:     jump v8        z 최고 0.344  최저 0.057  74 스텝에 넘어짐
#:
#: 시작 높이 0.344 를 한 번도 안 넘고 0.057(몸통이 바닥)까지 내려앉는다.
#: 그리고 **명령을 안 본다** -- vx 0.0 · 0.4 · 0.6 · 1.0 과 height 0.22 · 0.32 이
#: 소수점까지 같은 궤적을 낸다.
#:
#: 훈련 기록의 `best_reward` (수직 136.9, 전진 328.5) 는 `dense vz/flight_time/
#: squat_ext` 보상이라, 웅크림 항만 챙기고 도약을 안 해도 점수가 난다. 그리고
#: 각 스테이지 30M 을 `elapsed_h` 0.18 시간에 돌았다고 적혀 있는데 그 속도는
#: 나올 수 없다 -- **기록된 스텝 수를 믿지 말 것.**
#:
#: 켜기 전에 할 일 -- 호환되는 다른 점프 체크포인트 여섯 개를 같은 잣대로 재서
#: 실제로 뛰는 것을 고른다. `jump_commanded_dx010_050_...` 은 이름상 명령으로
#: 거리를 조절하도록 학습했으므로 첫 후보다.
#:
#: **주의 -- `max_steps=6` 은 확인된 값이 아니다.** 0.6 초라는 뜻인데, v8 의
#: 실제 체공 시간을 재서 맞춰야 한다. 짧으면 착지 전에 제어권이 돌아오고,
#: 길면 착지 뒤에도 점프 정책이 걷기를 막는다.

#: 게이트가 필요한 스킬만. **행동 벡터의 꼬리 길이가 이것으로 정해진다.**
GATED: tuple[Skill, ...] = tuple(s for s in REGISTRY if s.gated)

#: 게이트 칸 수. 스킬을 꺼도 줄지 않는다 -- 줄면 신경망 출력층이 바뀐다.
N_GATES = len(GATED)

#: HLC 행동 벡터 크기. 11축 명령 + 게이트.
ACTION_SIZE = spec.DIM + N_GATES

#: 켜진 게이트만 1. 꺼진 칸의 출력은 버린다 -- 탐색 잡음만 만들고 보상에 영향이
#: 없으면 엔트로피가 높은 채로 남아 다른 축의 탐색을 방해한다.
GATE_ENABLED = jnp.asarray(
    [1.0 if (s.enabled and s.trigger == "gate") else 0.0 for s in GATED],
    dtype=jnp.float32)

#: 발동 문턱. 원시 출력이 이보다 크면 누른 것으로 본다.
GATE_THRESHOLD = 0.0


#: 점프 발동을 허용하는 지형 조건.
#:
#: **규칙으로 좁히지 않으면 게이트가 죽는다.** 초기화 직후 정책은 아무 데서나
#: 누르고, 무작위 시점의 점프는 거의 다 넘어진다. 그러면 "절대 안 누른다"로
#: 수렴한다 (이 파일 머리말).
#:
#: 조건은 둘이다. 앞이 꺼져 있고, 그 너머가 다시 올라와 있어야 한다.
#:
#:     틈      1 m 와 2 m 앞의 지면이 발밑보다 `GAP_DROP` 이상 낮다
#:     착지    3 m 앞이 발밑과 `LAND_TOL` 안으로 같은 높이다
#:
#: 두 번째가 없으면 절벽에서도 뛴다. `maze.GAP_DEPTH` 가 0.5 이고 정상 요철은
#: 거침 0.06 · 돌 0.14 라, 0.35 면 도랑만 걸리고 요철은 안 걸린다.
JUMP_LOOKAHEAD = (1.0, 2.0)
JUMP_LANDING = 3.0
GAP_DROP = 0.35
LAND_TOL = 0.30


def jump_allow(height, xy, yaw, shape) -> jax.Array:
    """게이트별 발동 허용 (N_GATES,). `update` 의 `allow` 로 넘긴다.

    **jit 안에서 돈다.** 지형 조회는 `obs.ground_at` 과 같은 계산이다.
    """
    from . import obs as _obs

    xy = jnp.asarray(xy, jnp.float32).reshape(2)
    fwd = jnp.array([jnp.cos(yaw), jnp.sin(yaw)], jnp.float32)
    here = _obs.ground_at(height, xy, shape)
    ahead = jnp.stack([_obs.ground_at(height, xy + fwd * d, shape)
                       for d in JUMP_LOOKAHEAD])
    land = _obs.ground_at(height, xy + fwd * JUMP_LANDING, shape)

    gap = jnp.min(ahead) < here - GAP_DROP
    ok_land = jnp.abs(land - here) < LAND_TOL
    fire = (gap & ok_land).astype(jnp.float32)
    # 게이트 슬롯 순서는 `GATED` 순서다. 지금은 점프 하나뿐이다.
    return jnp.asarray([fire if s.name == "jump" else 0.0 for s in GATED],
                       jnp.float32).reshape(N_GATES)

@struct.dataclass
class SkillState:
    """지금 누가 로봇을 몰고 있는가.

    active -- -1 이면 걷기(LLC). 0 이상이면 `REGISTRY` 의 인덱스.
    left   -- 걸쇠에 남은 HLC 스텝. 0 이 되면 제어권이 HLC 로 돌아온다.
    fallen -- 넘어진 상태가 몇 스텝 이어졌나. 복구 방아쇠가 쓴다.
    """

    active: jax.Array
    left: jax.Array
    fallen: jax.Array


def initial() -> SkillState:
    """걷기 상태. 아무 스킬도 안 돌고 있다."""
    return SkillState(active=jnp.asarray(-1, jnp.int32),
                      left=jnp.zeros((), jnp.int32),
                      fallen=jnp.zeros((), jnp.int32))


def gates_of(raw_action) -> jax.Array:
    """행동 벡터의 꼬리에서 게이트만 뗀다. 명령은 앞 `spec.DIM` 칸이다."""
    return jnp.asarray(raw_action, jnp.float32).reshape(ACTION_SIZE)[spec.DIM:]


def command_of(raw_action) -> jax.Array:
    """행동 벡터의 앞부분. `action.to_command` 에 넣을 원시 명령."""
    return jnp.asarray(raw_action, jnp.float32).reshape(ACTION_SIZE)[:spec.DIM]


def update(state: SkillState, raw_action, gravity_z, allow) -> SkillState:
    """다음 스킬 상태. **jit 안에서 돈다. 분기가 없다.**

    순서가 권한 순서다.

        1  걸쇠가 걸려 있으면 그대로 둔다        약속을 지킨다
        2  넘어져 있으면 복구를 건다             HLC 를 선점한다
        3  게이트가 눌렸고 발동 집합 안이면 건다

    `allow` -- 게이트별 발동 허용 (N_GATES,). 규칙으로 좁힌 발동 집합이다.
    예를 들어 점프는 "정면에 틈이 있고 정렬돼 있을 때"만 1 이다. 이것이 없으면
    정책이 아무 데서나 눌러 넘어지고, "절대 안 누른다"로 수렴한다.

    지금은 `REGISTRY` 가 전부 `enabled=False` 라 이 함수는 항상 걷기를 낸다.
    """
    fallen = jnp.where(gravity_z > FALLEN_GRAVITY_Z, state.fallen + 1, 0)

    latched = state.left > 0
    nxt = SkillState(active=state.active, left=state.left - 1, fallen=fallen)

    # 복구. 규칙 발동이라 게이트를 안 본다.
    rec = [(i, s) for i, s in enumerate(REGISTRY) if not s.gated and s.enabled]
    for i, s in rec:
        fire = (fallen >= FALLEN_HOLD) & ~latched
        nxt = SkillState(
            active=jnp.where(fire, i, nxt.active),
            left=jnp.where(fire, s.max_steps, nxt.left),
            fallen=fallen)
        latched = latched | fire

    # 게이트 슬롯을 쓰는 스킬. **방아쇠는 `trigger` 가 정한다.**
    gates = gates_of(raw_action) * GATE_ENABLED
    allow = jnp.asarray(allow, jnp.float32).reshape(N_GATES)
    for g, s in enumerate(GATED):
        if not s.enabled:
            continue
        i = REGISTRY.index(s)
        # `"rule"` 이면 HLC 출력을 안 본다. 그 칸은 죽은 출력으로 남는데,
        # 슬롯을 빼면 서명이 깨지므로 그대로 둔다 (`Skill.gated` 주석).
        pressed = (gates[g] > GATE_THRESHOLD) if s.trigger == "gate" else True
        fire = pressed & (allow[g] > 0) & ~latched
        nxt = SkillState(
            active=jnp.where(fire, i, nxt.active),
            left=jnp.where(fire, s.max_steps, nxt.left),
            fallen=fallen)
        latched = latched | fire

    done = nxt.left <= 0
    return SkillState(active=jnp.where(done, -1, nxt.active),
                      left=jnp.maximum(nxt.left, 0),
                      fallen=fallen)


def any_enabled() -> bool:
    """켜진 스킬이 하나라도 있나. 없으면 부르는 쪽이 통째로 건너뛸 수 있다."""
    return any(s.enabled for s in REGISTRY)
