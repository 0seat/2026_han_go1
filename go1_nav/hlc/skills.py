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
    gated: bool         #: HLC 게이트가 필요한가. False 면 규칙이 발동한다
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
    Skill(name="getup", gated=False, enabled=False, max_steps=40,
          phase="go1_getup_11d_floorsafe_terrain_friction_curriculum_optuna_100m_v2",
          run="20260814_071905"),
    Skill(name="jump", gated=True, enabled=False, max_steps=6,
          phase="jump_05m_11d_floor_safe_optuna_100m_fixed_from_scratch",
          run="20260814_022050"),
)

#: 게이트가 필요한 스킬만. **행동 벡터의 꼬리 길이가 이것으로 정해진다.**
GATED: tuple[Skill, ...] = tuple(s for s in REGISTRY if s.gated)

#: 게이트 칸 수. 스킬을 꺼도 줄지 않는다 -- 줄면 신경망 출력층이 바뀐다.
N_GATES = len(GATED)

#: HLC 행동 벡터 크기. 11축 명령 + 게이트.
ACTION_SIZE = spec.DIM + N_GATES

#: 켜진 게이트만 1. 꺼진 칸의 출력은 버린다 -- 탐색 잡음만 만들고 보상에 영향이
#: 없으면 엔트로피가 높은 채로 남아 다른 축의 탐색을 방해한다.
GATE_ENABLED = jnp.asarray([1.0 if s.enabled else 0.0 for s in GATED],
                           dtype=jnp.float32)

#: 발동 문턱. 원시 출력이 이보다 크면 누른 것으로 본다.
GATE_THRESHOLD = 0.0


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

    # 게이트 스킬.
    gates = gates_of(raw_action) * GATE_ENABLED
    allow = jnp.asarray(allow, jnp.float32).reshape(N_GATES)
    for g, s in enumerate(GATED):
        if not s.enabled:
            continue
        i = REGISTRY.index(s)
        fire = (gates[g] > GATE_THRESHOLD) & (allow[g] > 0) & ~latched
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
