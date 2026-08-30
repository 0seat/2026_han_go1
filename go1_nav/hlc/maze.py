"""미로 — 랜드를 이어붙여 높이 격자와 막힘 격자를 만든다.

**여기는 mujoco를 import 하지 않는다.** 이 파일이 말하는 것은 "3행 4열은 경사"
까지고, 그것을 hfield로 굽는 것은 `env.py`, 격자로 읽는 것은 나중의 nav다.
여기가 mujoco를 물면 로봇 위에서 도는 nav가 mujoco를 끌고 다니게 된다.

    generate(seed)   오프라인. numpy 자유
    save / load      .npz. git에 넣지 않는다 (씨앗이 재현성을 보장한다)

지금은 `env.py` 혼자 쓴다. nav가 실제로 import 하는 날 `common/`으로 옮기고
격자 해상도와 좌표 규약을 `docs/contracts.md`에 C5로 올린다.

구조
----
정사각형 랜드 하나가 장애물 하나다. **랜드 전체가 그 성질을 갖는다** -- 경사
랜드는 랜드 전체가 한 방향 비탈이고, rough 랜드는 랜드 전체가 울퉁불퉁하다.
이것을 가로 세로 아무 개수로나 이어붙여 미로를 만든다.

높이는 **단으로 센다.** 한 단이 HIGH = 0.713 m이고, 경사 랜드 하나를 20도로
올라간 높이다. 단은 0부터 LEVEL_MAX까지.

높이를 바꾸는 랜드는 **경사뿐이다.** 나머지는 들어간 높이와 나온 높이가 같아서,
이어붙일 때 변 높이가 어긋날 일이 없다. 턱도 gap도 자기 안에서 올라갔다
내려오거나 파였다 돌아온다.

그래서 그래프가 저절로 나온다.

    경사로 올라간다        양방향 간선
    경사 없이 떨어진다     단방향 간선.  내려갈 수는 있고 올라올 수는 없다
    gap · 벽              간선 없음

왜 박스가 아니라 heightfield인가
--------------------------------
랜드마다 박스 K개로 만들면 rough 하나 때문에 모든 랜드가 박스 수십 개를 지고
다닌다. 높이로 표현되는 것은 hfield 한 장이 전부 흡수한다. mjx가 이것을
허용한다.

    hfield_nrow / ncol / size   np.ndarray   배치 전체가 같아야 한다
    hfield_data                 jax.Array    환경마다 달라도 된다

geom과 같은 구조다 -- 해상도는 고정, 값은 자유. 높이 함수로 표현할 수 없는 것
하나만 나중에 geom으로 간다.

    터널 천장   높이 함수는 위아래 두 겹을 못 만든다

벽도 돌맹이도 hfield다. 막느냐 아니냐는 높이가 아니라 **막힘 격자**가 정한다.
벽 랜드는 막힘으로 표시되어 nav가 돌아가고, 돌맹이 랜드는 통과로 남는다.

턱이 직각이 아닌 이유
---------------------
높이 격자는 함수라 한 점에 높이가 하나뿐이고, MuJoCo는 이웃한 두 점을 삼각형으로
잇는다. 그래서 턱의 면은 `atan(단높이 / CELL)`이 되고 **직각이 될 수 없다.**
CELL 0.04 m에서 0.2 m 턱은 78.7도다. 진짜 직각이 필요하면 박스 geom으로 가야
하는데, 그 전에 78.7도를 못 넘는지부터 확인할 것.

좌표 규약
---------
`hfield_data`의 행이 y, 열이 x이고 첫 값이 (-x, -y) 구석이다. 확인 방법 --
어느 한 랜드만 올려 렌더하면 `tile_center()`가 내는 자리에 나타난다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# ---------- 랜드 ----------

TILE = 2.0                  #: 랜드 한 변 (m)
SLOPE_DEG = 20.0            #: 경사 랜드의 기울기. Go1 등판 한계 25도에서 5도 여유

CELLS_PER_TILE = 50         #: 랜드 하나에 들어가는 격자 수. 나눠떨어지게 잡는다
CELL = TILE / CELLS_PER_TILE                      #: 격자 간격 = 0.04 m

# 위 높이를 랜드 크기가 아니라 **경사가 실제로 쓰는 칸 수**에서 뽑는다.
#
# 경사는 낮은 이웃과 높은 이웃을 잇는다. 격자점 n개로 이으면 구간은 n-1개다.
# HIGH를 `TILE * tan(20도)`로 잡으면 구간 하나가 `TILE/(n-1)`보다 커져서, 칸당
# 기울기가 20.4도가 되고 이음매에 작은 턱이 남는다. 구간 수로 나눠 잡으면
# **모든 칸이 정확히 20.0도이고 이음매 단차가 0이다.**
HIGH = (CELLS_PER_TILE - 1) * CELL * math.tan(math.radians(SLOPE_DEG))  #: 0.713 m

# 높이 격자의 값은 [0, 1]이고 실제 높이는 `값 * hfield_size[2]`다. 즉 **0 아래를
# 표현할 수 없다.** gap은 0 아래로 파여야 하므로 z=0을 격자 한가운데로 옮긴다.
#
#     hfield_size[2] = ELEVATION + DEPTH      격자가 덮는 세로 범위
#     floor geom 의 z = -DEPTH                그래서 평지가 다시 z=0에 온다
#
# 이 두 가지를 `env.py`가 같이 해야 한다. 하나만 하면 지형 전체가 위아래로
# 통째로 밀린다.
LEVEL_MAX = 2               #: 높이 단 상한. 0, 1, 2 -> 세 단
# 언덕이 정답지 바깥으로 몇 랜드까지 퍼지는가. 0이면 정답지 폭 그대로다.
# 넓힐수록 언덕이 세로 전체를 덮어 **단방향 간선이 사라진다** (6 x 16, 200개) --
#
#     0    단방향 간선 평균 5.6개,  있는 미로 169/200
#     1    단방향 간선 평균 2.9개,  있는 미로 108/200
HILL_PAD = 0
ELEVATION = LEVEL_MAX * HIGH + 0.8 + 0.05   #: 0 위 최대 높이. 맨 위 단의 벽까지 담는다
DEPTH = 1.10                #: 0 아래로 표현 가능한 최대 깊이. 도랑 · 다리 낭떠러지를 담는다
BASE = 1.0                  #: hfield_size[3]. 격자 아래로 채우는 살덩이 두께
SPAN = ELEVATION + DEPTH    #: hfield_size[2]

#: 턱 높이 (m). **실측으로 정했다** -- 근거 없이 고른 0.2 를 대체한다.
#:
#: 2026-08-19, phase18 체크포인트, 고정 명령 vx=0.6, `hlc/measure.step_limit`.
#:
#:     0.04  통과      면각도 45.0도
#:     0.05  통과      면각도 51.3도
#:     0.06  통과      면각도 56.3도      <- 여기
#:     0.07  실패      면각도 60.3도
#:     0.08  실패      면각도 63.4도
#:
#: 실패 유형은 **발이 면에 걸리는 것**이다. 미끄러지지도 전복하지도 않는다.
#: 면 앞에서 서서 계속 밀다가 13초쯤에 자세가 무너진다. 실측 발 들림이 2.7 cm
#: (`llc/spec.py`의 `FIXED_GAIT`)이고 그 두 배 언저리가 한계라는 그림이다.
#:
#: **이 값은 커리큘럼 구간의 아래끝이지 정답이 아니다.** LLC의 `footswing` 축이
#: 열리면 (명령 범위 0.06~0.15) 다시 재서 위끝을 올린다. 그때 바뀌는 것은 이
#: 숫자뿐이고 미로의 그래프 성질은 안 바뀐다 -- 턱은 `blocked`를 세우지 않는
#: 난이도 요소라, 높이를 낮춰도 단방향 간선은 경사와 낙차가 그대로 만든다.
#:
#: 면 각도는 자유 변수가 아니다. 한 셀(0.04 m) 안에서 올라가므로 높이가 정하면
#: 따라온다. 0.06 이면 56.3도다.
STEP_HEIGHT = 0.06
STEP_SPAN = 0.6             #: 턱 상판 길이 (m). 올라섰다 내려온다
GAP_WIDTH = 0.5             #: gap 폭 (m)
GAP_DEPTH = 0.5             #: gap 깊이 (m). 구멍이 아니라 도랑이다

WALL_HEIGHT = 0.8           #: 벽 높이 (m). 턱과 모양이 같고 높이만 다르다
WALL_SPAN = 0.3             #: 벽 두께 (m)
ROCK_COUNT = 5              #: 돌맹이 랜드에 놓는 돌 개수
ROCK_HEIGHT = 0.14          #: 돌 높이 (m). 넘을 수 있게 턱보다 낮게 둔다
ROCK_FACE_DEG = 45.0        #: 돌의 최대 경사. 여기서 반지름이 정해진다
#: 돌 반지름 (m). 높이와 최대 경사에서 나온다 -- 셋 중 둘을 정하면 나머지가 따라온다.
#: 융기 코사인 z = H/2 * (1 + cos(pi d / R)) 의 최대 기울기가 H*pi/(2R)라서,
#: 그것이 tan(ROCK_FACE_DEG)가 되는 R을 쓴다. 지금 값으로 0.2199 m다.
ROCK_RADIUS = ROCK_HEIGHT * math.pi / (2.0 * math.tan(math.radians(ROCK_FACE_DEG)))

ROUGH_HEIGHT = 0.06         #: 울퉁불퉁 진폭 (m). 위아래로 이만큼씩
ROUGH_FEATURE = 0.25        #: 요철 하나의 크기 (m). 발보다 커야 발이 빠지지 않는다

#: 외나무다리 기둥 폭 (m). **실측으로 고른 값이다.**
#:
#: 예전 값 0.40 은 stance width 0.202 m 의 두 배라는 것 말고 근거가 없었다.
#: 요각 지터를 끄고 대조군 16 판씩 굴려 도달률을 재니 이랬다.
#:
#:      폭     고정 (직진만)    조준 (조향한다)
#:      0.40   0.062            0.062
#:      0.50   0.188            --
#:      0.60   0.312            0.375
#:      0.80   0.438            --
#:      1.00   0.438            0.688
#:
#: 0.40 은 조향을 해도 0.062 다. **폭이 아니라 발 놓는 정밀도의 문제다.** HLC 는
#: 속도를 명령할 뿐 발 자리를 못 고르므로 학습으로 메울 여지가 얇다. 터널과
#: 다르다 -- 터널의 대조군 0.000 은 조준 제어기에 height 축이 없어서였고 그 축은
#: PPO 가 쓸 수 있었다.
#:
#: 고정이 1.00 에서도 0.438 에 머무는 것은 폭 때문이 아니라 직진 명령의 요각
#: 표류다. 조준이 같은 폭에서 0.688 인 것이 그 증거다.
#:
#: 그래서 첫 칸은 0.60 이다. 조준 0.375 라 벽이 아니고 위로 여지가 있다. 0.40 은
#: 나중 칸으로 미룬다. **이 값은 서명 항목이 아니라 바꿔도 restore 는 된다.**
BRIDGE_WIDTH = 0.60
BRIDGE_BAR = 0.30           #: I 의 가로대 길이 (m). 진행 방향으로 이만큼
BRIDGE_DROP = 1.00          #: 양옆이 파인 깊이 (m). 떨어지면 못 돌아온다
PIT_DEPTH = 1.00            #: 절벽 랜드가 꺼진 깊이 (m)

TUNNEL_WIDTH = 0.80         #: 터널 통로 폭 (m)
#: 터널 천장 아랫면 높이 (m).
#:
#: 0.32 에서 올렸다. 근거 -- 몸통 높이 명령의 실측 도달 범위가 0.218~0.264 이고
#: 몸통 두께 절반이 약 0.057 이라, 천장까지 여유가 이랬다.
#:
#:     0.32    기본 자세(명령 0.30)  +8 mm     낮춰도(명령 0.22)  +45 mm
#:     0.36    기본 자세             +48 mm    낮춰도             +85 mm
#:
#: 위 두 줄은 기하로 **계산한** 값이라 틀렸다. 모델의 몸통 geom 오프셋이 0.2359 m
#: 로 나오는데 그대로면 몸통 윗면이 0.53 이고, 그러면 0.36 짜리 터널을 못 지나야
#: 한다. 실제로는 지난다. 충돌에 안 쓰이는 geom 이 섞여 있다는 뜻이다.
#:
#: 그래서 **통과 경계를 직접 스윕했다** (2026-08-21, phase18, vx 0.6, 직진).
#: 턱 높이를 정할 때와 같은 방법이다.
#:
#:     천장    명령 0.30 (기본 자세)   명령 0.25 (정책이 낼 수 있는 최저)
#:     0.36    통과  x=3.01            통과  x=3.01
#:     0.34    실패  x=0.87            통과  x=3.01
#:     0.32    실패  x=0.87            통과  x=3.01
#:     0.30    실패  x=0.88            실패  x=2.02
#:     0.28    실패  x=0.87            실패  x=0.87
#:
#: 터널은 x=1.0 에서 시작한다. 기본 자세는 **입구 앞 0.87 에서 막힌다.**
#:
#: 학습 가능한 창이 **0.32 ~ 0.34** 다. 아래 끝을 고른다 -- "숙여야만 통과"가
#: 확실하고, 보간한 값이 아니라 실측한 격자점이다. 0.30 에서는 낮춰도 x=2.02 에서
#: 걸리므로 한 칸 여유가 있다.
#:
#: 왜 0.36 을 버리는가 -- 그 값에서는 **기본 자세로 통과된다.** 실제로 2026-08-21
#: 에 도달 0.953 까지 학습된 정책의 성공판 몸통 최저 z 가 0.250 으로 평지와 같았다.
#: 정렬해서 들어가는 법을 배웠을 뿐 몸을 낮춘 적이 없다. 이 랜드의 목적은
#: **`height` 축을 쓰게 만드는 것**이므로 그 값으로는 목적을 못 이룬다.
#:
#: `action.SCALE["height"]` 은 안 건드린다. 지금 정책이 낼 수 있는 최저 명령
#: 0.25(= CENTRE 0.30 - SCALE 0.05)로 통과한다. 실측 몸통 z 는 보행 중
#: 명령 0.25 에서 0.243, 명령 0.32 에서 0.291 -- **권한 47 mm 에 보행 진동
#: 8~13 mm** 라 판정이 진동에 묻히지 않는다.
TUNNEL_CLEAR = 0.32         #: 천장 아래 여유 높이 (m). **실측으로 정했다**
TUNNEL_THICK = 0.10         #: 천장 두께 (m)
TUNNEL_WALL = TUNNEL_CLEAR + TUNNEL_THICK   #: 옆벽 높이. 천장 윗면까지 막는다

assert ELEVATION >= LEVEL_MAX * HIGH + WALL_HEIGHT, (
    f"ELEVATION({ELEVATION:.3f})이 최대 지형 높이"
    f"({LEVEL_MAX * HIGH + WALL_HEIGHT:.3f})보다 작으면 잘린다"
)
assert DEPTH >= max(GAP_DEPTH, BRIDGE_DROP, PIT_DEPTH), (
    f"DEPTH({DEPTH})가 도랑·다리·절벽 깊이"
    f"({max(GAP_DEPTH, BRIDGE_DROP, PIT_DEPTH)})보다 작으면 잘린다"
)

# ---------- 랜드 종류 ----------

FLAT = 0        #: 평지
RAMP = 1        #: 경사. **높이를 바꾸는 유일한 랜드**
STEP = 2        #: 턱. 올라섰다 내려온다
GAP = 3         #: 도랑. 점프가 생기기 전까지 막힘으로 둔다
ROCK = 4        #: 돌맹이. 넘거나 비켜간다. **지나갈 수 있다**
WALL = 5        #: 벽. 턱과 같은 띠인데 넘을 수 없는 높이
ROUGH = 6       #: 울퉁불퉁. 랜드 전체가 잔요철. **지나갈 수 있다**
BRIDGE = 7      #: 외나무다리. 대문자 I 모양. **한 축으로만 지나간다**
TUNNEL = 8      #: 터널. 옆벽은 hfield, **천장만 geom**
PIT = 9         #: 절벽. 랜드 전체가 꺼져 있다. 들어갈 수는 있고 못 나온다

IMPLEMENTED = (FLAT, RAMP, STEP, GAP, ROCK, WALL, ROUGH, BRIDGE, TUNNEL, PIT)

#: 종류 번호 -> 사람이 읽는 이름. 로그와 영상 요약이 쓴다. 숫자만 찍으면 무엇을
#: 본 것인지 매번 이 파일을 열어 봐야 한다.
NAMES = {FLAT: "평지", RAMP: "경사", STEP: "턱", GAP: "도랑", ROCK: "돌",
         WALL: "벽", ROUGH: "거침", BRIDGE: "다리", TUNNEL: "터널",
         PIT: "절벽"}
assert set(NAMES) == set(IMPLEMENTED), "이름표에서 빠진 종류가 있습니다"

#: 그림에 쓰는 영문 이름표. **그래프 라벨은 영어로만 쓴다** -- 한글 글꼴이 없는
#: 환경에서 두부(네모)로 깨지고, 그림은 다른 컴퓨터에서 열린다. 표와 터미널
#: 출력은 `NAMES` 를 그대로 쓴다.
NAMES_EN = {FLAT: "flat", RAMP: "ramp", STEP: "step", GAP: "gap", ROCK: "rock",
            WALL: "wall", ROUGH: "rough", BRIDGE: "bridge", TUNNEL: "tunnel",
            PIT: "pit"}
assert set(NAMES_EN) == set(IMPLEMENTED), "영문 이름표에서 빠진 종류가 있습니다"

#: 막힘 격자에서 지나갈 수 없는 종류.
IMPASSABLE = (GAP, WALL, PIT)

# 돌맹이와 벽을 나눈 기준은 **막힘 격자에 나타나는가**다.
#
#     벽      랜드를 막는다.    nav 가 돌아가야 한다  ->  blocked
#     돌맹이  지나갈 수 있다.   nav 는 신경 안 쓴다   ->  통과
#
# 그래서 둘 다 hfield로 만든다. 앞서 "바위는 geom이어야 한다"고 했던 것은
# **랜드보다 작은 장애물이 막힘을 만드는 경우**를 상정한 것인데, 랜드 단위
# 그래프에서는 그런 것을 표현할 수 없다. 막을 것이면 랜드 하나를 막는다.
#
# 턱과 벽은 **같은 띠이고 높이만 다르다.** 그래서 이 둘이 "넘을 수 있는 턱"과
# "못 넘는 턱"이다.
#
# 주의 -- 돌맹이 랜드는 nav 가 못 본다. HLC 가 알아서 넘거나 비켜야 하는데,
# **지금 HLC 관측에는 지형 정보가 없다** (경로 9개뿐). 돌맹이를 학습에 넣으려면
# 높이 지도든 라이다든 관측을 하나 더 줘야 한다. 이건 아직 안 정했다.

# ---------- 띠의 방향 ----------
#
# 다리와 터널만 방향이 있다. 이 값은 **통로가 뻗는 방향**이고, 그 방향으로만
# 지나갈 수 있다. 턱과 도랑은 + 모양이라 방향이 없다.

RUN_Y = 0       #: 긴 쪽이 y축을 따라 뻗는다 (기본)
RUN_X = 1       #: 긴 쪽이 x축을 따라 뻗는다

# 주의 -- 이 값은 **모양이 뻗는 방향**이지 지나가는 방향이 아니다. 둘의 관계는
# 랜드 종류마다 다르다.
#
#     외나무다리 · 터널   통로를 따라 걷는다.  RUN_Y 면 y 로 지나간다
#     턱 · 도랑           + 모양.  방향이 없다
#     벽                  랜드를 통째로 채운다.  방향이 없다


@dataclass
class Maze:
    """랜드 표와 거기서 나온 격자 둘.

    `height`는 `env.py`가, `blocked`와 `level`은 nav가 읽는다. **한 곳에서
    나왔으므로 벽 위치가 어긋날 수 없다.**

    주의 -- `blocked`만으로는 단방향 간선을 표현할 수 없다. 경사 없이 위에서
    아래로 떨어지는 것은 갈 수 있고 그 반대는 못 간다. nav는 `level`을 같이
    읽어 방향을 판단해야 한다.
    """

    seed: int
    gate: float           #: 만들 때 쓴 관문 비율. 씨앗만으로 다시 만들 때 필요하다
    kind: np.ndarray      #: (TY, TX) int8. 랜드 종류
    level: np.ndarray     #: (TY, TX) int8. 높이 단 0..LEVEL_MAX. 실제 높이는 level*HIGH
    axis: np.ndarray      #: (TY, TX) int8. 다리 · 터널 통로의 방향 (RUN_Y / RUN_X)
    height: np.ndarray    #: (NROW, NCOL) float32, [0, 1]. mujoco hfield_data
    blocked: np.ndarray   #: (TY, TX) bool. 지나갈 수 없는 랜드
    path: np.ndarray      #: (TY, TX) bool. 생성기가 뚫어둔 정답지
    ceiling: np.ndarray   #: (N, 6) float32. 천장 박스. hfield로 안 되는 유일한 것
    start: np.ndarray     #: (2,) float32. 월드 xy
    start_yaw: float      #: 라디안. 정답지가 출발 랜드에서 나가는 방향
    goal: np.ndarray      #: (2,) float32. 월드 xy
    #: (N, 2) int32. 정답지를 **출발에서 도착 순서대로** 적은 랜드 좌표 (행, 열).
    #:
    #: `path` 만으로는 순서를 복원할 수 없다. 길이 자기 옆을 지나가면 어느 칸이
    #: 꺾임인지 판정이 갈린다 -- 같은 질문("장애물이 꺾임에서 몇 칸인가")에
    #: 복원 방식을 바꿔가며 30% · 79% 두 답을 얻은 적이 있다. 순서를 생성기가
    #: 직접 남기면 그 애매함이 사라진다.
    #:
    #: 미로 위 임의 지점에서 출발시키는 커리큘럼도 이 배열이 있어야 만든다.
    route: np.ndarray = None      # type: ignore[assignment]

    @property
    def shape(self) -> tuple[int, int]:
        """랜드 개수 (세로, 가로)."""
        return self.kind.shape

    @property
    def route_xy(self) -> np.ndarray:
        """(N, 2) float32. `route` 를 랜드 중심의 월드 xy 로."""
        if self.route is None or len(self.route) == 0:
            return np.zeros((0, 2), dtype=np.float32)
        return np.stack([tile_center(int(r), int(c), self.kind.shape)
                         for r, c in self.route])

    @property
    def turns(self) -> np.ndarray:
        """(M,) int32. `route` 에서 진행 방향이 바뀌는 칸의 **인덱스**.

        꺾임의 정의를 여기 한 곳에 둔다. 밖에서 격자를 보고 다시 판정하면
        답이 갈린다.
        """
        r = self.route
        if r is None or len(r) < 3:
            return np.zeros((0,), dtype=np.int32)
        d = np.diff(r, axis=0)
        return (np.nonzero((d[1:] != d[:-1]).any(axis=1))[0] + 1).astype(np.int32)

    @property
    def extent(self) -> tuple[float, float]:
        """맵 크기 (x, y) 미터."""
        ty, tx = self.kind.shape
        return tx * TILE, ty * TILE


def tile_center(row: int, col: int, shape: tuple[int, int]) -> np.ndarray:
    """랜드 중심의 월드 xy. 맵 중심이 원점이다."""
    ty, tx = shape
    return np.asarray([(col + 0.5) * TILE - tx * TILE / 2,
                       (row + 0.5) * TILE - ty * TILE / 2], dtype=np.float32)


#: 정답지에 놓는 장애물 종류. `generate(kinds=...)` 로 좁힐 수 있다.
#:
#: **턱을 빼고 돌릴 수 있어야 한다.** `STEP_HEIGHT = 0.06` 은 실측한 LLC 한계
#: 바로 그 값이다 (0.06 통과 · 0.07 실패). 여유가 0 이라 미로의 모든 턱이
#: 한계선에 걸친 장애물이고, 실패 유형이 "발이 면에 걸려 앞에서 밀다가 무너짐"
#: 이라 시간 초과로 찍힌다 -- 다른 원인과 구분이 안 된다.
#:
#: 넘을 수 없다고 이미 실측된 것을 빼면 남은 실패가 전부 진짜 원인이 된다.
#: `footswing` 이 열려 `STEP_HEIGHT` 를 올리면 그때 도로 넣는다.
#:
#: `IMPLEMENTED` 는 안 건드린다 -- 그쪽을 줄이면 `kind_onehot` 길이가 바뀌어
#: 관측 서명이 깨진다. 여기는 **생성에만** 쓰는 목록이다.
PLACED = (ROUGH, ROCK, STEP, BRIDGE, TUNNEL)


def generate(seed: int = 0, shape: tuple[int, int] = (4, 10),
             gate: float = 0.5, kinds=None, density=None) -> Maze:
    """씨앗에서 미로를 만든다. 같은 씨앗이면 같은 미로다.

    `.npz`를 git에 넣지 않는 근거가 이 결정성이다. 재현성은 파일이 아니라
    씨앗이 보장한다.

    만드는 순서
    -----------
    1. **정답지를 먼저 뚫는다.** 열마다 한 칸씩 오른쪽으로 가고 그 사이에
       위아래로 조금씩 움직인다. 단조롭게 오른쪽으로만 가므로 자기와 겹치지
       않고 반드시 도착한다.
    2. **높이를 열의 속성으로 정한다.** 평평한 구간 몇 개를 두고, 그 사이의
       열 하나를 경사 열로 만든다. 경사 열에서 높이가 한 단 바뀐다.
    3. 정답지 위에 **장애물을 고루** 놓는다. 종류마다 최소 하나씩 들어간다.
    4. 정답지 밖은 무작위. 벽과 도랑은 여기에만 놓는다.
    5. 장애물이 놓인 열을 **관문**으로 막는다. 그 열의 나머지를 전부 벽으로
       만들어 장애물을 피해 갈 수 없게 한다.
    6. `reachable()`로 목표까지 닿는지 확인한다.

    관문이 필요한 이유 -- 장애물을 놓기만 하면 옆으로 돌아가는 평지가 늘 있다.
    그러면 정책은 **평지만 밟는 로얄로드**를 배우고 장애물은 영영 안 배운다.
    `gate`로 관문 비율을 조절한다. 0이면 관문 없음, 1이면 가능한 전부. 1로 두면
    길이 통로 하나만 남아 우회로가 사라지므로 기본은 0.5다. 실측(6 x 16, 100개) --

        gate 0.0    반드시 지나야 하는 장애물 미로당 4.7개
        gate 1.0    반드시 지나야 하는 장애물 미로당 7.6개

    왜 정답지를 먼저 뚫는가 -- 아무렇게나 놓고 통하는지 확인하면 대부분 안 통해서
    계속 다시 만들게 된다. 길을 먼저 두면 **통하는 것이 보장된다.**

    왜 높이가 랜드가 아니라 **열**의 속성인가
    ----------------------------------------
    길이 한 열 안에서 위아래로 움직일 때가 있다. 같은 열이 같은 높이면 그 세로
    이동에 절벽이 안 생긴다. 랜드마다 높이를 흩뿌리면 길 옆에 계속 절벽이 생겨
    어디로 갈 수 있는지가 엉킨다.

    그리고 경사의 방향을 따로 정할 필요가 없다. `_ramp`가 이웃 높이를 읽는데,

        경사 열의 좌우 이웃   높이가 다르다        ->  x축 경사로 잡힌다
        경사 열의 상하 이웃   같은 열이라 높이가 같다  ->  헷갈릴 일이 없다

    경사 열은 **열 전체가 경사다.** 일부만 경사로 두면 같은 열 안에 경사면과
    평면이 섞여 그 경계가 절벽이 된다.

    한계 -- 길이 한 갈래다. 갈림길도 막다른 길도 없다.
    """
    ty, tx = shape
    if tx < 4:
        raise ValueError(f"가로 랜드가 {tx}개면 경사를 넣을 자리가 없습니다. 4 이상.")
    rng = np.random.default_rng(seed)

    # ---- 1. 정답지 뚫기 ----
    # 높이를 먼저 정한다. 길이 언덕 안에서 위아래로 헤매면 언덕이 세로 전체로
    # 퍼져서, 같은 열이 전부 같은 높이가 되고 **단방향 간선이 사라진다.**
    col_level, is_ramp = _column_levels(rng, tx)
    hill = np.zeros(tx, dtype=bool)
    for a, b in _runs(col_level, is_ramp):
        hill[a:b] = True

    # 열마다 한 칸씩 흔들면 세로로 긴 맵에서 길이 한 줄에만 붙어 있게 된다.
    # 가끔 크게 건너뛰어 맵 전체를 쓴다. 다만 **언덕 안에서는 행을 고정한다.**
    # 언덕 밖에서는 **매 열 행을 바꾼다.** 조금씩만 흔들면 길이 거의 직선이라
    # 세로로 지나는 구간이 안 생기고, 다리 · 터널을 세로로 놓을 자리도 없다.
    # 지그재그가 많을수록 정답지가 맵을 넓게 쓰고 장애물 자리도 늘어난다.
    rows = [int(rng.integers(0, ty))]
    for c in range(1, tx):
        if hill[c] and hill[c - 1]:
            # 언덕 안에서는 한 칸씩만. 크게 건너뛰면 언덕 직사각형이 세로 전체로
            # 퍼져서 같은 열이 전부 같은 높이가 되고 **단방향 간선이 사라진다.**
            rows.append(int(np.clip(rows[-1] + rng.integers(-1, 2), 0, ty - 1)))
        elif ty < 2:
            rows.append(rows[-1])
        else:
            rows.append(int(rng.choice([r for r in range(ty) if r != rows[-1]])))

    on_path = np.zeros(shape, dtype=bool)
    travel: dict[tuple[int, int], set[int]] = {}

    def mark(r, c, ax):
        on_path[r, c] = True
        travel.setdefault((r, c), set()).add(ax)

    mark(rows[0], 0, 1)
    # **지나는 순서를 여기서 적는다.** 나중에 `path` 격자를 보고 복원하면
    # 길이 자기 옆을 스칠 때 순서가 갈린다. 뚫는 쪽이 알고 있으니 적어 둔다.
    order = [(rows[0], 0)]
    for c in range(1, tx):
        lo, hi = sorted((rows[c - 1], rows[c]))
        if lo != hi:
            for r in range(lo, hi + 1):
                mark(r, c, 0)               # 세로 이동
        else:
            on_path[lo, c] = True
        mark(rows[c - 1], c - 1, 1)         # 오른쪽으로 나감
        mark(rows[c - 1], c, 1)             # 오른쪽에서 들어옴
        # 오른쪽으로 한 칸 들어온 뒤, 그 열 안에서 세로로 움직인다.
        order.append((rows[c - 1], c))
        step = 1 if rows[c] > rows[c - 1] else -1
        for r in range(rows[c - 1] + step, rows[c] + step, step):
            order.append((r, c))

    # ---- 2. 언덕 ----
    kind = np.full(shape, FLAT, dtype=np.int8)
    axis = np.full(shape, RUN_Y, dtype=np.int8)
    level = np.zeros(shape, dtype=np.int8)

    # 언덕은 **정답지 주변만** 올린다. 열 전체를 올리면 같은 열이 전부 같은
    # 높이라 옆으로 떨어질 데가 없고, 그러면 **단방향 간선이 하나도 안 생긴다.**
    # 직사각형으로 올려 위아래 변을 절벽으로 남긴다.
    for lo_c, hi_c in _runs(col_level, is_ramp):
        span = [rows[c] for c in range(lo_c, hi_c)] +                [rows[max(0, lo_c - 1)]]
        r0 = max(0, min(span) - HILL_PAD)
        r1 = min(ty - 1, max(span) + HILL_PAD)
        for c in range(lo_c, hi_c):
            level[r0:r1 + 1, c] = col_level[c]
            if is_ramp[c]:
                kind[r0:r1 + 1, c] = RAMP   # 언덕 폭 전체. 일부만 두면 경계가 절벽
                axis[r0:r1 + 1, c] = RUN_X

    # ---- 3. 정답지 위 장애물 ----
    spots = [(r, c) for (r, c), d in travel.items()
             if len(d) == 1 and kind[r, c] != RAMP
             and (r, c) not in ((rows[0], 0), (rows[-1], tx - 1))]
    rng.shuffle(spots)
    # 자리가 종류 수보다 적을 수 있으므로 **순서도 섞는다.** 고정 순서로 두면
    # 늘 뒤쪽 종류(다리 · 터널)만 빠진다.
    want = [int(k) for k in (PLACED if kinds is None else kinds)]
    assert want, "놓을 장애물 종류가 하나는 있어야 합니다"
    rng.shuffle(want)
    # 종류 수보다 자리가 많으면 나머지는 무작위다. 평지를 두 번 넣어 장애물
    # 밀도를 낮춘다 -- 정답지가 장애물로만 채워지면 조주 거리가 사라진다.
    fill = [FLAT, FLAT] + want
    for i, (r, c) in enumerate(spots):
        k = want[i] if i < len(want) else int(rng.choice(fill))
        kind[r, c] = k
        t = next(iter(travel[(r, c)]))      # 0 = y 로 지나간다, 1 = x 로
        if k in (BRIDGE, TUNNEL):           # 통로를 따라 걷는다
            axis[r, c] = RUN_Y if t == 0 else RUN_X

    # ---- 3b. 밀도 채우기 ----
    #
    # **자리가 모자란 것이 밀도의 진짜 한계다.** 장애물은 한 방향으로만 지나는
    # 칸에만 놓는데, 정답지는 열마다 행을 바꿔서 대부분이 꺾임 칸이다. 실측 --
    # 4 x 10 에서 정답지 20칸 중 장애물이 6칸이고 그중 3칸은 언덕이 칠한 경사다.
    # 그래서 미로당 터널이 0.6 개이고, 하필 경로 끝에 놓이면 `span` 창 12개 중
    # **한 차선**에만 들어간다. 터널 통과 능력이 0.875 에서 0.000 으로 지워진
    # 사건의 배경이 이것이다.
    #
    # 꺾임 칸에도 놓는다. 다만 **통로 축이 있는 종류는 못 놓는다** -- 다리 상판과
    # 터널 천장은 곧은 띠라 그 위에서 90도를 돌면 띠 밖으로 나간다. 방향이 없는
    # 돌 · 거침 · 턱만 쓴다. 경사는 이미 언덕 코드가 꺾임 칸에 칠하고 있고,
    # 실측으로 정답지 장애물의 31 % 가 그것이었다.
    if density is not None:
        free = [int(k) for k in want if k not in (BRIDGE, TUNNEL)]
        if free:
            corners = [(r, c) for (r, c), d in travel.items()
                       if len(d) > 1 and kind[r, c] == FLAT
                       and (r, c) not in ((rows[0], 0), (rows[-1], tx - 1))]
            rng.shuffle(corners)
            walked = list(travel)          # 정답지가 실제로 밟는 칸
            have = sum(1 for (r, c) in walked if kind[r, c] != FLAT)
            need = int(round(float(density) * len(walked))) - have
            for r, c in corners[:max(0, need)]:
                kind[r, c] = int(rng.choice(free))

    # ---- 4. 정답지 밖 ----
    # 터널은 여기 없다. 천장이 geom이라 **개수가 모델에 박히고 배치 전체가 그
    # 비용을 진다.** 정답 경로 위의 터널은 반드시 지나야 하므로 값어치가 있지만,
    # 바깥에 흩뿌린 터널은 지나갈 일이 없으면서 비용만 늘린다.
    for r in range(ty):
        for c in range(tx):
            if on_path[r, c] or kind[r, c] == RAMP:
                continue
            kind[r, c] = int(rng.choice([PIT, WALL, GAP, ROUGH, ROCK, FLAT]))
            # 방향도 섞는다. 안 그러면 `axis`의 기본값 때문에 벽과 도랑이
            # 전부 세로 막대가 되어 위에서 보면 바코드처럼 나온다.
            axis[r, c] = int(rng.choice([RUN_Y, RUN_X]))

    # ---- 5. 관문 ----
    # 그 열에 정답지 랜드가 하나뿐일 때만 막는다. 세로 복도가 있는 열을 막으면
    # 정답지 자신을 끊는다. 경사 열과 출발 · 도착 열도 건드리지 않는다.
    per_col = on_path.sum(axis=0)
    gates = [(r, c) for (r, c) in spots
             if per_col[c] == 1 and kind[r, c] != FLAT
             and 0 < c < tx - 1 and 1 in travel[(r, c)]]
    rng.shuffle(gates)
    for r, c in gates[:int(round(len(gates) * float(np.clip(gate, 0.0, 1.0))))]:
        for rr in range(ty):
            if rr != r:
                kind[rr, c] = WALL

    # 출발 방향. 정답지가 출발 랜드에서 어느 쪽으로 나가는지를 그대로 쓴다.
    # 이걸 안 주면 리셋할 때 로봇이 어디를 보고 서야 할지 아무도 모른다.
    # 첫 스텝부터 벽을 보고 있으면 초반 보상이 전부 잡음이 된다.
    first = tile_center(rows[0], 1, shape) - tile_center(rows[0], 0, shape)
    mz = build(seed, kind, level, axis, on_path,
               start_yaw=float(np.arctan2(first[1], first[0])), gate=float(gate),
               route=np.asarray(order, dtype=np.int32))
    mz.start[:] = tile_center(rows[0], 0, shape)
    mz.goal[:] = tile_center(rows[-1], tx - 1, shape)
    if not reachable(mz)[rows[-1], tx - 1]:
        raise RuntimeError(
            f"씨앗 {seed}: 뚫어둔 길이 통하지 않습니다. 생성 규칙과 통과 규칙이 "
            f"어긋난 것이므로 씨앗을 바꿔 넘기지 말 것."
        )
    return mz


def _runs(col_level: list[int], is_ramp: list[bool]) -> list[tuple[int, int]]:
    """높이가 0이 아니거나 경사인 열들의 연속 구간. 언덕 하나가 구간 하나다."""
    out, start = [], None
    for c, (lv, rp) in enumerate(zip(col_level, is_ramp)):
        active = lv > 0 or rp
        if active and start is None:
            start = c
        elif not active and start is not None:
            out.append((start, c))
            start = None
    if start is not None:
        out.append((start, len(col_level)))
    return out


def _column_levels(rng, tx: int) -> tuple[list[int], list[bool]]:
    """열마다 높이 단과 경사 여부. 경사 열에서 **한 단씩만** 바뀐다.

    한 열에서 두 단을 올리면 그 랜드가 40도가 되어 못 오른다. 경사 하나는
    한 단이고, 두 단을 오르려면 경사 열이 두 개 필요하다.

    첫 열과 마지막 열은 경사가 될 수 없다. 경사는 좌우 이웃에서 높이를 읽는데
    한쪽이 없기 때문이다.
    """
    col_level = [0]
    is_ramp = [False]
    lv = 0
    while len(col_level) < tx:
        for _ in range(int(rng.integers(3, 7))):        # 평평한 구간
            if len(col_level) >= tx:
                return col_level, is_ramp
            col_level.append(lv)
            is_ramp.append(False)
        if len(col_level) >= tx - 1:                    # 마지막 열엔 경사를 못 둔다
            continue
        # 갈 수 있는 방향에서만 고른다. `clip`으로 눌러버리면 바닥과 꼭대기에서
        # 절반이 제자리걸음이 되고, 그러면 언덕 없는 미로가 잔뜩 나온다.
        #
        # 내려가는 쪽에 무게를 준다. 균등하게 뽑으면 한 번 오른 뒤 0으로 잘
        # 안 돌아와서 **맵의 대부분이 언덕**이 된다. 언덕 안은 길이 곧게 가야
        # 하므로, 언덕이 넓으면 지그재그가 사라진다.
        opts = [d for d in (-1, 1) if 0 <= lv + d <= LEVEL_MAX]
        w = np.asarray([3.0 if d < 0 else 1.0 for d in opts])
        nxt = lv + int(rng.choice(opts, p=w / w.sum()))
        col_level.append(lv)                            # 경사 열. 값은 `_ramp`가 덮는다
        is_ramp.append(True)
        lv = nxt
    return col_level, is_ramp


def reachable(maze: "Maze") -> np.ndarray:
    """출발 랜드에서 걸어서 닿는 랜드. (TY, TX) bool.

    **통과 규칙이 여기 한 곳에만 있다.** nav의 D*도 이 규칙을 써야 하므로,
    규칙을 바꾸려면 여기를 고치고 계약에 올린다.

        막힌 랜드          못 간다
        높이가 같다        간다
        경사               양끝이 이웃 높이라 간다
        높이가 다르다      내려가는 쪽만 간다 (단방향)
        다리 · 터널        제 축으로만 간다
    """
    ty, tx = maze.kind.shape
    r0, c0 = _tile_of(maze.start, maze.kind.shape)
    seen = np.zeros(maze.kind.shape, dtype=bool)
    stack = [(r0, c0)]
    seen[r0, c0] = True
    while stack:
        r, c = stack.pop()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if not (0 <= nr < ty and 0 <= nc < tx) or seen[nr, nc]:
                continue
            if _can_move(maze, r, c, nr, nc, moving_y=(dr != 0)):
                seen[nr, nc] = True
                stack.append((nr, nc))
    return seen


def route(maze: "Maze") -> np.ndarray:
    """출발에서 목표까지 **실제 최단 경로** 하나. (TY, TX) bool.

    `maze.path`와 다르다. 그쪽은 **생성기가 뚫은 자국**이고 이쪽은 통과 규칙으로
    다시 찾은 길이다. 둘이 다를 수 있다 -- 생성기가 굽이굽이 판 뒤에 바깥 채움이
    우연히 지름길을 열어주면, 실제 최단 경로는 판 자국을 벗어난다.

    **어느 쪽도 "정답"으로 학습에 넣으면 안 된다.** 지나갈 수 있는 길은 여럿이고,
    정책이 그중 어느 것을 고르든 틀린 게 아니다. 이 함수는 눈으로 보고 미로가
    풀리는지 확인하는 용도다.
    """
    ty, tx = maze.kind.shape
    src = _tile_of(maze.start, maze.kind.shape)
    dst = _tile_of(maze.goal, maze.kind.shape)
    prev: dict[tuple[int, int], tuple[int, int] | None] = {src: None}
    queue = [src]
    while queue:
        nxt = []
        for r, c in queue:
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                step = (r + dr, c + dc)
                if not (0 <= step[0] < ty and 0 <= step[1] < tx) or step in prev:
                    continue
                if _can_move(maze, r, c, step[0], step[1], moving_y=(dr != 0)):
                    prev[step] = (r, c)
                    nxt.append(step)
        queue = nxt
    out = np.zeros(maze.kind.shape, dtype=bool)
    if dst not in prev:
        return out
    node = dst
    while node is not None:
        out[node] = True
        node = prev[node]
    return out


def _can_move(maze: "Maze", r, c, nr, nc, moving_y: bool) -> bool:
    """랜드 (r,c) 에서 (nr,nc) 로 갈 수 있는가. `moving_y`는 y로 움직이는가."""
    if maze.blocked[r, c] or maze.blocked[nr, nc]:
        return False
    for rr, cc in ((r, c), (nr, nc)):
        if int(maze.kind[rr, cc]) in (BRIDGE, TUNNEL):
            # 통로가 뻗은 방향으로만 지나간다. RUN_Y 면 y 로 지난다.
            if (int(maze.axis[rr, cc]) == RUN_Y) != moving_y:
                return False
    # 경사는 **제 축으로만** 이웃과 맞물린다. 옆에서 경사면 중턱으로 올라탈 수는
    # 없다. 같은 경사 열 안에서 옆으로 걷는 것(둘 다 경사, 같은 축)은 된다.
    ka, kb = int(maze.kind[r, c]), int(maze.kind[nr, nc])
    aa, ab = int(maze.axis[r, c]), int(maze.axis[nr, nc])
    if ka == RAMP and kb == RAMP:
        return aa == ab
    if ka == RAMP or kb == RAMP:
        ramp_ax = aa if ka == RAMP else ab
        return (ramp_ax == RUN_Y) == moving_y

    la, lb = int(maze.level[r, c]), int(maze.level[nr, nc])
    if la == lb:
        return True
    return la > lb        # 내려가는 것만 된다. 여기가 단방향 간선이다


def _tile_of(xy, shape) -> tuple[int, int]:
    """월드 xy -> 랜드 인덱스."""
    ty, tx = shape
    c = int((float(xy[0]) + tx * TILE / 2) // TILE)
    r = int((float(xy[1]) + ty * TILE / 2) // TILE)
    return int(np.clip(r, 0, ty - 1)), int(np.clip(c, 0, tx - 1))


def build(seed: int, kind: np.ndarray, level: np.ndarray,
          axis: np.ndarray | None = None, path: np.ndarray | None = None,
          start_yaw: float = 0.0, gate: float = 0.0,
          route: np.ndarray | None = None) -> Maze:
    """랜드 표 -> 미로. 종류가 늘어도 이 함수의 모양은 안 바뀐다."""
    if axis is None:
        axis = np.full(kind.shape, RUN_Y, dtype=np.int8)
    shape = kind.shape
    return Maze(
        route=(np.zeros((0, 2), dtype=np.int32) if route is None
               else np.asarray(route, dtype=np.int32)),
        seed=int(seed), gate=float(gate), kind=kind, level=level, axis=axis,
        height=heightfield(kind, level, axis, seed),
        blocked=blocked(kind),
        path=np.zeros(kind.shape, dtype=bool) if path is None else path,
        ceiling=ceilings(kind, level, axis),
        start=tile_center(0, 0, shape),
        start_yaw=float(start_yaw),
        goal=tile_center(shape[0] - 1, shape[1] - 1, shape),
    )


# ---------- 높이 격자 ----------

def heightfield(kind: np.ndarray, level: np.ndarray,
                axis: np.ndarray | None = None, seed: int = 0) -> np.ndarray:
    """랜드 표 -> (NROW, NCOL) 높이 격자. mujoco가 요구하는 [0, 1] 정규화.

    실제 높이(m) = 값 * SPAN - DEPTH. `env.py`가 바닥 geom을 -DEPTH로 내려두므로
    값 0이 아니라 **미터 0이 z=0**이다 (CPU 레이캐스트로 확인).
    """
    if axis is None:
        axis = np.full(kind.shape, RUN_Y, dtype=np.int8)
    ty, tx = kind.shape
    n = CELLS_PER_TILE
    metres = np.repeat(np.repeat(level.astype(np.float32) * HIGH, n, axis=0), n, axis=1)

    for r in range(ty):
        for c in range(tx):
            k = int(kind[r, c])
            if k == FLAT:
                continue
            patch = metres[r * n:(r + 1) * n, c * n:(c + 1) * n]
            if k == RAMP:
                _ramp(patch, level, int(axis[r, c]), r, c)
            elif k == STEP:
                _step(patch)
            elif k == GAP:
                _gap(patch)
            elif k == WALL:
                _wall(patch)
            elif k == PIT:
                _pit(patch)
            elif k == ROCK:
                _rock(patch, seed, r, c)
            elif k == ROUGH:
                _rough(patch, seed, r, c)
            elif k == BRIDGE:
                _bridge(patch, int(axis[r, c]))
            elif k == TUNNEL:
                _tunnel(patch, int(axis[r, c]))
            else:
                raise NotImplementedError(f"랜드 종류 {k}는 아직 구현되지 않았습니다.")

    return normalize(metres)


def normalize(metres: np.ndarray) -> np.ndarray:
    """미터 -> mujoco가 요구하는 [0, 1]. 눈금은 `DEPTH`와 `SPAN`이 소유한다.

    시험용 지형(`lands.py`)도 이 함수를 쓴다. 눈금을 두 번 적으면 `env.py`가
    바닥 geom을 `-DEPTH`로 내려둔 것과 어긋나서, 지형이 통째로 위아래로
    밀린 채 조용히 돈다.
    """
    if metres.max() > ELEVATION or metres.min() < -DEPTH:
        raise ValueError(
            f"높이 {metres.min():.3f}~{metres.max():.3f} m가 "
            f"[-{DEPTH}, {ELEVATION}] 밖입니다. 잘려서 조용히 평평해집니다."
        )
    return ((metres + DEPTH) / SPAN).astype(np.float32)


def to_metres(height: np.ndarray) -> np.ndarray:
    """`normalize`의 역. 잰 값을 사람이 읽을 때 쓴다."""
    return np.asarray(height, dtype=np.float32) * SPAN - DEPTH


def _ramp(patch: np.ndarray, level: np.ndarray, ax: int, r: int, c: int) -> None:
    """경사 랜드. **랜드 전체가 한 방향 비탈이다.**

    어느 축인지는 `axis`가 정하고, **끝 높이는 이웃에서 읽는다.** 둘을 나눈
    이유가 있다.

        축을 이웃에서 유도하면   위아래 이웃도 높이가 다를 때 엉뚱한 축을 고른다
        끝 높이를 따로 적으면    이웃과 어긋나 경사 끝에 턱이 생긴다

    그래서 축만 적고 높이는 읽는다. 이러면 어느 쪽 이음매도 단차가 0이다.
    """
    n = patch.shape[0]
    ty, tx = level.shape

    def lv(rr, cc):
        return int(level[rr, cc]) if 0 <= rr < ty and 0 <= cc < tx else None

    a, b = (lv(r - 1, c), lv(r + 1, c)) if ax == RUN_Y else (lv(r, c - 1), lv(r, c + 1))
    if a is None or b is None or a == b:
        return                          # 이을 것이 없다. 평지로 남는다
    t = np.arange(n, dtype=np.float32) / (n - 1)
    line = (a + (b - a) * t) * HIGH
    patch[:] = line[:, None] if ax == RUN_Y else line[None, :]


def _band(n: int, width_m: float) -> tuple[int, int]:
    """랜드 한가운데를 가로지르는 띠의 격자 범위.

    칸 수를 짝수로 강제하지 않는다. 강제하면 폭이 `2 * CELL` 단위로만 나와서
    0.6 m를 넣어도 0.56 m가 됐다. 홀수를 허용하면 폭은 정확해지고 대신 띠의
    중심이 반 칸(0.02 m) 어긋난다. **폭이 더 중요하다.**

    주의 -- `CELL`로 나눠떨어지지 않는 폭은 여전히 정확할 수 없다. 0.5 m는
    12.5칸이라 13칸(0.52 m)이 된다. 정확히 0.5가 필요하면 `CELLS_PER_TILE`을
    올려야 한다 (100이면 CELL 0.02 m라 0.5도 0.6도 딱 맞는다).
    """
    w = max(1, int(math.floor(width_m / CELL + 0.5)))
    lo = (n - w) // 2
    return lo, lo + w


def _cross(patch: np.ndarray, width_m: float) -> np.ndarray:
    """랜드를 가로 · 세로로 모두 지르는 **+ 모양** 마스크.

    띠 하나로는 한 축밖에 못 막는다. y로 뻗은 턱은 x로 지날 때만 넘게 되고,
    **y로 지나가면 옆으로 비껴가서 안 넘어도 된다.** 도랑도 같다. 두 팔을 다
    두면 어느 쪽으로 지나든 반드시 하나를 만난다.

    두 팔이 겹치는 가운데를 두 번 세면 안 되므로 더하지 않고 마스크로 고른다.
    더하면 교차점만 두 배로 솟거나 두 배로 파인다.
    """
    n = patch.shape[0]
    lo, hi = _band(n, width_m)
    mask = np.zeros(patch.shape, dtype=bool)
    mask[:, lo:hi] = True
    mask[lo:hi, :] = True
    return mask


def _step(patch: np.ndarray) -> None:
    """턱 랜드. + 모양 상판을 올린다. **양 끝 높이는 같다.**

    올라섰다 내려오는 형태라 높이를 바꾸지 않는다. 높이가 둘 이상인 구조에서
    턱을 높이 차이로 만들면 0.713 m가 되어 넘을 수 없다.
    """
    patch[_cross(patch, STEP_SPAN)] += STEP_HEIGHT


def _gap(patch: np.ndarray) -> None:
    """gap 랜드. + 모양 도랑. **양 끝 높이는 같다.**

    구멍이 아니라 도랑인 이유 -- 높이 격자는 구멍을 못 뚫는다. 오히려 이쪽이
    낫다. 빠지면 바닥에 떨어지지 세상 밖으로 사라지지 않아서, 실패를 감지하고
    리셋하기 쉽다.
    """
    patch[_cross(patch, GAP_WIDTH)] -= GAP_DEPTH


def _wall(patch: np.ndarray) -> None:
    """벽 랜드. **랜드를 통째로 올려 막는다.**

    처음에는 턱처럼 띠로 만들었다. 모양은 그럴듯했지만 `blocked` 격자와 어긋났다
    -- 띠는 가로지르는 축만 막고 띠를 따라가는 축으로는 옆을 지나갈 수 있는데,
    `blocked`는 완전 차단으로 표시한다. **랜드 단위 격자는 "한 축만 막힘"을
    표현할 수 없다.**

    지도가 세상과 다르면 그 차이는 반드시 학습에서 튀어나온다. nav가 "막혔다"고
    한 곳을 로봇이 지나갈 수 있으면, 경로를 따라가라는 신호와 실제로 갈 수 있는
    방향이 어긋난다. 그래서 **막을 것이면 통째로 막는다.**

    면이 87.7도라 넘을 수 없고, 자기 높이 단 위로 WALL_HEIGHT만큼 솟는다.
    """
    patch += WALL_HEIGHT


def _pit(patch: np.ndarray) -> None:
    """절벽 랜드. **랜드 전체가 꺼져 있다.**

    벽과 하는 일은 같다 -- 못 지나간다. 다른 점은 들어갈 수는 있다는 것이다.
    떨어지면 1 m 아래라 걸어서 못 올라온다. 그래서 `blocked`에 막힘으로 적어도
    거짓말이 아니다. **지나가는 길**이 될 수 없기 때문이다.

    벽은 관문에만 쓰고 바깥 지형은 절벽으로 채운다. 그러면 위에서 봤을 때
    "여기는 관문이고 저기는 그냥 못 가는 땅"이 구별된다.
    """
    patch -= PIT_DEPTH


def _bridge(patch: np.ndarray, axis: int) -> None:
    """외나무다리. 위에서 보면 **대문자 I**다.

        ########      가로대.  변 전체가 이웃 높이와 같다
           ##
           ##         기둥.  여기만 밟고 건넌다
           ##
        ########      가로대

    가로대를 두는 이유 -- 기둥만 두면 이웃과 닿는 폭이 0.4 m뿐이라, 이웃 랜드
    쪽에서 보면 변의 대부분이 낭떠러지가 된다. 가로대가 변을 **끝까지 막아**
    이웃과 높이가 정확히 맞는다. 그래서 어느 랜드가 옆에 와도 이어붙는다.

    양옆은 파여 있으므로 **한 축으로만 지나갈 수 있다.** 옆에서 진입하려 하면
    떨어진다. 이것이 랜드 하나로 방향을 강제하는 유일한 종류다.

    막힘 격자에는 통과로 남긴다 -- 지나갈 수 있기 때문이다. 다만 nav 는 이
    랜드가 한 축만 허용한다는 것을 모른다. `kind` 를 같이 읽어야 한다.
    """
    n = patch.shape[0]
    ground = patch.copy()               # 파기 전 높이. 여기로 되돌린다
    patch -= BRIDGE_DROP
    bar = max(1, int(math.floor(BRIDGE_BAR / CELL + 0.5)))
    lo, hi = _band(n, BRIDGE_WIDTH)
    if axis == RUN_Y:                   # 기둥이 y 로 뻗는다 -> y 로 건넌다
        patch[:bar, :] = ground[:bar, :]
        patch[n - bar:, :] = ground[n - bar:, :]
        patch[:, lo:hi] = ground[:, lo:hi]
    else:
        patch[:, :bar] = ground[:, :bar]
        patch[:, n - bar:] = ground[:, n - bar:]
        patch[lo:hi, :] = ground[lo:hi, :]


def _tunnel(patch: np.ndarray, axis: int) -> None:
    """터널의 **옆벽만** 그린다. 천장은 높이 격자로 만들 수 없다.

    높이 함수는 같은 자리에 값이 하나뿐이라 위아래 두 겹을 못 만든다. 그래서
    천장은 박스 geom이고 `ceilings()`가 그 목록을 낸다. 지금까지 만든 랜드 중
    **유일하게 hfield만으로 끝나지 않는 종류다.**
    """
    n = patch.shape[0]
    lo, hi = _band(n, TUNNEL_WIDTH)
    if axis == RUN_Y:               # 통로가 y 로 뻗는다 -> y 로 지나간다
        patch[:, :lo] += TUNNEL_WALL
        patch[:, hi:] += TUNNEL_WALL
    else:
        patch[:lo, :] += TUNNEL_WALL
        patch[hi:, :] += TUNNEL_WALL


def ceilings(kind: np.ndarray, level: np.ndarray,
             axis: np.ndarray | None = None) -> np.ndarray:
    """천장 박스 목록. (N, 6) = 중심 xyz + 반크기 xyz. 없으면 (0, 6).

    **개수가 배치 전체에 고정된다.** geom은 모델 컴파일 때 확정되므로, 환경마다
    터널 수가 다른 미로를 한 배치에 섞을 수 없다. 지형을 하나만 두기로 한
    결정 덕에 지금은 문제가 안 된다.
    """
    if axis is None:
        axis = np.full(kind.shape, RUN_Y, dtype=np.int8)
    shape = kind.shape
    out = []
    for r, c in zip(*np.nonzero(kind == TUNNEL)):
        cx, cy = tile_center(int(r), int(c), shape)
        cz = float(level[r, c]) * HIGH + TUNNEL_CLEAR + TUNNEL_THICK / 2
        if int(axis[r, c]) == RUN_Y:
            hx, hy = TUNNEL_WIDTH / 2, TILE / 2
        else:
            hx, hy = TILE / 2, TUNNEL_WIDTH / 2
        out.append([cx, cy, cz, hx, hy, TUNNEL_THICK / 2])
    return np.asarray(out, dtype=np.float32).reshape(-1, 6)


def _rough(patch: np.ndarray, seed: int, r: int, c: int) -> None:
    """울퉁불퉁 랜드. **랜드 전체가 잔요철이다.**

    격자 간격(0.04 m)마다 난수를 넣으면 안 된다. 그건 백색잡음이라 발 크기보다
    작은 뾰족한 봉우리가 생기고, 접촉이 매 스텝 튀어 학습이 망가진다. 그래서
    성긴 격자에 난수를 놓고 **선형 보간으로 부풀린다.** 요철 하나가
    `ROUGH_FEATURE`(0.25 m)라 발이 빠지지 않는다.

    가장자리를 0으로 고정하는 것이 핵심이다. 안 그러면 이웃 랜드와 만나는 자리에
    최대 0.06 m 턱이 생긴다 -- 랜드를 이어붙이는 구조 전체가 거기서 깨진다.
    """
    n = patch.shape[0]
    k = max(3, int(round(TILE / ROUGH_FEATURE)) + 1)   # 성긴 격자점 수
    rng = np.random.default_rng((int(seed) * 7_919 + r * 131 + c + 1) & 0xFFFFFFFF)
    coarse = np.zeros((k, k), dtype=np.float32)
    coarse[1:-1, 1:-1] = rng.uniform(-1.0, 1.0, size=(k - 2, k - 2))
    patch += _upsample(coarse, n) * ROUGH_HEIGHT


def _upsample(coarse: np.ndarray, n: int) -> np.ndarray:
    """성긴 격자를 n x n으로 선형 보간. 모서리 값은 그대로 남는다."""
    k = coarse.shape[0]
    t = np.linspace(0.0, k - 1, n, dtype=np.float32)
    i0 = np.floor(t).astype(np.int32)
    i1 = np.minimum(i0 + 1, k - 1)
    f = (t - i0).astype(np.float32)
    rows = coarse[i0] * (1.0 - f)[:, None] + coarse[i1] * f[:, None]
    return rows[:, i0] * (1.0 - f)[None, :] + rows[:, i1] * f[None, :]


def _rock(patch: np.ndarray, seed: int, r: int, c: int) -> None:
    """돌맹이 랜드. 둥근 돌 몇 개를 흩는다. **지나갈 수 있다.**

    씨앗과 랜드 위치에서 자리를 뽑으므로 같은 미로면 같은 자리에 놓인다.
    돌 높이를 턱보다 낮게 두는 것은 의도다 -- 막는 것이 아니라 발을 걸리게 하는
    것이 목적이고, 막을 것이면 벽을 써야 한다.

    **반구를 쓰지 않는다.** 반구는 테두리에서 접선이 수직이다. 높이 0.14 반지름
    0.22로 격자에 얹으면 마지막 한 셀에서 0.0805 m가 떨어져 63.6도가 나온다 --
    돌 하나마다 둘레 전체에 8 cm짜리 턱이 생기는 셈이라, "모서리를 없애려고
    반구로 한다"는 원래 의도와 반대 결과였다.

    대신 융기 코사인을 쓴다. 중심과 테두리 양쪽에서 기울기가 0이라 이어붙는
    자리에 꺾임이 없고, 최대 경사가 `ROCK_FACE_DEG`로 정확히 정해진다.
    """
    n = patch.shape[0]
    rng = np.random.default_rng((int(seed) * 1_000_003 + r * 1_009 + c) & 0xFFFFFFFF)
    # 랜드 가장자리에 걸치면 이웃 랜드로 새어 나간다. 반지름만큼 안쪽으로 넣는다.
    margin = ROCK_RADIUS / TILE * n
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
    bumps = np.zeros_like(patch)
    for _ in range(ROCK_COUNT):
        cy, cx = rng.uniform(margin, n - margin, size=2)
        d = np.hypot(yy - cy, xx - cx) * CELL          # 중심까지 거리 (m)
        inside = d < ROCK_RADIUS
        # 융기 코사인. 테두리에서 높이도 기울기도 0이라 꺾임이 없다.
        one = np.zeros_like(patch)
        one[inside] = ROCK_HEIGHT * 0.5 * (
            1.0 + np.cos(np.pi * d[inside] / ROCK_RADIUS))
        # 겹치면 더한 게 아니라 **높은 쪽**. 더하면 두 겹친 돌이 두 배로 솟는다.
        bumps = np.maximum(bumps, one)
    patch += bumps


GROUND_RGB = (214, 209, 198)    #: 바닥 기본색
PATH_RGB = (120, 178, 236)      #: 정답지 색. 단색 하나뿐


def texture(path: np.ndarray) -> np.ndarray:
    """정답지만 칠한 (NROW, NCOL, 3) uint8. 높이 격자와 같은 크기다.

    종류마다 색을 달리 칠해봤지만 3D에서는 읽히지 않았다. 지형은 이미 모양으로
    구별되고, 색까지 다르면 눈이 어디를 봐야 할지 모른다. **색이 답해야 하는
    질문은 하나다 -- 어디가 정답지인가.**

    종류를 보고 싶으면 2D 지도나 출력된 표를 본다. 3D는 걷는 모습을 보는 곳이다.
    """
    ty, tx = path.shape
    n = CELLS_PER_TILE
    small = np.empty((ty, tx, 3), dtype=np.uint8)
    small[:] = GROUND_RGB
    small[path.astype(bool)] = PATH_RGB
    return np.repeat(np.repeat(small, n, axis=0), n, axis=1)


def blocked(kind: np.ndarray) -> np.ndarray:
    """랜드 표 -> nav의 점유 격자. hfield가 아니라 **랜드 단위**다.

    D*는 랜드 그래프 위에서 돈다. 격자 500칸 해상도로 경로를 찾을 이유가 없고,
    랜드 단위면 통로가 막혔는지가 정의상 자명하다.
    """
    out = np.zeros(kind.shape, dtype=bool)
    for k in IMPASSABLE:
        out |= kind == k
    return out


# ---------- 저장 ----------

#: npz를 해석하는 데 필요한 눈금. **높이 격자는 [0, 1]로 정규화된 값이라**
#: 이 상수들이 없으면 미터로 되돌릴 수 없다. 파일에 같이 넣고 읽을 때 대조한다.
#: 안 그러면 `maze.py`를 고친 뒤 예전 npz를 열었을 때 **지형이 조용히 달라진다.**
_SCALE = ("cell", "span", "depth", "tile")


def _scale_now() -> dict:
    return {"cell": CELL, "span": SPAN, "depth": DEPTH, "tile": TILE}


def save(maze: Maze, path) -> None:
    """`.npz`로 저장. `outputs/` 아래에 두고 git에는 넣지 않는다.

    이 파일 하나로 지형이 완전히 복원된다 -- 높이 격자, 천장 박스, 출발 · 목표 ·
    방향, 그리고 눈금까지 들어 있다. `maze.py`가 없어도 읽을 수 있고, 있으면
    눈금이 맞는지 대조한다.
    """
    np.savez_compressed(
        path, **_scale_now(),
        seed=maze.seed, gate=maze.gate, kind=maze.kind, level=maze.level,
        axis=maze.axis,
        height=maze.height, blocked=maze.blocked, ceiling=maze.ceiling,
        path_grid=maze.path, start_yaw=maze.start_yaw, route=maze.route,
        start=maze.start, goal=maze.goal,
    )


def load(path) -> Maze:
    """학습 시작할 때 한 번 읽는다. 읽는 쪽만 나중에 jax로 올린다.

    눈금이 지금 `maze.py`와 다르면 **여기서 멈춘다.** 그냥 읽으면 같은 배열이
    다른 미터로 해석되어, 경사가 완만해지거나 도랑이 얕아진 채로 학습이 돈다.
    """
    z = np.load(path)
    now = _scale_now()
    bad = {k: (float(z[k]), now[k]) for k in _SCALE
           if k in z.files and abs(float(z[k]) - now[k]) > 1e-9}
    if bad:
        raise ValueError(
            f"npz의 눈금이 지금 maze.py와 다릅니다: "
            + ", ".join(f"{k} 파일 {a} vs 코드 {b}" for k, (a, b) in bad.items())
            + ". 높이 격자는 [0,1] 정규화 값이라 눈금이 다르면 지형이 달라집니다."
        )
    return Maze(seed=int(z["seed"]), gate=float(z["gate"]), kind=z["kind"],
                level=z["level"],
                axis=z["axis"], height=z["height"], blocked=z["blocked"],
                ceiling=z["ceiling"], path=z["path_grid"],
                route=z["route"] if "route" in z else None,
                start_yaw=float(z["start_yaw"]),
                start=z["start"], goal=z["goal"])
