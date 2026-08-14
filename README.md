# 2026_han_go1

Go1 사족보행 로봇의 계층 제어. 경로(nav)를 받아 상위 제어기(HLC)가 지형을
넘어가고, 하위 제어기(LLC)가 관절을 움직인다.

끝에서 끝까지 도는 뼈대가 있고, 이제 **넘어갈 지형이 생겼다.**

---

## 지금 상태

```
경로 -> 변환 -> HLC -> LLC -> mujoco
직선    진짜   더미   phase14  미로
```

| | 지금 | 누가 바꾸나 |
|---|---|---|
| `nav/stub.py` | 직선 두 점 | nav |
| `hlc/stub.py` | 고정 전진 | HLC |
| `hlc/maze.py` | **진짜** | HLC |
| `hlc/env.py` | **진짜** (미로를 씬에 굽는다) | HLC |
| `common/path.py` | **진짜** | 공유 · PR 필수 |
| `llc/` | **진짜** (phase14) | LLC |

평지에서 1000스텝(20초) 돌려 안 넘어지고 6.86 m 걷는다. 속도 0.343 m/s는 명령
`vx=0.4`에 실측 이득 0.921을 곱한 값과 맞는다 -- 명령이 끝까지 전달된다.

**아직 미로 위에서 걷는 것은 확인하지 못했다.** LLC 체크포인트가 필요하다.

---

## 미로

랜드(2 m 정사각형)를 이어붙여 만든다. 랜드 하나가 장애물 하나다.

```
평지    경사 20도    턱 0.2 m    벽       도랑 0.5 m
돌      거침         외나무다리   터널     절벽
```

높이는 단으로 센다. 한 단이 0.713 m이고 **경사 랜드 하나를 20도로 오른 높이**다.
높이를 바꾸는 랜드는 경사뿐이라, 이어붙일 때 변 높이가 어긋나지 않는다.

그래서 그래프가 저절로 나온다.

```
경사로 오른다        양방향
경사 없이 떨어진다   단방향.  내려갈 수는 있고 올라올 수는 없다
벽 · 도랑 · 절벽     간선 없음
```

만드는 법과 통과 규칙은 [go1_nav/hlc/maze.py](go1_nav/hlc/maze.py)의 docstring에
있다. 형식은 [docs/contracts.md](docs/contracts.md)의 C5.

### 보기

```bash
python tools/render_maze.py 11 6 16 0.4
```

`<씨앗> <세로 랜드> <가로 랜드> <관문 비율>`. `outputs/`에 그림 셋과 `.npz`가
나오고, 터미널에 랜드 표가 찍힌다. 파란 띠가 통과 규칙으로 찾은 최단 경로다.

렌더는 **CPU MuJoCo**다. mjx 렌더는 warp 백엔드 전용이고 이 저장소는
`impl="jax"`다. 물리는 mjx가 돌고 그림은 CPU가 그린다.

---

## 돌리기

```python
from go1_nav import loop
loop.run("<phase14 체크포인트 경로>", nsteps=1000, record=True)
```

통과 기준 세 줄이다.

```
안 넘어진다        요약의 넘어진_스텝이 -1
앞으로 간다        이동거리가 는다
영상에서 걷는다    record=True
```

---

## 각자 할 일

`stub.py`의 **안만** 바꾼다. 함수 이름과 입출력은 안 건드린다.

```python
nav   def path(world, robot_xy, goal_xy) -> (N, 2)   직선을 D*로
HLC   def act(features) -> (6,)                      고정값을 학습된 정책으로
```

nav의 D*는 `maze.reachable`과 **같은 통과 규칙**을 써야 한다. 규칙이 두 벌이면
"경로는 나왔는데 못 간다"가 된다.

---

## 폴더

```
go1_nav/
  common/path.py   경로 -> 숫자.  nav와 HLC가 공유하는 유일한 함수
  llc/spec.py      명령 11축의 단일 출처
  llc/loader.py    체크포인트 로드
  nav/stub.py      경로
  hlc/maze.py      미로 생성 · 통과 규칙 · 저장
  hlc/env.py       환경.  미로를 mujoco 씬에 굽는다
  hlc/stub.py      정책
  loop.py          잇는 것만.  판단도 계산도 없다
tools/render_maze.py   미로를 눈으로 보는 것
docs/contracts.md      합의 없이 바꾸면 안 되는 것
outputs/               산출물.  git에 안 올라간다
```

import 방향은 [go1_nav/__init__.py](go1_nav/__init__.py)에 있다.

---

## 문서를 두 장만 두는 이유

사양은 **코드의 docstring이 소유한다.** `spec.py`를 열면 11축 표와 실측
추종값이 거기 있고, `maze.py`를 열면 왜 턱이 직각이 될 수 없는지가 거기 있다.
같은 내용을 문서에 또 적으면 코드만 고치고 문서는 안 고쳐서 어긋난다.

docstring이 못 담는 두 가지만 문서로 둔다.

```
README            파일을 열기 전에 알아야 하는 것
docs/contracts    어느 파일도 혼자 소유할 수 없는 것
```

---

## 채울 것

- 담당자 배정 (nav / HLC / LLC)
- 체크포인트를 어디에 두고 어떻게 넘길지
- 터널 높이 · 턱 높이 · 다리 폭의 실측 (지금 값은 근거 없이 고른 것)
