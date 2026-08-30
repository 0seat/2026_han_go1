# 2026_han_go1

Go1 사족보행 로봇의 계층 제어. 경로(nav)를 받아 상위 제어기(HLC)가 지형을
넘어가고, 하위 제어기(LLC)가 관절을 움직인다.

끝에서 끝까지 도는 뼈대가 있고, 이제 **넘어갈 지형이 생겼다.**

---

## 지금 상태

```
경로 -> 변환 -> HLC -> LLC -> mujoco
직선    진짜   학습중  phase18  미로
```

| | 지금 | 누가 바꾸나 |
|---|---|---|
| `nav/stub.py` | 직선 두 점 | nav |
| `hlc/stub.py` | 고정 전진. `stage1` 이 대체하는 중 | HLC |
| `hlc/maze.py` | **진짜** | HLC |
| `hlc/env.py` | **진짜** (미로를 씬에 굽는다) | HLC |
| `hlc/stage1.py` | **진짜** (PPO 로 학습하는 상위 제어기) | HLC |
| `common/path.py` | **진짜** | 공유 · PR 필수 |
| `llc/` | **진짜** (phase18_speed_fwd) | LLC |

LLC 계보는 `phase11 -> 12 -> 13 -> 14 -> 18` 이다. 지금 쓰는 것은 **phase18**이고
자리는 [go1_nav/paths.py](go1_nav/paths.py)의 `LLC_PHASE` 한 곳에만 적는다.
phase15~17이 드라이브에 없어 18이 14의 heightfield 능력을 물려받았는지는
문서로 확인할 수 없다 -- 자세한 것은 [go1_nav/llc/spec.py](go1_nav/llc/spec.py).

미로 위를 걷는다. 8x16 미로에서 차선별 도달률 0.948까지 올렸고, 지금은 64x64
미로를 양방향으로 학습하는 중이다. 남은 문제는 **역방향 통과**다 -- 같은 지형을
반대로 지날 때 도달률이 0으로 떨어지는 차선이 있었다.

---

## 미로

랜드(2 m 정사각형)를 이어붙여 만든다. 랜드 하나가 장애물 하나다.

```
평지    경사 20도    턱 0.06 m   벽       도랑 0.5 m
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

## 환경 설정

**학습은 콜랩, 테스트와 영상은 로컬**로 나눈다. 로컬은 CPU라 8192 환경 학습을
못 돌리고, 콜랩은 렌더가 느리고 세션이 끊긴다.

### 로컬 (테스트 · 영상)

```bash
conda create -n mujoco_env python=3.11
conda activate mujoco_env
pip install mujoco==3.8.0 mujoco-mjx==3.8.0 playground==0.1.0
pip install brax flax optax jax jaxlib
pip install imageio imageio-ffmpeg mediapy matplotlib pillow numpy
```

**주의 —** `mujoco` 를 3.9 이상으로 올리면 `playground 0.1.0` 이 깨진다.
`nconmax` 가 `naconmax` 로 바뀌었다. 올리려면 playground 도 0.2.0 으로 같이
올린다.

실측 조합 (2026-08-30 기준)

```
python 3.11   mujoco 3.8.0   mjx 3.8.0   playground 0.1.0
jax 0.6.2     brax 0.14.1    flax 0.10.7
```

**주의 —** 사내 pip 미러가 낡은 판을 최신이라 답하는 경우가 있다. 로컬 색인으로
"최신"을 판정하지 말 것.

### 콜랩 (학습)

```python
import os; os.environ["MUJOCO_GL"] = "egl"
```

```python
from google.colab import drive; drive.mount('/content/drive')
!pip -q install mujoco==3.10.0 brax==0.14.2 playground==0.2.0 mediapy
import sys; sys.path.insert(0, '/content/drive/MyDrive/2026_han_go1')
```

`import go1_nav` 이 헤드리스 렌더와 jax 호환 문제를 알아서 잡는다. **`import
mujoco` 보다 먼저** 부를 것.

### 컴파일 캐시 (선택)

XLA 컴파일이 로컬 테스트 비용의 대부분이다. 영속 캐시를 켜면 프로세스를 새로
띄워도 다시 굽지 않는다.

```
기본        자동으로 켜진다 (%LOCALAPPDATA%\go1_nav\jax, 리눅스는 ~/.cache)
자리 옮기기  GO1_JAX_CACHE=<경로>
끄기        GO1_JAX_CACHE=""
```

미로 하나당 한 벌이 필요하다 -- 지형 배열이 컴파일 결과에 상수로 박힌다
(64x64 기준 27 MB). 콜랩 캐시와 로컬 캐시는 서로 못 읽는다. 백엔드와 jaxlib
버전이 열쇠에 들어간다.

---

## 체크포인트 (가중치)

구글 드라이브의 `go1_walking` 폴더에 있다. **저장소에는 안 들어간다** --
`.gitignore` 가 `*.pkl` 을 막는다.

```
go1_walking/
  phase18_speed_fwd/<실행>/final_checkpoint    LLC. 지금 쓰는 보행 정책
  hlc6/02_목표고침/params_latest.pkl           HLC. 8x16 미로
  hlc7/params_latest.pkl                       HLC. 64x64 미로 + 역방향
```

### 가져오기

**콜랩** -- 마운트하면 끝이다.

```python
from go1_nav import paths
paths.mount()
```

**로컬** -- 구글 드라이브 데스크톱을 켜 두면 `paths` 가 알아서 찾는다.
드라이브 문자와 흔한 자리를 훑어 `go1_walking` 폴더를 집는다.

```python
from go1_nav import paths
paths.walking()      # 찾은 자리를 낸다
```

못 찾으면 뿌리를 직접 준다. **`go1_walking` 자체가 아니라 그 부모**다.

```bash
set GO1_DRIVE=G:\.shortcut-targets-by-id\<id>      # 윈도우
export GO1_DRIVE=~/Google\ Drive                    # 리눅스 · 맥
```

폴더 접근 권한이 없으면 관리자에게 공유를 요청한다. 경로만으로는 안 열린다.

---

## 사용법

### 미로를 눈으로 본다

```bash
python tools/render_maze.py 11 6 16 0.4
```

`<씨앗> <세로 랜드> <가로 랜드> <관문 비율>`. `outputs/` 에 그림과 `.npz` 가
나온다. 파란 띠가 통과 규칙으로 찾은 최단 경로다.

### 정책을 표로 잰다

```bash
python tools/maze_test.py --체크포인트 hlc7/params_latest.pkl \
    --씨앗 0 --모양 64 64 --밀도 0.7 --역방향 --판수 2048
```

차선별 도달률 표가 나온다. 역방향 차선은 이름 앞에 `역·` 가 붙는다.

### 실패 장면을 영상으로 뽑는다

```bash
python tools/maze_test.py --체크포인트 hlc7/params_latest.pkl \
    --씨앗 0 --모양 64 64 --밀도 0.7 --역방향 --표없이 \
    --차선 3 10 25 30 --고를것 시간초과
```

`--고를것` 은 `fail` · `성공` · `시간초과` · `넘어짐` 이다. **`fail` 은 넘어짐과
시간초과를 구분하지 않는다** -- 흔한 실패를 보려면 `시간초과` 를 준다.

**차선을 여러 개 한 번에 준다.** 프로세스를 나누면 각자 컴파일을 다시 물어
느려진다. 실측으로 4편이 한 프로세스 442초, 네 프로세스 391초였다.

### 학습 (콜랩)

```python
from go1_nav import paths
from go1_nav.hlc import lands, maze, obs, stage1, train

assert obs.SIGNATURE == '4f5aa2e2400160b7'
params = train.load(paths.walking() / 'hlc7' / 'params_latest.pkl')

턱빼고 = tuple(k for k in maze.PLACED if k != maze.STEP)
mz = maze.generate(0, shape=(64, 64), kinds=턱빼고, density=0.7)
h, c, p = lands.maze_segments(mz, span=6, reverse=True)
task = stage1.Task({'height': h, 'ceiling': c, 'plan': p})

train.train(task, num_timesteps=100_000_000, num_envs=8192, num_evals=100,
            restore=params,
            video_dir=str(paths.walking() / 'hlc7'),
            stop_at=0.95, stop_patience=3)
```

**주의 — `video_dir` 을 비우면 체크포인트도 안 쌓인다.** 렌더가 싫으면
`video_dir` 은 주고 `render` 를 기본값(False)으로 두면 된다. 저장만 하고
녹화는 안 한다.

`num_evals` 는 저장 주기이자 **중단 가능 지점**이다. jit 한 판이 통째로 GPU 로
가므로 그 사이에는 인터럽트가 안 걸린다. 40으로 두면 한 판이 한 시간을 넘는다.

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
- LLC의 `footswing` 축이 열리면 턱 높이 재측정 (지금 0.06은 그 전의 한계)
