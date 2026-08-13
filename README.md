# 2026_han_go1

Go1 사족보행 로봇의 계층 제어. 경로(nav)를 받아 상위 제어기(HLC)가 지형을
넘어가고, 하위 제어기(LLC)가 관절을 움직인다.

끝에서 끝까지 도는 뼈대가 이미 있다. **안이 아직 더미일 뿐이다.**

---

## 지금 상태

```
경로 -> 변환 -> HLC -> LLC -> mujoco
직선    진짜   더미   phase14  평지
```

1000스텝(20초) 돌려 안 넘어지고 6.86 m 걷는다. 속도 0.343 m/s는 명령
`vx=0.4`에 실측 이득 0.921을 곱한 값과 맞는다 -- 명령이 끝까지 전달된다.

**로봇이 목표에 도착하지 않는 것이 정상이다.** HLC가 아직 경로를 안 본다.

| | 지금 | 누가 바꾸나 |
|---|---|---|
| `nav/stub.py` | 직선 두 점 | nav |
| `hlc/stub.py` | 고정 전진 | HLC |
| `hlc/env.py` | 평지 | HLC |
| `common/path.py` | **진짜** | 공유 · PR 필수 |
| `llc/` | **진짜** (phase14) | LLC |

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

`stub.py`의 **안만** 바꾼다. 함수 이름과 입출력은 안 건드린다. 그러면
나머지 전부가 그대로 돈다.

```python
nav   def path(world, robot_xy, goal_xy) -> (N, 2)   직선을 D*로
HLC   def act(features) -> (6,)                      고정값을 학습된 정책으로
```

`hlc/env.py`도 HLC 담당 것이다. 지형 · 목표 · 리셋 조건을 여기서 만든다.

---

## 폴더

```
go1_nav/
  common/path.py   경로 -> 숫자.  nav와 HLC가 공유하는 유일한 함수
  llc/spec.py      명령 11축의 단일 출처
  llc/loader.py    체크포인트 로드
  nav/stub.py      경로
  hlc/env.py       환경
  hlc/stub.py      정책
  loop.py          잇는 것만.  판단도 계산도 없다
docs/contracts.md  합의 없이 바꾸면 안 되는 것
```

import 방향은 [go1_nav/__init__.py](go1_nav/__init__.py)에 있다.

---

## 문서를 두 장만 두는 이유

사양은 **코드의 docstring이 소유한다.** `spec.py`를 열면 11축 표와 실측
추종값이 거기 있고, `path.py`를 열면 왜 방향과 거리를 쪼갰는지가 거기 있다.
같은 내용을 README에 또 적으면 코드만 고치고 문서는 안 고쳐서 어긋난다.

docstring이 못 담는 두 가지만 문서로 둔다.

```
README            파일을 열기 전에 알아야 하는 것
docs/contracts    어느 파일도 혼자 소유할 수 없는 것
```

---

## 채울 것

- 저장소 URL
- 담당자 배정 (nav / HLC / LLC)
- 체크포인트를 어디에 두고 어떻게 넘길지
