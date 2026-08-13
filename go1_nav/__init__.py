"""go1_nav — Go1 계층 제어 스택.

    common/path.py   경로 -> 숫자.  nav와 HLC가 공유하는 유일한 함수
    llc/             사전학습 보행 정책의 사양 · 로더
    nav/             경로
    hlc/             환경 · 상위 제어기 (PPO)
    loop.py          잇는 것만

import 방향을 지킬 것. 위쪽이 아래쪽을 import 하면 병렬이 깨진 것이다.

    common/   아무 내부 모듈도 import 하지 않는다
    llc/      common 만
    nav/      common 만
    hlc/      common, llc
    loop.py   전부

사양은 각 파일의 docstring이 소유한다. 계약은 docs/contracts.md.
"""
