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

새 파일을 어디에 둘 것인가

    지금 쓰는 사람이 한 명       그 사람 폴더
    두 명 이상이 실제로 import   common/
    자주 바뀐다                  common/ 에 두지 않는다
    애매하다                     자기 폴더

common/ 은 느리게 바뀌고 합의가 필요한 자리다. 자주 바뀌는 것을 여기 두면
고칠 때마다 남의 PR을 기다린다. **나중에 옮기는 것이 미리 놓는 것보다 싸다** --
git mv 한 번과 import 한 줄이다. 반대로 미리 놓으면 아직 없는 공유를 가정한
채로 굳는다.

옮겨서 common/ 에 들어가는 날, 그 형식이 docs/contracts.md 에 추가된다.
그때부터 바꾸려면 합의가 필요하다.

사양은 각 파일의 docstring이 소유한다. 계약은 docs/contracts.md.
"""
