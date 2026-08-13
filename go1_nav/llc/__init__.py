"""llc — 사전학습 보행 정책. 여기서 학습하지 않는다.

동결된 정책을 **쓰기 위한 것만** 둔다.

    spec.py     명령 사양의 단일 출처. 순서 · 범위 · 실측 추종 특성. JAX 불필요
    loader.py   체크포인트 로드 + 추론 함수

스윕 · 자기검사 같은 측정 도구는 여기 없다. LLC 학습 노트북이 소유한다.

import 규칙: common 만 import 한다.
사양은 이 폴더의 README.md, 계약은 docs/01_contracts.md.
"""
