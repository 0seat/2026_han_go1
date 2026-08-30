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

# ---------- 헤드리스 렌더 ----------
#
# MuJoCo 는 GL 백엔드를 **`import mujoco` 시점에** 고른다. 콜랩처럼 화면이 없는
# 곳에서 그때 `MUJOCO_GL` 이 안 잡혀 있으면 GLFW(X11) 로 가고, 렌더할 때 이렇게
# 죽는다.
#
#     GLFWError: (65550) X11: The DISPLAY environment variable is missing
#     an OpenGL platform library has not been loaded into this process ...
#
# 물리는 멀쩡히 돌고 **영상 저장에서만** 터지기 때문에, 학습을 몇 시간 돌린 뒤
# 첫 평가에서 처음 만나기 쉽다. 영상을 필수로 삼은 이 프로젝트에서는 치명적이다.
#
# 그래서 패키지가 로드될 때 대신 잡아 준다. 조건을 좁게 둔 이유
#
#     이미 정해 놨으면      건드리지 않는다. 사람이 고른 것이 이긴다
#     리눅스 + DISPLAY 없음  헤드리스다. egl
#     그 밖 (윈도우 · 맥)    그냥 둔다. 강제로 egl 을 넣으면 로컬 렌더가 깨진다
#
# **주의 —** `import mujoco` 가 `import go1_nav` 보다 먼저면 늦는다. 노트북 첫
# 칸에서 직접 잡는 편이 확실하다.
#
#     import os; os.environ["MUJOCO_GL"] = "egl"
import os as _os
import sys as _sys

if "MUJOCO_GL" not in _os.environ:
    if _sys.platform.startswith("linux") and not _os.environ.get("DISPLAY"):
        _os.environ["MUJOCO_GL"] = "egl"
        _os.environ.setdefault("PYOPENGL_PLATFORM", "egl")


# ---------- jax 0.11 에서 없어진 이름 ----------
#
# flax 가 `jax.core.get_opaque_trace_state` 를 부른다.
#
#     flax/core/tracers.py:30    convention="flax"
#     flax/nnx/tracers.py:29     convention="nnx"
#
# 그 이름이 jax 0.10 에서 deprecated 되고 **0.11 에서 없어졌다.** 자리를
# `jax.extend.core` 로 옮겼을 뿐 하는 일은 같다.
#
# **주의 —** 이 기계의 pip 색인은 낡은 미러다. jax 0.6.2 · flax 0.10.7 을 최신이라
# 답하는데 콜랩엔 이미 더 높은 판이 있다. **로컬 색인으로 "최신"을 판정하지 말 것** --
# 그래서 "flax 를 올려서는 못 고친다"고 한 번 틀린 판정을 냈다.
#
# 콜랩은 jax 를 알아서 올린다. 로컬은 안 올라가므로 **같은 코드가 양쪽에서
# 돌게 하려면** 여기서 이어 주는 편이 낫다 -- 콜랩에서 jax 를 내려 깔면
# jaxlib · CUDA 플러그인까지 맞춰야 하고 런타임을 다시 띄워야 한다.
#
# **주의 —** 이것은 이름 하나를 이어 주는 것이지 버전 호환을 보장하지 않는다.
# flax 가 이 자리를 고친 판을 내면 지우는 것이 맞다.
try:
    import jax as _jax

    if not hasattr(_jax.core, "get_opaque_trace_state"):
        import jax.extend.core as _jec

        _jax.core.get_opaque_trace_state = _jec.get_opaque_trace_state
except Exception:      # jax 가 없는 환경(문서 빌드 등)에서 패키지가 죽지 않게
    pass


# ---------- brax 가 쓰는데 jax 가 지운 함수 ----------
#
# brax 의 학습 루프가 `jax.device_put_replicated` 를 부른다.
#
#     brax/training/pmap.py:27          brax/training/agents/ppo/train.py:751
#     brax/training/agents/apg:313      brax/training/agents/sac:113
#
# jax 가 이것을 지웠다. **경계가 0.11 이 아니다** -- 실측으로 0.6.2 에는 있고
# 0.10.2 · 0.11.1 에는 없다. `pmap` 은 0.11.1 에도 남아 있으므로, 콜랩에서
# 걸리는 것은 이 함수 하나다. "0.11 미만으로 내려라"로는 안 고쳐진다.
#
# 하는 일은 단순하다 -- pytree 의 잎마다 맨 앞에 장치 축을 붙이고 각 장치에
# 하나씩 올린다. 그다음 `pmap` 이 그 축을 먹는다. 장치가 하나면 축을 붙여
# 그 장치에 올리는 것이 전부다.
#
# **주의 —** 장치가 둘 이상이면 흉내 내지 않고 멈춘다. 그때는 잎마다 어느
# 조각이 어느 장치로 가는지가 중요해지는데, 그것을 여기서 대충 맞히면 학습이
# 조용히 틀린다. 콜랩은 GPU 한 장이라 이 경로로 안 온다.
try:
    import jax as _jax

    if not hasattr(_jax, "device_put_replicated"):
        import jax.numpy as _jnp

        def _device_put_replicated(value, devices):
            n = len(devices)
            if n != 1:
                raise RuntimeError(
                    f"장치가 {n} 개입니다. go1_nav 의 device_put_replicated "
                    f"대체는 한 장짜리만 다룹니다. jax 를 0.7 미만으로 내리세요 "
                    f'-- pip install -q "jax[cuda12]<0.7" "jaxlib<0.7"')

            def _rep(leaf):
                a = _jnp.asarray(leaf)
                return _jnp.broadcast_to(a, (n,) + a.shape)

            return _jax.device_put(_jax.tree.map(_rep, value), devices[0])

        _jax.device_put_replicated = _device_put_replicated
except Exception:
    pass


# ---------- 컴파일 캐시 ----------
#
# 로컬 테스트·영상의 비용은 거의 전부 **XLA 컴파일**이다. 실측 (8x16, 45차선,
# CPU) -- 첫 판 172 초, 그다음 판 9~14 초. 41 스텝에 죽은 판과 227 스텝을 완주한
# 판이 둘 다 345 초였던 적도 있는데, 그때는 `rollout` 이 판마다 `jax.jit` 을 새로
# 만들어 캐시가 늘 비어 있었다 (`hlc/stage1.py` 의 `_jitted` 참고).
#
# 그것을 고쳐도 **프로세스마다 한 번은 굽는다.** 그래서 영상 여러 편을 프로세스로
# 나눠도 이득이 거의 없다 -- 실측으로 네 편을 한 프로세스에 넣으면 442 초, 네
# 프로세스로 가르면 391 초다 (1.13 배). 각자 컴파일을 따로 물기 때문이다.
#
# jax 의 영속 캐시가 그 바닥값을 없앤다. 실측 -- 같은 스크립트를 두 번 띄웠을 때
#
#     1회차 (빈 캐시)    첫 판 172.2 초
#     2회차 (캐시 적중)  첫 판  28.3 초
#
# 남는 28 초는 캐시를 읽어 역직렬화하는 값이고 이건 프로세스마다 어쩔 수 없다.
# 캐시에 실제로 남는 것은 `jit_reset` · `jit_step` 두 항목, 3.3 MB 뿐이다.
#
# 끄려면 `GO1_JAX_CACHE=""`. 자리를 옮기려면 거기에 경로를 준다. 사람이 이미
# `JAX_COMPILATION_CACHE_DIR` 을 잡아 놨으면 건드리지 않는다.
#
# 열쇠는 코드 해시가 아니라 **XLA 가 받는 HLO** 다. 여기에 백엔드 · jaxlib 버전 ·
# 컴파일 옵션이 함께 들어간다. 그래서
#
#     콜랩 캐시를 로컬에서 쓸 수 없다   cuda/0.10 과 cpu/0.6.2 는 열쇠가 다르다
#     주석만 고치면 그대로 적중한다     HLO 가 안 변한다
#     env.py · obs.py 를 고치면 안 맞는다  HLO 가 변한다
#
# **미로 하나당 한 벌이 필요하다.** 지형 배열이 HLO 에 상수로 박히기 때문이다.
# 실측 -- 차선 수가 63 으로 같은 씨앗 1 과 24 가 서로 적중하지 않고 각자 168~178
# 초를 다시 구웠다. 크기도 그 값이다 (8x16 hfield 1.28 MB -> 항목 3.3 MB,
# 64x64 41 MB -> 27 MB). 반면 PRNGKey · 차선 번호는 캐시를 늘리지 않는다 --
# 추적 시점에 값이 아니라 인자로 흐른다.
#
# 따라서 씨앗을 여럿 도는 시험은 **씨앗당 프로세스 하나**로 묶는 것이 맞다.
# 섞어 나누면 같은 컴파일을 여러 번 문다.
#
# **주의 —** 컴파일러 플래그만 바뀌는 경우처럼 드물게 낡은 항목을 물 수 있다.
# 이상하면 캐시 폴더를 통째로 지우면 된다 -- 다시 구울 뿐이다.
try:
    _cache = _os.environ.get("GO1_JAX_CACHE")
    if _cache != "" and not _os.environ.get("JAX_COMPILATION_CACHE_DIR"):
        if not _cache:
            _base = (_os.environ.get("LOCALAPPDATA")
                     or _os.path.expanduser("~/.cache"))
            _cache = _os.path.join(_base, "go1_nav", "jax")

        # **쓸 수 있는지 지금 확인한다.** 콜랩에서 드라이브 자리를 줬는데 아직
        # 마운트 전이면 여기서 걸린다. 확인을 안 하면 config 는 조용히 성공하고
        # 몇 시간 뒤 첫 컴파일에서 학습이 죽는다.
        _ok = False
        try:
            _os.makedirs(_cache, exist_ok=True)
            _probe = _os.path.join(_cache, ".probe")
            with open(_probe, "w") as _f:
                _f.write("")
            _os.remove(_probe)
            _ok = True
        except OSError as _e:
            print(f"  주의 — 컴파일 캐시를 끕니다. {_cache} 에 못 씁니다 ({_e}). "
                  f"콜랩이면 드라이브를 먼저 마운트하세요.", flush=True)

        if _ok:
            import jax as _jax

            _jax.config.update("jax_compilation_cache_dir", _cache)
            _jax.config.update(
                "jax_persistent_cache_min_compile_time_secs", 1.0)
            # **크기 상한을 걸지 않는다.** `jax_compilation_cache_max_size` 를
            # 주면 LRU 축출이 켜지는데(`jax/_src/lru_cache.py` 의
            # `eviction_enabled = max_size != -1`), 그러면 쓸 때마다 기존 항목
            # **전부**의 `-atime` 짝을 읽는다. 축출이 꺼진 채로 만들어진 옛
            # 항목에는 그 짝이 없어서 FileNotFoundError 로 죽는다 -- 콜랩에서
            # 08-21 에 쌓인 항목들을 물고 캐시 쓰기가 전부 실패했다.
            #
            #     Error writing persistent compilation cache entry for
            #     'jit_reset': FileNotFoundError: .../pmap_reset-...-atime
            #
            # 학습은 계속 돌지만 캐시가 무용지물이 된다. 커지면 폴더를 지우면
            # 된다 -- 미로 하나당 한 벌(64x64 기준 27 MB)이다.
            # **캐시 사고로 학습을 죽이지 않는다.** 드라이브는 학습 도중에도
            # 끊긴다. 그때 할 일은 다시 굽는 것이지 몇 시간을 버리는 것이 아니다.
            _jax.config.update("jax_raise_persistent_cache_errors", False)
except Exception:
    pass
