"""통과 한계를 잰다 -- 0단계. **학습이 아니라 자질을 재는 것이다.**

이 파일이 답하는 질문은 둘이다.

    1  지금 LLC가 넘을 수 있는 턱은 몇 cm 인가        -> `maze.STEP_HEIGHT`가 된다
    2  아무것도 안 배운 정책은 미로에서 몇 %를 가나   -> PPO 곡선의 기준선

둘 다 학습 전에 있어야 한다. 기준선이 없으면 나중에 보상 곡선이 올라가도 그것이
학습인지 미로가 쉬운 것인지 구분할 방법이 없다.

영상은 선택이 아니다
--------------------

`video_dir`에 기본값을 두지 않았다. 부르는 쪽이 어디에 저장할지 반드시 정해야
돈다. 숫자만 보고 판정하면 **틀린 이유로 통과한 것을 못 잡는다** -- 이 프로젝트
에서 이미 겪었다. 발이 hfield를 뚫고 미끄러져 x가 늘어난 것을 "걸었다"로 읽거나,
십자 턱의 팔 사이로 빠져나간 것을 "넘었다"로 읽는 식이다. 표는 무엇이 일어났는지
말해주지 않는다.

모든 주행을 찍지는 않는다. **경계만** 찍는다 -- 같은 속도에서 마지막으로 성공한
높이와 처음 실패한 높이. 거기에 성공 기준점 하나를 더한다. 씨앗이 고정이라
측정과 영상이 같은 주행이다.
"""

from __future__ import annotations

import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from ..common import path as path_enc
from ..llc import loader, spec
from . import env as hlc_env
from . import lands, maze

#: 낙하·정착 구간. 이 동안의 `done`은 지형 탓이 아니라 home 자세에서 떨어지는 탓이다.
SETTLE = 25

#: 스텝 수를 자동으로 잡을 때의 여유 배수. 넘어질 뻔하다 회복하면 느려지므로 넉넉히.
TIME_MARGIN = 2.5

#: 스텝 수 상한. 자동 계산이 폭주하지 않게 자른다.
NSTEPS_MAX = 2000


def steps_for(distance_m: float, vx_command: float, ctrl_dt: float = 0.02) -> int:
    """거리와 명령 속도에서 필요한 스텝 수. **느린 속도의 가짜 실패를 막는다.**

    고정 스텝 수를 쓰면 0.3 m/s 명령이 시간이 모자라 실패로 찍힌다. 그러면 표에서
    "느리면 못 넘는다"로 읽히는데 사실은 "10초 안에 도착을 못 했다"이다.

    실측 이득을 곱하는 것이 요점이다. 명령 0.6은 실제 0.55 m/s다 (`spec.TRACKING`).
    """
    gain, bias, _ = spec.TRACKING["vx"]
    speed = max(abs(gain * float(vx_command) + bias), 1e-3)
    return int(min(NSTEPS_MAX, max(100, distance_m / speed / ctrl_dt * TIME_MARGIN)))


def _rollout(env, policy_fn, command, nsteps, seed, record=False, goal_x=None):
    """한 번 걷는다. 명령은 처음부터 끝까지 고정이다.

    고정 명령인 것이 요점이다. 여기서 재는 것은 **LLC의 자질**이지 상위 제어기의
    솜씨가 아니다. 접근 제어를 넣으면 둘이 섞여서 `STEP_HEIGHT`를 못 정한다.
    """
    reset, step = jax.jit(env.reset_at), jax.jit(env.step)
    with_command, infer = jax.jit(env.with_command), jax.jit(policy_fn)

    key = jax.random.PRNGKey(int(seed))
    key, sub = jax.random.split(key)
    # 부모의 무작위 배치를 쓰지 않는다. 복도를 따라 걷게 하려면 자세를 우리가
    # 정해야 한다 -- `reset_at`의 docstring 참고.
    state = with_command(reset(sub, xy=(0.0, 0.0), yaw=0.0),
                         jnp.asarray(command, jnp.float32))

    track = np.empty((nsteps, 3), dtype=np.float32)
    frames = [] if record else None
    fell = -1
    i = 0
    for i in range(nsteps):
        key, sub = jax.random.split(key)
        action, _ = infer(state.obs, sub)
        state = step(state, action)
        state = with_command(state, jnp.asarray(command, jnp.float32))
        # **스텝 뒤의** 위치를 적는다. 앞에서 적으면 마지막 한 스텝이 빠지고,
        # 하필 그것이 판정선을 넘는 스텝이라 통과가 실패로 찍힌다 -- 실제로 당했다.
        track[i] = np.asarray(state.data.qpos[0:3])
        if record:
            frames.append(state)
        if bool(state.done) and fell < 0 and i > SETTLE:
            fell = i
            break
        # 판정선을 지나면 거기서 끝낸다. 계속 걸리면 맵 밖으로 걸어 나가 떨어지고,
        # 그 넘어짐이 턱 실패로 기록된다 -- 이것도 실제로 당했다.
        if goal_x is not None and track[i, 0] > goal_x:
            break
    return track[:i + 1], fell, frames


def _verdict(track, fell, plan):
    """통과했는가. **세 가지를 다 만족해야 통과다.**

        판정선(턱 + 반 랜드)을 지났다
        그 전에 넘어지지 않았다
        차선 밖으로 나가지 않았다

    "그 전에"가 중요하다. 판정선을 지난 뒤의 넘어짐은 턱과 무관하다 -- 첫 측정에서
    h=0.04 가 턱을 넘고 계속 걸어 맵 밖으로 떨어졌고, 표에는 실패로 찍혔다.

    차선을 보는 이유는 옆으로 굴러 떨어지면서 x 가 늘어난 것을 통과로 세지
    않기 위해서다.
    """
    x, y, z = track[:, 0], track[:, 1], track[:, 2]
    goal = float(plan["goal_x"])
    over = np.nonzero(x > goal)[0]
    reached = int(over[0]) if over.size else -1
    before = track[:reached + 1] if reached >= 0 else track
    in_lane = bool(np.abs(before[:, 1]).max() < plan["lane_y"])
    fell_first = fell >= 0 and (reached < 0 or fell < reached)
    return {
        "통과": reached >= 0 and in_lane and not fell_first,
        "도달스텝": reached,
        "넘어짐": fell,
        "최대x": round(float(x.max()), 3),
        "판정선": round(goal, 3),
        "차선이탈": round(float(np.abs(before[:, 1]).max()), 3),
        "최저z": round(float(before[:, 2].min()), 3),
    }


def step_limit(heights, speeds, video_dir, *, checkpoint=None,
               nsteps=None, seed=0, verbose=True, **corridor):
    """턱 높이 x 접근 속도 격자를 재고, 경계 주행의 영상을 남긴다.

    `heights`는 미터. `maze.CELL`(0.04)의 배수로 주는 것이 좋다 -- 아니면 격자에
    얹히면서 실제 높이가 요청과 달라진다.

    `speeds`는 `vx` 명령 (m/s). 나머지 열 축은 `spec.BASELINE` 그대로다.

    `nsteps=None`이면 **속도마다 따로 잡는다**(`steps_for`). 고정으로 두면 느린
    명령이 시간이 모자라 실패로 찍히고, 표에서는 "느리면 못 넘는다"로 읽힌다.
    """
    video_dir = Path(video_dir)
    video_dir.mkdir(parents=True, exist_ok=True)
    ckpt = checkpoint
    if ckpt is None:
        from .. import paths
        ckpt = paths.llc()

    rows = []
    t0 = time.perf_counter()
    for h in heights:
        terrain, plan = lands.step_corridor(float(h), **corridor)
        env = hlc_env.make(terrain=terrain)
        policy_fn = loader.load_policy(ckpt, loader.env_observation_size(env))
        for v in speeds:
            command = list(spec.BASE_VECTOR)
            command[spec.index("vx")] = float(v)
            ns = nsteps or steps_for(plan["goal_x"], v)
            track, fell, _ = _rollout(env, policy_fn, command, ns, seed,
                                      goal_x=plan["goal_x"])
            row = {"높이": round(float(h), 3), "면각도": round(lands.face_degrees(h), 1),
                   "속도": float(v), **_verdict(track, fell, plan)}
            rows.append(row)
            if verbose:
                mark = "통과" if row["통과"] else "실패"
                print(f"  h={row['높이']:.2f}m {row['면각도']:5.1f}도  v={v:.1f}  "
                      f"{mark}  최대x {row['최대x']:+.2f}/{row['판정선']:+.2f}  "
                      f"넘어짐 {row['넘어짐']:4d}  이탈 {row['차선이탈']:.2f}  "
                      f"({ns}스텝)")
        del env
    if verbose:
        print(f"  ({time.perf_counter() - t0:.0f}초)")

    _record_boundaries(rows, video_dir, ckpt, nsteps, seed, verbose, corridor)
    return rows


def _record_boundaries(rows, video_dir, ckpt, nsteps, seed, verbose=True,
                       corridor=None):  # noqa: C901
    """경계 주행만 다시 돌려 영상으로 남긴다. 씨앗이 같아 같은 주행이다."""
    want = set()
    for v in sorted({r["속도"] for r in rows}):
        same = sorted([r for r in rows if r["속도"] == v], key=lambda r: r["높이"])
        ok = [r for r in same if r["통과"]]
        no = [r for r in same if not r["통과"]]
        if ok:
            want.add((ok[0]["높이"], v))          # 가장 낮은 성공 -- 정상 주행 기준점
            want.add((ok[-1]["높이"], v))         # 마지막 성공 -- 한계
        if no:
            want.add((no[0]["높이"], v))          # 첫 실패 -- 무엇이 무너지는가

    if verbose:
        print(f"영상 {len(want)}편 (경계만)")
    for h, v in sorted(want):
        terrain, _ = lands.step_corridor(float(h), **(corridor or {}))
        env = hlc_env.make(terrain=terrain)
        policy_fn = loader.load_policy(ckpt, loader.env_observation_size(env))
        command = list(spec.BASE_VECTOR)
        command[spec.index("vx")] = float(v)
        _, plan = lands.step_corridor(float(h), **(corridor or {}))
        ns = nsteps or steps_for(plan["goal_x"], v)
        _, _, frames = _rollout(env, policy_fn, command, ns, seed, record=True,
                                goal_x=plan["goal_x"])
        save_video(env, frames, video_dir / f"step_h{h:.2f}_v{v:.1f}.mp4")
        del env


def _progress(items):
    """tqdm 이 있으면 쓰고 없으면 그냥 돈다. 렌더는 느려서 진행 표시가 값을 한다."""
    try:
        import tqdm as _tqdm
        return _tqdm.tqdm(items)
    except Exception:
        return items


#: 렌더용 밝기. **전부 시각 전용 필드다.**
#:
#: `light_*` · `mat_*` · `geom_rgba` · `vis.*` 는 물리가 한 번도 읽지 않는다.
#: 접촉 · 마찰 · 질량과 무관하다. 게다가 물리는 `mjx.put_model` 로 올라간 모델이
#: 돌리고 렌더는 CPU `mj_model` 을 쓰므로, 여기서 고쳐도 시뮬레이션은 이미 끝난
#: 뒤다. 그래도 **모델을 복사해서** 고친다 -- 원본을 건드리면 같은 Task 를 쓰는
#: 다른 코드가 영향을 받는다고 의심할 여지가 생긴다.
#:
#: 원래 씬이 어두운 이유
#:
#:     조명이 하나뿐이고 (3, 0, 4) 에 고정이다. 복도가 14 x 10 m 라 가장자리까지
#:     안 닿는다. ambient 가 0 이라 그늘이 새까맣다
#:     지형 재질이 `groundplane` 인데 반사율 0.8 이라 어둡고 번들거린다
BRIGHT = {
    "headlight_ambient": 0.60,   # 전에 쓰던 값. 이것만으로는 부족했다
    "headlight_diffuse": 0.85,
    "headlight_specular": 0.05,
    "light_diffuse": 0.70,
    "light_ambient": 0.45,       # 원래 0.0 이라 그늘이 새까맸다
    "light_specular": 0.05,
    "ground_reflect": 0.0,       # 원래 0.8. 어둡고 번들거렸다
    # **`mat_rgba` 는 무늬가 있으면 안 먹는다** (실측: 값을 바꿔도 픽셀이 동일).
    # 그래서 색은 무늬 픽셀에 직접 넣는다. 아래 `checker` 가 그것이다.
    "ground_rgba": (1.0, 1.0, 1.0, 1.0),
    # 원래 텍스처가 평균 (97, 70, 49) 인 어두운 갈색이라 바닥이 탁했다. 떼면
    # 백지가 되어 굴곡을 읽을 단서가 사라진다. **무늬는 남기고 밝게 다시 그린다.**
    # 밝은 두 톤을 푸르게. 회색 무늬에 회색 로봇이면 서로 안 읽힌다.
    # 대비가 크면 굴곡보다 무늬가 먼저 읽히므로 두 톤을 가깝게 둔다.
    "checker": ((198, 208, 220), (170, 181, 196)),
    "checker_tiles": 8,
    "texrepeat": 6.0,
}


def brighten(model):
    """렌더용 밝은 사본. **물리에 영향이 없다.** 위 주석 참고."""
    import copy
    import mujoco

    m = copy.deepcopy(model)
    h = m.vis.headlight
    h.ambient[:] = BRIGHT["headlight_ambient"]
    h.diffuse[:] = BRIGHT["headlight_diffuse"]
    h.specular[:] = BRIGHT["headlight_specular"]
    for i in range(m.nlight):
        m.light_diffuse[i] = BRIGHT["light_diffuse"]
        m.light_ambient[i] = BRIGHT["light_ambient"]
        m.light_specular[i] = BRIGHT["light_specular"]
        m.light_castshadow[i] = 0        # 그림자가 지형 굴곡을 가린다
    # 지형 geom 의 재질만 손본다. 로봇 색은 그대로 둔다 -- 지형과 구별돼야 한다.
    for g in range(m.ngeom):
        if int(m.geom_type[g]) != int(mujoco.mjtGeom.mjGEOM_HFIELD):
            continue
        m.geom_rgba[g] = BRIGHT["ground_rgba"]
        mat = int(m.geom_matid[g])
        if mat >= 0:
            m.mat_reflectance[mat] = BRIGHT["ground_reflect"]
            m.mat_rgba[mat] = BRIGHT["ground_rgba"]
            m.mat_texrepeat[mat] = BRIGHT["texrepeat"]
            for tex in set(int(t) for t in m.mat_texid[mat]):
                if tex >= 0:
                    _repaint_checker(m, tex)
    return m


def _repaint_checker(m, tex):
    """텍스처 하나를 밝은 회색 체커로 다시 그린다. **픽셀만 바꾼다.**"""
    import numpy as _np

    w, h = int(m.tex_width[tex]), int(m.tex_height[tex])
    nch = int(m.tex_nchannel[tex]) if hasattr(m, "tex_nchannel") else 3
    adr = int(m.tex_adr[tex])
    hi, lo = (_np.asarray(c, _np.uint8) for c in BRIGHT["checker"])
    k = max(1, h // BRIGHT["checker_tiles"])
    rows = (_np.arange(h) // k)[:, None]
    cols = (_np.arange(w) // k)[None, :]
    even = ((rows + cols) % 2 == 0)[:, :, None]
    img = _np.where(even, hi[:nch], lo[:nch]).astype(_np.uint8)
    m.tex_data[adr:adr + w * h * nch] = img.reshape(-1)


def render_frames(env, states, camera="track", height=180, width=240,
                  bright=True):
    """상태 목록 -> 이미지 목록. **playground 의 `render_array` 를 대체한다.**

    왜 직접 쓰는가
    --------------

    `mjx_env.render_array` 가 프레임마다 이렇게 한다.

        d = mujoco.MjData(mj_model)      # 매 프레임 새로 할당
        d.qpos, d.qvel = ...
        mujoco.mj_forward(mj_model, d)   # 충돌 검출까지 전부

    그림에 필요한 것은 **정기구학뿐**이다. 그런데 `mj_forward` 는 250 x 150 격자
    높이맵에 대한 충돌 검출을 CPU 로 매 프레임 돌린다. 콜랩에서 프레임당 1 초가
    나온 이유가 이것이고, 200 프레임이면 3 분이다.

    그래서 둘을 고친다.

        MjData 를 한 번만 만든다
        mj_forward -> mj_kinematics + mj_camlight

    `update_scene` 이 쓰는 것은 `xpos` · `xquat` · `geom_xpos`(정기구학)와 카메라
    · 조명 위치(`mj_camlight`)다. 접촉점 시각화는 못 쓰게 되는데 우리가 안 쓴다.

    **주의 —** 물리를 다시 계산하지 않으므로 `states` 안의 자세를 그대로 믿는다.
    그게 맞다 -- 이미 시뮬레이션이 끝난 궤적을 그리는 것이지 다시 굴리는 것이
    아니다.
    """
    import mujoco

    m = brighten(env.mj_model) if bright else env.mj_model
    renderer = mujoco.Renderer(m, height=height, width=width)
    d = mujoco.MjData(m)
    cam = camera if camera is not None else -1
    out = []
    try:
        for state in _progress(states):
            d.qpos[:] = np.asarray(state.data.qpos)
            d.qvel[:] = np.asarray(state.data.qvel)
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)      # track 카메라가 subtree_com 을 본다
            mujoco.mj_camlight(m, d)
            renderer.update_scene(d, camera=cam)
            out.append(renderer.render())
    finally:
        renderer.close()
    return out


def save_video(env, states, filename, fps=50, stride=2, camera="track",
               height=180, width=240, bright=True):
    """영상 저장. 세 단계로 물러난다.

        1  mediapy         PATH 의 ffmpeg 을 쓴다. 콜랩은 여기서 끝난다
        2  imageio         `imageio_ffmpeg` 번들 바이너리를 직접 지정한다
        3  GIF             둘 다 없을 때. 크고 화질이 나쁘지만 없는 것보다 낫다

    2단계가 있는 이유 -- 윈도우 로컬에는 ffmpeg 이 PATH 에 없는 것이 보통인데
    `imageio_ffmpeg` 패키지가 실행파일을 들고 있다. mediapy 는 PATH 만 보므로
    그대로 두면 멀쩡한 인코더를 놔두고 GIF 로 떨어진다.

    `camera="track"` 이 아니면 로봇이 화면 밖으로 걸어 나가 아무것도 안 보인다.

    **렌더가 물리보다 비싸다.** 복도 지형이 150 x 250 격자라 하이트필드가 37,500
    칸이고 매 프레임 삼각형으로 펼쳐진다. 콜랩에서 프레임당 1.2 초가 나왔다 --
    200 프레임이면 4.6 분으로, 평가 자체(104초)보다 오래 걸린다.
    그래서 기본 해상도를 playground 기본값(240x320)보다 낮춰 잡았다. 무엇이
    일어났는지 보는 데는 이걸로 충분하다.
    """
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    # 밝기는 `render_frames` 가 사본에 건다 (`brighten`). 전에는 여기서 원본
    # 모델의 헤드라이트를 직접 고쳤는데, 그 값(ambient 0.6 / diffuse 0.8)으로도
    # 어두웠다. **원인은 헤드라이트가 아니라 지형 재질과 조명 ambient 였다.**
    rendered = render_frames(env, states[::stride], camera=camera,
                             height=height, width=width, bright=bright)
    out_fps = max(fps // stride, 1)

    try:
        import mediapy
        mediapy.write_video(str(filename), rendered, fps=out_fps)
        print(f"  영상 {filename.name}")
        return filename
    except Exception:
        pass

    try:
        import imageio, imageio_ffmpeg, os
        os.environ.setdefault("IMAGEIO_FFMPEG_EXE", imageio_ffmpeg.get_ffmpeg_exe())
        imageio.mimwrite(str(filename), list(rendered), fps=out_fps,
                         codec="libx264", quality=8)
        print(f"  영상 {filename.name}")
        return filename
    except Exception as exc:
        print(f"[measure] mp4 실패({type(exc).__name__}), GIF 로 저장합니다.")

    from PIL import Image
    filename = filename.with_suffix(".gif")
    img = [Image.fromarray(f) for f in rendered]
    img[0].save(filename, save_all=True, append_images=img[1:],
                duration=int(1000 / out_fps), loop=0)
    print(f"  영상 {filename.name}")
    return filename


def table(rows) -> str:
    """행들을 사람이 읽는 표로. 노트북이 배열을 되돌려받지 않게 문자열로 낸다."""
    speeds = sorted({r["속도"] for r in rows})
    out = ["", "턱 통과 한계  (행 = 높이, 열 = vx 명령)", ""]
    out.append("  높이   면각도  " + "  ".join(f"{v:>5.1f}" for v in speeds))
    for h in sorted({r["높이"] for r in rows}):
        deg = next(r["면각도"] for r in rows if r["높이"] == h)
        cells = []
        for v in speeds:
            r = next((r for r in rows if r["높이"] == h and r["속도"] == v), None)
            cells.append("    O" if r and r["통과"] else "    X")
        out.append(f"  {h:.2f}  {deg:5.1f}도  " + "  ".join(cells))
    ok = [r["높이"] for r in rows if r["통과"]]
    out += ["", f"  통과한 최대 높이  {max(ok):.2f} m" if ok else "  통과한 높이가 없다"]
    return "\n".join(out)


# ---------- 차선별로 갈라 보기 ----------

#: 표에서 빼는 지형. 어느 구간에나 있어서 적어봐야 자리만 먹는다.
_COMMON = (maze.FLAT, maze.WALL)


def _lane_name(plan, lane: int) -> str:
    """차선 이름표. **구간 구성을 보여준다.**

    `lane_kind` 하나만 찍으면 안 된다 -- 그 값은 "처음 만나는 장애물"이라,
    `generate(density=...)` 로 꺾임 칸을 돌 · 거침으로 채우면 어느 구간이든
    돌 아니면 거침으로 나온다. 실측 -- 터널 다섯 칸이 있는 판인데 23차선 표에
    터널이 한 번도 안 찍혔다.

    드문 것부터 적는다. 터널 · 다리가 있는 구간인지가 표를 읽는 핵심이다.
    """
    rev = plan.get("lane_reverse")
    # 역방향 차선은 앞에 표시한다. 지형 구성은 정방향 짝과 같아서 구성만으로는
    # 구분이 안 되는데, 실측상 성적이 전혀 다르다 (씨앗 0 에서 정방향 경사
    # 차선은 1.000, 같은 지형의 역방향은 0.000).
    head = "역·" if rev is not None and bool(np.asarray(rev)[lane]) else ""

    counts = plan.get("lane_counts")
    if counts is None:
        return head + maze.NAMES.get(int(np.asarray(plan["lane_kind"])[lane]), "?")
    row = np.asarray(counts)[lane]
    rare = (maze.TUNNEL, maze.BRIDGE, maze.RAMP, maze.STEP, maze.GAP,
            maze.PIT, maze.ROCK, maze.ROUGH)
    out = []
    for k in rare:
        c = int(row[maze.IMPLEMENTED.index(k)])
        if c and k not in _COMMON:
            out.append(maze.NAMES[k] + (str(c) if c > 1 else ""))
    return head + ("·".join(out[:3]) if out else "평지")


def lane_report(task, policy, *, n: int = 128, seed: int = 0, nsteps=None,
                표: bool = True, batch: int = 128):
    """차선마다 도달·넘어짐·시간초과를 따로 낸다. **평균은 거짓말을 한다.**

    실측 -- 꺾임 뒤 장애물 판에서 전체 도달이 0.977 이었다. 차선이 넷이니
    한 차선이 0.91 이어도 같은 숫자가 나온다. 그 숫자를 보고 "경사도 잘 된다"로
    읽었는데 근거가 없었다. 무엇이 안 되는지 물으려면 갈라야 한다.

    실패를 셋으로 나눈다. 셋은 고치는 방법이 서로 다르다.

        넘어짐      지형을 못 넘었다        -> LLC 한계이거나 진입 자세
        시간초과    안 넘어졌는데 못 갔다   -> 막혀 서 있다. 대개 벽으로 읽은 것
        발산        물리가 터졌다           -> 판이 잘못됐다. 보상 문제가 아니다

    `policy` 는 `train.policy(params, task)` 가 준 `(obs, key) -> action` 이다.
    에피소드 `n` 개를 vmap 으로 한 번에 굴린다. 끝난 에피소드는 상태를 얼려
    뒤 스텝이 값을 덮지 않게 한다 -- 여기엔 brax 래퍼가 없다.

    반환값은 행 목록이고 표는 이미 찍었다. 노트북이 배열을 되돌려받지 않는다.
    """
    import jax
    import jax.numpy as jnp
    import numpy as np

    from . import stage1

    nsteps = int(stage1.MAX_STEPS if nsteps is None else nsteps)

    def episode(key):
        k0, k1 = jax.random.split(key)
        st = task.reset(k0)

        def body(carry, _):
            st, key, live = carry
            key, sub = jax.random.split(key)
            nxt = task.step(st, policy(st.obs, sub))
            # 끝난 판은 얼린다. `live` 가 0 이면 이전 상태를 그대로 들고 간다.
            st = jax.tree_util.tree_map(
                lambda a, b: jnp.where(live > 0.5, b, a), st, nxt)
            return (st, key, live * (1.0 - st.done)), live

        (st, _, _), lived = jax.lax.scan(
            body, (st, k1, jnp.ones(())), None, length=nsteps)
        return {
            "lane": st.info["lane"],
            "도달": st.metrics["도달"],
            "넘어짐": st.metrics["넘어짐"],
            "빠짐": st.metrics["빠짐"],
            "발산": st.metrics["발산"],
            "스텝": jnp.sum(lived),
            "목표거리": st.metrics["목표거리"],
        }

    # **폭을 늘리지 말고 여러 번 부른다.** vmap 폭이 HLO 에 들어가므로 폭이
    # 곧 컴파일 캐시의 열쇠다. n 을 키우면 캐시가 통째로 빗나가고, 컴파일 자체도
    # 폭에 따라 급히 무거워진다. 실측 -- n=128 은 컴파일까지 325 초인데 n=2832 는
    # 21 분이 지나도 컴파일이 안 끝났고 24 코어 중 1.7 개만 돌았다.
    #
    # 조각으로 나누면 실행체 하나를 n/batch 번 재사용한다. 두 번째 부름부터는
    # 컴파일이 없고, 다음 실행에서도 캐시가 맞는다.
    run = jax.jit(jax.vmap(episode))
    keys = jax.random.split(jax.random.PRNGKey(int(seed)), int(n))
    batch = int(batch) if batch else int(n)
    parts, t0 = [], time.perf_counter()
    for i in range(0, int(n), batch):
        k = keys[i:i + batch]
        take = k.shape[0]
        if take < batch:
            # **마지막 조각도 폭을 맞춘다.** 안 맞추면 그 조각만 다시 컴파일한다.
            k = jnp.concatenate([k, keys[:batch - take]])
        parts.append((jax.device_get(run(k)), take))
        print(f"    {min(i + batch, int(n)):>6} / {int(n)}판   "
              f"{time.perf_counter() - t0:5.0f}초", flush=True)
    out = {key: np.concatenate([np.asarray(part[key])[:take]
                                for part, take in parts])
           for key in parts[0][0]}

    rows = []
    for lane in range(task.n_lanes):
        m = out["lane"] == lane
        if not m.any():
            continue
        reach, fell, sunk = out["도달"][m], out["넘어짐"][m], out["빠짐"][m]
        # 빠지면서 넘어지는 판이 있다. 한 번만 센다 -- 넘어짐 쪽으로 몰아준다.
        sunk = sunk * (1 - fell)
        rows.append({
            "차선": lane,
            "랜드": _lane_name(task.plan, lane),
            "판수": int(m.sum()),
            "도달": float(reach.mean()),
            "넘어짐": float(fell.mean()),
            "빠짐": float(sunk.mean()),
            "발산": float(out["발산"][m].mean()),
            # 안 넘어지고 안 빠졌는데 못 간 것. 넷을 더하면 1 이 된다.
            "시간초과": float(((1 - reach) * (1 - fell) * (1 - sunk)).mean()),
            "스텝": float(out["스텝"][m].mean()),
            "남은거리": float(out["목표거리"][m].mean()),
        })
    if 표:
        print(lane_table(rows, total=float(out["도달"].mean())))
    else:
        print(f"\n  전체 도달  {float(out['도달'].mean()):.3f}")
    return rows


def lane_table(rows, total=None) -> str:
    """`lane_report` 의 행들을 사람이 읽는 표로."""
    out = ["", "  차선  랜드   판수    도달  넘어짐   빠짐  시간초과    발산   스텝  남은거리", ""]
    for r in rows:
        out.append(f"  {r['차선']:>4}  {r['랜드']:<5} {r['판수']:>4}   "
                   f"{r['도달']:.3f}   {r['넘어짐']:.3f}  {r['빠짐']:.3f}     "
                   f"{r['시간초과']:.3f}   "
                   f"{r['발산']:.3f}  {r['스텝']:5.0f}     {r['남은거리']:.2f}")
    if total is not None:
        out += ["", f"  전체 도달  {total:.3f}"]
    return "\n".join(out)


def merge_rows(*row_lists):
    """여러 프로세스가 낸 차선 표를 하나로 합친다. **판수로 가중한다.**

    각 행의 숫자는 이미 그 프로세스의 판에 대한 평균이라, 판수를 무게로 주면
    합친 평균이 전체 평균과 정확히 같다.

    왜 프로세스를 나누는가 -- MJX 를 CPU 에서 vmap 128 로 굴리면 24 코어 중
    1.5 개밖에 안 쓴다. 폭을 늘리면 캐시가 빗나가고 컴파일이 폭발하므로
    (`lane_report` 의 batch 주석), 남는 코어는 **프로세스로** 쓴다. 컴파일이
    캐시에 있으면 프로세스마다 다시 안 문다.

    프로세스마다 `lane_report(seed=...)` 를 다르게 준다. 같으면 같은 판을
    두 번 굴리는 것이라 판수만 늘고 정보는 안 는다.
    """
    import numpy as np

    KEYS = ("도달", "넘어짐", "빠짐", "발산", "시간초과", "스텝", "남은거리")
    bag: dict[int, list] = {}
    for rows in row_lists:
        for r in rows:
            bag.setdefault(int(r["차선"]), []).append(r)

    out = []
    for lane in sorted(bag):
        rs = bag[lane]
        w = np.array([r["판수"] for r in rs], np.float64)
        tot = w.sum()
        merged = {"차선": lane, "랜드": rs[0]["랜드"], "판수": int(tot)}
        for k in KEYS:
            merged[k] = float(np.dot([r[k] for r in rs], w) / max(tot, 1.0))
        out.append(merged)
    return out

def lane_spread(rows) -> str:
    """차선별 도달률의 **분포**. 평균 하나로는 두 세계가 구분이 안 된다.

        전 차선이 0.75            -> 아직 덜 배웠다. 더 돌리면 오른다
        3/4 는 1.0, 1/4 는 0.0    -> 못 가는 차선이 따로 있다. 더 돌려도 그대로

    전체 도달률은 둘 다 0.75 로 같다. PPO 곡선이 정체로 보일 때 어느 쪽인지
    먼저 갈라야 한다 -- 뒤쪽이면 보상이나 스텝 수를 만져도 안 움직인다.
    """
    import numpy as np

    v = np.array([r["도달"] for r in rows], np.float64)
    w = np.array([r["판수"] for r in rows], np.float64)
    edges = [0.0, 0.001, 0.25, 0.5, 0.75, 0.999, 1.001]
    names = ["0.00", "~0.25", "~0.50", "~0.75", "~1.00", "1.00"]
    out = ["", "  차선 도달률 분포", ""]
    for i, name in enumerate(names):
        m = (v >= edges[i]) & (v < edges[i + 1])
        out.append(f"  {name:>6}  차선 {int(m.sum()):>4}  "
                   f"({m.mean() * 100:5.1f}%)  판 {int(w[m].sum()):>5}")
    out.append("")
    out.append(f"  차선 평균 {v.mean():.3f}   중앙값 {np.median(v):.3f}")
    return "\n".join(out)


def lane_groups(rows, *, min_lanes: int = 4) -> str:
    """차선을 **지형 포함 여부로** 묶는다. 차선 하나당 판이 몇 개뿐일 때 쓴다.

    구성 문자열 그대로 묶으면 안 된다 -- 씨앗 0 의 64x64 는 차선 944 개에 구성이
    295 가지라 대부분이 한 차선짜리 묶음이 된다. 그래서 "터널이 있는 차선",
    "다리가 있는 차선" 처럼 **겹치게** 센다. 한 차선이 여러 줄에 들어간다.

    방향도 함께 가른다. 같은 지형이라도 정 · 역 성적이 다르다는 것이 이미
    실측돼 있다 (`_lane_name` 주석).
    """
    import re

    import numpy as np

    def kinds_of(name):
        body = name[2:] if name.startswith("역·") else name
        return {re.sub(r"[0-9]+$", "", t) for t in body.split("·") if t}

    tags = sorted({k for r in rows for k in kinds_of(r["랜드"])})
    lines = ["", "  지형    방향    차선   판수    도달  넘어짐   빠짐  시간초과   남은거리", ""]
    agg = []
    for tag in tags:
        for 방향, want in (("정", False), ("역", True)):
            rs = [r for r in rows
                  if tag in kinds_of(r["랜드"])
                  and r["랜드"].startswith("역·") == want]
            if len(rs) < min_lanes:
                continue
            w = np.array([r["판수"] for r in rs], np.float64)
            tot = max(w.sum(), 1.0)

            def wm(key, rs=rs, w=w, tot=tot):
                return float(np.dot([r[key] for r in rs], w) / tot)

            agg.append((wm("도달"), tag, 방향, len(rs), int(w.sum()),
                        wm("넘어짐"), wm("빠짐"), wm("시간초과"), wm("남은거리")))
    for 도달, tag, 방향, n_lane, n_ep, fell, sunk, over, dist in sorted(agg):
        lines.append(f"  {tag:<6} {방향:<4} {n_lane:>6}  {n_ep:>5}   {도달:.3f}   "
                     f"{fell:.3f}  {sunk:.3f}     {over:.3f}     {dist:5.2f}")
    lines.append("")
    lines.append("  한 차선이 여러 줄에 들어간다. 합계는 차선 수와 다르다.")
    return "\n".join(lines)


def direction_split(rows) -> str:
    """정방향 · 역방향으로만 가른 요약. **역방향이 천장인지 먼저 본다.**"""
    import numpy as np

    lines = ["", "  방향    차선   판수    도달  넘어짐   빠짐  시간초과", ""]
    for tag, want in (("정방향", False), ("역방향", True)):
        rs = [r for r in rows if r["랜드"].startswith("역·") == want]
        if not rs:
            continue
        w = np.array([r["판수"] for r in rs], np.float64)
        tot = max(w.sum(), 1)
        def wm(key):
            return float(np.dot([r[key] for r in rs], w) / tot)
        lines.append(f"  {tag}  {len(rs):>6}  {int(w.sum()):>5}   {wm('도달'):.3f}   "
                     f"{wm('넘어짐'):.3f}  {wm('빠짐'):.3f}     {wm('시간초과'):.3f}")
    return "\n".join(lines)


# ---------- 지도 그리기 ----------

#: 랜드 종류별 색. **평지와 벽의 대비를 크게** 둔다. 나머지는 서로 구분만 되면 된다.
KIND_COLOR = {
    maze.FLAT: "#e8e8e8",
    maze.WALL: "#3a3a3a",
    maze.PIT: "#1b1b1b",
    maze.GAP: "#5b6b8a",
    maze.RAMP: "#e0b062",
    maze.STEP: "#c98a5a",
    maze.ROCK: "#9b8f7a",
    maze.ROUGH: "#c3c0a8",
    maze.BRIDGE: "#7fa8d0",
    maze.TUNNEL: "#8f7fb8",
}


def plot_map(mz, plan=None, filename=None, *, title=None, dpi=140,
             lanes=None, zoom=None, scale=0.62):
    """미로를 위에서 본 그림. **정답지와 출발점을 같이 찍는다.**

    표로는 "출발점이 5개"까지만 보인다. 그 다섯이 미로의 어디인지, 어느 쪽을
    보고 서는지, 앞에 무엇이 있는지는 그림이라야 한다. 커리큘럼 단계를 올릴 때
    출발점이 실제로 뒤로 물러났는지도 여기서 확인한다.

    `plan` 은 `lands.maze_segments` 가 낸 것이다. 없으면 지형과 정답지만 그린다.

    `lanes`
        그릴 차선 번호. `None` 이면 전부다. **큰 지도에서는 반드시 줄인다** --
        64x64 양방향이 944 차선이라 전부 그리면 라벨이 서로 덮어 아무것도 안
        읽힌다. 빈 목록을 주면 화살표 없이 지형과 정답지만 그린다.

    `zoom`
        `(행0, 행1, 열0, 열1)` 타일 범위. 큰 지도의 한 구석을 확대해 차선이
        실제로 어디서 출발하는지 보는 용도다.

    `scale`
        타일 한 변이 그림에서 차지하는 인치. 64x64 를 기본값으로 그리면 한 변이
        40 인치가 된다.

    반환값 -- 저장한 경로. 노트북이 figure 객체를 되돌려받지 않는다.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.patches import Rectangle

    # 한글 글꼴. 없으면 라벨이 전부 두부(네모)로 나온다 -- 그러면 그림의 절반이
    # 쓸모없다. 있는 것 중 하나를 고르고, 없으면 그 사실을 알린다.
    have = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Malgun Gothic", "NanumGothic", "AppleGothic", "Noto Sans CJK KR"):
        if name in have:
            plt.rcParams["font.family"] = name
            break
    else:
        print("  주의 -- 한글 글꼴이 없어 라벨이 깨집니다")
    plt.rcParams["axes.unicode_minus"] = False      # 마이너스도 두부가 된다

    ty, tx = mz.kind.shape
    r0, r1, c0, c1 = (0, ty, 0, tx) if zoom is None else [int(v) for v in zoom]
    r0, r1 = max(0, r0), min(ty, r1)
    c0, c1 = max(0, c0), min(tx, c1)
    # 글자는 타일이 충분히 클 때만 읽힌다. 작으면 지워서 색만 남긴다.
    text_on = scale >= 0.4
    fig, ax = plt.subplots(figsize=((c1 - c0) * scale,
                                    (r1 - r0) * scale + 1.1))

    for r in range(r0, r1):
        for c in range(c0, c1):
            k = int(mz.kind[r, c])
            xy = maze.tile_center(r, c, mz.kind.shape)
            ax.add_patch(Rectangle(
                (xy[0] - maze.TILE / 2, xy[1] - maze.TILE / 2),
                maze.TILE, maze.TILE,
                facecolor=KIND_COLOR.get(k, "#ff00ff"),
                edgecolor="white", linewidth=0.6))
            lv = int(mz.level[r, c])
            if lv and text_on:           # 단이 0 이 아니면 숫자를 적는다
                ax.text(xy[0] + maze.TILE / 2 - 0.15, xy[1] + maze.TILE / 2 - 0.2,
                        str(lv), fontsize=6, color="#444", ha="right", va="top")
            if text_on and k not in (maze.FLAT, maze.WALL, maze.PIT):
                ax.text(*xy, maze.NAMES_EN.get(k, "?"), fontsize=7,
                        ha="center", va="center", color="#222")

    if mz.route is not None and len(mz.route):
        pts = mz.route_xy
        ax.plot(pts[:, 0], pts[:, 1], color="#d03030", linewidth=1.6,
                alpha=0.75, zorder=3)
        for i in np.asarray(mz.turns):   # 꺾이는 칸
            ax.plot(*pts[int(i)], marker="o", markersize=3.5,
                    color="#d03030", zorder=4)

    ax.plot(*mz.goal, marker="*", markersize=16, color="#1a7f1a", zorder=6)
    ax.text(mz.goal[0], mz.goal[1] - 0.55, "GOAL", fontsize=7, ha="center",
            color="#1a7f1a", zorder=6)

    if plan is not None:
        starts = np.asarray(plan["lane_start_xy"])
        yaws = np.asarray(plan.get("lane_yaw", np.zeros(len(starts))))
        tiles = np.asarray(plan.get("lane_tiles", np.arange(len(starts))))
        rev = np.asarray(plan.get("lane_reverse", np.zeros(len(starts), bool)))
        pick = range(len(starts)) if lanes is None else [int(i) for i in lanes]
        for i in pick:
            s, yaw = starts[i], yaws[i]
            # 역방향 차선은 색을 나눈다. 같은 칸에서 정 · 역이 함께 출발하므로
            # 화살표만으로는 겹쳐 보인다.
            col = "#b03090" if bool(rev[i]) else "#1050c0"
            ax.arrow(s[0], s[1], 0.55 * np.cos(yaw), 0.55 * np.sin(yaw),
                     width=0.07, head_width=0.28, length_includes_head=True,
                     color=col, zorder=5)
            if not text_on:
                continue
            # 라벨은 랜드 이름 위에 겹치므로 흰 바탕을 깔고 타일 위쪽으로 올린다.
            ax.text(s[0], s[1] + 0.62, f"{i}: {tiles[i]}t", fontsize=6.5,
                    ha="center", color=col, zorder=7,
                    bbox=dict(boxstyle="round,pad=0.12", facecolor="white",
                              edgecolor=col, linewidth=0.5, alpha=0.9))

    lo = maze.tile_center(r0, c0, mz.kind.shape) - maze.TILE / 2
    hi = maze.tile_center(r1 - 1, c1 - 1, mz.kind.shape) + maze.TILE / 2
    ax.set_xlim(lo[0] - 0.2, hi[0] + 0.2)
    ax.set_ylim(lo[1] - 0.2, hi[1] + 0.2)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)", fontsize=8)
    ax.set_ylabel("y (m)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_title(title or f"maze seed {mz.seed}   {ty} x {tx}   gate {mz.gate}",
                 fontsize=9)
    fig.tight_layout()

    from pathlib import Path
    if filename is None:
        from .. import paths
        filename = paths.outputs("지도") / f"maze_{mz.seed}.png"
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filename, dpi=dpi)
    plt.close(fig)
    print(f"  지도 {filename.name}")
    return filename


# ---------- 사면 위 능력 (0단계) ----------

def _slope_rollout(env, policy_fn, command, nsteps, seed, heading=0.0,
                   start_xy=(0.0, 0.0), z_offset=0.0):
    """고정 명령으로 사면을 걷는다. 자세까지 기록한다.

    `_rollout` 과 나눈 이유 -- 저쪽은 판정선을 넘으면 끊는데, 회전을 재려면
    끝까지 돌려야 한다. 그리고 요각이 필요해서 `qpos[0:7]` 을 통째로 남긴다.
    """
    reset, step = jax.jit(env.reset_at), jax.jit(env.step)
    with_command, infer = jax.jit(env.with_command), jax.jit(policy_fn)

    key = jax.random.PRNGKey(int(seed))
    key, sub = jax.random.split(key)
    state = with_command(reset(sub, xy=tuple(float(v) for v in start_xy),
                               yaw=float(heading), z_offset=float(z_offset)),
                         jnp.asarray(command, jnp.float32))

    track = np.empty((nsteps, 7), dtype=np.float32)
    fell, i = -1, 0
    for i in range(nsteps):
        key, sub = jax.random.split(key)
        action, _ = infer(state.obs, sub)
        state = step(state, action)
        state = with_command(state, jnp.asarray(command, jnp.float32))
        track[i] = np.asarray(state.data.qpos[0:7])
        if bool(state.done) and i > SETTLE:
            fell = i
            break
    return track[:i + 1], fell


def _yaw_of(quat: np.ndarray) -> np.ndarray:
    """(N, 4) wxyz -> 요각 (rad). 이어붙여서 한 바퀴 넘게 돈 것도 센다."""
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    return np.unwrap(np.arctan2(2 * (w * z + x * y),
                                1 - 2 * (y * y + z * z)))




def obstacle_test(kinds, speeds, *, width: int = 3, nsteps: int = 400,
                  seed: int = 0, checkpoint=None, verbose: bool = True):
    """장애물 한 칸을 **LLC 단독으로** 정면 통과한다. 한계인지 아닌지만 본다.

    `ramp_test` 와 같은 틀이다 -- 미로와 같은 생성기(`lands.obstacle_corridor`)가
    지형을 만들고, 상위 제어기 없이 고정 명령으로 밀어 넣는다. 차이가 나면
    그것은 그 지형 자체의 차이다.

    **왜 필요한가** -- 턱은 `STEP_HEIGHT = 0.06` 이 실측 한계라는 근거로 미로에서
    뺐다. 그런데 `ROCK_HEIGHT` 는 0.14 로 그 2.3 배인데 같은 실측을 거친 적이
    없다. 미로 표에서 돌 칸이 늘수록 도달이 0.966 -> 0.697 로 무너지고 실패가
    넘어짐이라, 이것이 LLC 한계인지 HLC 미학습인지 갈라야 한다.

    통과 판정은 안 한다. **얼마나 갔고 넘어졌는가**만 낸다 -- 판정선을 두면
    복도 길이가 답을 정해 버린다.
    """
    ckpt = checkpoint
    if ckpt is None:
        from .. import paths
        ckpt = paths.llc()

    rows = []
    t0 = time.perf_counter()
    for kind in kinds:
        k = maze.NAMES_EN.get(kind, str(kind)) if isinstance(kind, int) else kind
        code = kind if isinstance(kind, int) else getattr(maze, str(kind))
        height, _, plan = lands.obstacle_corridor(
            code, level_after=0, axis=maze.RUN_X, width=int(width))
        env = hlc_env.make(terrain=height)
        policy_fn = loader.load_policy(ckpt, loader.env_observation_size(env))
        for v in speeds:
            command = list(spec.BASE_VECTOR)
            command[spec.index("vx")] = float(v)
            track, fell = _slope_rollout(env, policy_fn, command, nsteps, seed)
            row = {"랜드": maze.NAMES.get(code, str(code)), "속도": float(v),
                   "스텝": int(len(track)), "넘어짐": int(fell),
                   "전진": round(float(track[-1, 0] - track[0, 0]), 2),
                   "옆으로": round(float(track[-1, 1] - track[0, 1]), 2),
                   "최저z": round(float(track[:, 2].min()), 3)}
            rows.append(row)
            if verbose:
                mark = "넘어짐" if fell >= 0 else "  버팀"
                print(f"  {row['랜드']}  vx {v:.1f}  {mark}  "
                      f"전진 {row['전진']:+6.2f} m  옆으로 {row['옆으로']:+6.2f} m  "
                      f"최저z {row['최저z']:.3f}  ({row['스텝']}스텝)", flush=True)
        del env
    if verbose:
        print(f"  ({time.perf_counter() - t0:.0f}초)")
    return rows


def obstacle_table(rows) -> str:
    """`obstacle_test` 의 행들을 표로. 행이 랜드, 열이 속도다."""
    speeds = sorted({r["속도"] for r in rows})
    kinds = list(dict.fromkeys(r["랜드"] for r in rows))
    out = ["", "장애물 한 칸  (LLC 단독, 정면 통과. x 전진 m)", "",
           "  랜드   " + "  ".join(f"{v:>7.1f}" for v in speeds)]
    for k in kinds:
        cells = []
        for v in speeds:
            r = next((r for r in rows if r["랜드"] == k and r["속도"] == v), None)
            cells.append("      X" if r is None or r["넘어짐"] >= 0
                         else f"{r['전진']:>+7.2f}")
        out.append(f"  {k:<5}  " + "  ".join(cells))
    out += ["", "  X 는 넘어짐."]
    return "\n".join(out)

def ramp_test(modes, speeds, *, width: int = 7, nsteps: int = 400, seed: int = 0,
              yaw_cmd: float = 0.0, checkpoint=None, verbose: bool = True):
    """경사 한 칸을 **오르는 것과 가로지르는 것**을 나란히 잰다.

    지형을 `lands.obstacle_corridor(RAMP, level_after=1)` 로 만든다. 즉 미로와
    **같은 생성기가 만든 같은 경사 한 칸**이다. 지형은 하나로 두고 로봇의 출발
    위치와 방향만 바꾸므로, 차이가 나면 그것은 횡단이라는 것 자체의 차이다.

    ```
        y                                   경사 열은 x 로 기울어 있다
        ^  ....  ....  ////  ####  ####     (단 0 -> 1)
        |  ....  ....  ////  ####  ####
        |  ..@>  ....  ////  ####  ####     등반 -- 원점에서 +x
        |  ....  ....  //^/  ####  ####     횡단 -- 경사 칸 위에서 +y
        +---------------------------> x
    ```

    **왜 이렇게 만들어야 하는가** -- 앞서 연속 10 m 비탈로 만든 측정틀은
    무효였다. 대조군인 정면 등반이 그 틀에서 -0.64 m (뒤로 밀림) 였는데 미로에서는
    같은 20 도를 1.000 으로 오른다. 결과가 뒤집히면 그 틀의 다른 수치도 못 쓴다.
    차이는 각도가 아니라 **경사의 폭과 진입**이었다.

        미로     평지에서 조주 -> 2 m 한 칸에서 0.713 m -> 다시 평지
                 미끄러져도 1 m 안에 평지에 닿아 저절로 멈춘다
        옛 틀    10 m 연속 비탈. 그 한가운데 정지 상태로 시작. 멈출 데가 없다

    `modes` 는 `"등반"` · `"횡단"` 중 골라 넣는다. `width` 는 세로 랜드 수이고
    횡단 거리를 정한다 (7이면 10 m 를 걷는다).

    반환값 -- 행 목록. 표는 `ramp_table` 이 만든다.
    """
    ckpt = checkpoint
    if ckpt is None:
        from .. import paths
        ckpt = paths.llc()

    height, _, plan = lands.obstacle_corridor(
        maze.RAMP, level_after=1, axis=maze.RUN_X, width=int(width))
    env = hlc_env.make(terrain=height)
    policy_fn = loader.load_policy(ckpt, loader.env_observation_size(env))
    ex, ey = plan["extent"]
    ramp_x = float(plan["obstacle_x"])
    # 횡단은 경사 칸 위에서 아래쪽 끝에서 출발한다. 맵 가장자리는 낭떠러지라
    # 한 랜드를 남긴다.
    y0 = -ey / 2 + maze.TILE * 1.5

    rows = []
    t0 = time.perf_counter()
    def ground_at(x, y):
        """그 자리 지면 높이 (m). 경사 중턱에서 출발시키려면 필요하다."""
        h, w = height.shape
        j = int(np.clip((x + ex / 2) / ex * w, 0, w - 1))
        i = int(np.clip((y + ey / 2) / ey * h, 0, h - 1))
        return float(height[i, j]) * maze.SPAN - maze.DEPTH

    # **횡단은 좌우 두 쪽을 다 재야 한다.** 사면이 로봇의 왼쪽에 있느냐
    # 오른쪽에 있느냐는 지형이 아니라 로봇 쪽 사정이고, 대칭이면 같아야 한다.
    # 미로 실측에서 이 둘이 도달 0.031 대 0.767 로 갈렸기 때문에 대조가 필요하다.
    # 경사는 +x 로 오르므로 북(+y)을 보면 오르막이 오른쪽, 남(-y)을 보면 왼쪽이다.
    HEADING = {"등반": 0.0, "횡단·오른쪽오르막": np.pi / 2,
               "횡단·왼쪽오르막": -np.pi / 2}
    ALIAS = {"횡단": "횡단·오른쪽오르막"}
    modes = [ALIAS.get(m, m) for m in modes]

    for mode in modes:
        assert mode in HEADING, f"모르는 방식 {mode}. {tuple(HEADING)} 중에서 고르세요"
        xy = (0.0, 0.0) if mode == "등반" else (ramp_x, y0 if "오른쪽" in mode
                                               else -y0)
        yaw0 = HEADING[mode]
        # **몸통을 그 자리 지면만큼 띄운다.** keyframe 의 z 는 평지 기준이라
        # 경사 중턱에서 출발시키면 땅에 박힌 채로 시작한다.
        z0 = ground_at(*xy)
        for v in speeds:
            command = list(spec.BASE_VECTOR)
            command[spec.index("vx")] = float(v)
            command[spec.index("yaw")] = float(yaw_cmd)
            track, fell = _slope_rollout(env, policy_fn, command, nsteps, seed,
                                         heading=yaw0, start_xy=xy, z_offset=z0)
            # 나아가야 하는 축. 등반은 x, 횡단은 y 다.
            k = 0 if mode == "등반" else 1
            # 남쪽을 보고 횡단하면 전진이 -y 다. 부호를 맞춰야 두 쪽을 견준다.
            sgn = -1.0 if mode == "횡단·왼쪽오르막" else 1.0
            row = {
                "방식": mode,
                "속도": float(v),
                "스텝": int(len(track)),
                "넘어짐": int(fell),
                "전진": round(sgn * float(track[-1, k] - track[0, k]), 2),
                # 옆으로 밀린 양. 횡단이면 이것이 사면 아래로 흘러내린 거리다.
                "옆으로": round(float(track[-1, 1 - k] - track[0, 1 - k]), 2),
                "고도": round(float(track[:, 2].max() - track[0, 2]), 2),
                "요각변화": round(float(np.degrees(
                    _yaw_of(track[:, 3:7])[-1] - _yaw_of(track[:, 3:7])[0])), 1),
            }
            rows.append(row)
            if verbose:
                mark = "넘어짐" if fell >= 0 else "  버팀"
                print(f"  {mode}  vx {v:.1f}  {mark}  전진 {row['전진']:+6.2f} m  "
                      f"옆으로 {row['옆으로']:+6.2f} m  고도 {row['고도']:+5.2f} m  "
                      f"요각 {row['요각변화']:+7.1f}도  ({row['스텝']}스텝)")
    del env
    if verbose:
        print(f"  ({time.perf_counter() - t0:.0f}초)")
    return rows


def ramp_table(rows) -> str:
    """`ramp_test` 의 행들을 표로. **등반과 횡단을 나란히 놓는 것이 요점이다.**"""
    speeds = sorted({r["속도"] for r in rows})
    modes = sorted({r["방식"] for r in rows}, reverse=True)
    out = ["", "경사 한 칸  (행 = 방식, 열 = vx 명령. 진행축 전진 m)", "",
           "  방식   " + "  ".join(f"{v:>7.1f}" for v in speeds)]
    for m in modes:
        cells = []
        for v in speeds:
            r = next((r for r in rows if r["방식"] == m and r["속도"] == v), None)
            cells.append("      X" if r is None or r["넘어짐"] >= 0
                         else f"{r['전진']:>+7.2f}")
        out.append(f"  {m}  " + "  ".join(cells))
    out += ["", "  X 는 넘어짐. 등반은 x, 횡단은 y 방향 전진."]
    return "\n".join(out)


def weights_from(rows, *, floor: float = 0.5, ceiling: float = 3.0):
    """`lane_report` 행들 -> 차선 추출 가중치. **바닥과 천장을 둔다.**

    실패율을 그대로 쓰지 않는다. 정말 안 되는 차선 하나가 배치를 독점하면
    실패율 100 % 인데 배울 것이 0 인 상황에 갇힌다 (ACCEL 논문이 지적하는
    "줄일 수 없는 후회"). 그리고 되던 차선을 빼면 잊는다 -- 이 판에서 세 번
    겪었다.

        가중치 = clip(1 - 도달, floor 대응, ceiling 대응)

    `floor` 는 균등 대비 최소 배율이고 `ceiling` 은 최대 배율이다. 기본값이면
    도달 1.000 인 차선도 균등의 절반은 돌고, 0.000 인 차선이 3배를 넘지 않는다.

    반환값 -- 차선 번호 순서의 배열. `stage1.Task.set_lane_weight` 에 넣는다.
    """
    n = max(int(r["차선"]) for r in rows) + 1
    w = np.full(n, float(floor), dtype=np.float64)
    for r in rows:
        # 도달 1.0 -> floor, 도달 0.0 -> ceiling. 그 사이는 선형.
        w[int(r["차선"])] = floor + (ceiling - floor) * (1.0 - float(r["도달"]))
    return w / w.sum()


# ---------- 턱 한계를 축 조합으로 다시 잰다 ----------

def step_sweep(heights, grid, *, seeds: int = 16, nsteps=None, checkpoint=None,
               verbose: bool = True):
    """턱 높이 x **명령 조합** 격자를 통과율로 잰다. LLC 는 안 건드린다.

    기존 `step_limit` 과 다른 점이 둘이다.

        축      저쪽은 `vx` 하나만 흔들고 나머지를 기준값에 고정했다. 그런데
                학습된 축이 여섯이다. 자세를 조절해서 넘는 길이 있는지 안 봤다
        판정    저쪽은 씨앗 하나짜리 이진 판정이다. 여기서는 씨앗을 여럿 돌려
                **통과율**을 낸다 -- 한 판짜리 "통과"는 신뢰 구간이 없다

    이 프로젝트에서 "고정 명령으로 못 한다"를 "정책이 못 한다"로 읽어 세 번
    틀렸다. 턱 한계 0.06 도 그 자리에 있다 -- 실패 유형이 "발이 면에 걸린다"라
    발 궤적 문제이고, `pitch` 와 `height` 가 직접 건드리는 부분이다.

    `grid` 는 축 이름 -> 값 목록이다. 데카르트 곱을 전부 돈다.

        {"vx": [0.4, 0.6, 0.8], "pitch": [-0.3, 0.0, 0.3], "height": [0.22, 0.32]}

    **미학습 축을 넣어도 된다.** 응답이 없으면 표가 평평하게 나올 뿐이고,
    그것 자체가 그 축의 상태를 확인해 준다.

    반환값 -- 행 목록. 표는 `step_table` 이 만든다.
    """
    import itertools

    ckpt = checkpoint
    if ckpt is None:
        from .. import paths
        ckpt = paths.llc()

    names = list(grid)
    combos = [dict(zip(names, v)) for v in itertools.product(*grid.values())]
    if verbose:
        print(f"턱 {len(heights)}개 x 조합 {len(combos)}개 x 씨앗 {seeds}회 "
              f"= {len(heights) * len(combos) * seeds}판", flush=True)

    rows = []
    t0 = time.perf_counter()
    for h in heights:
        terrain, plan = lands.step_corridor(float(h))
        env = hlc_env.make(terrain=terrain)
        policy_fn = loader.load_policy(ckpt, loader.env_observation_size(env))
        goal_x = float(plan["goal_x"])
        lane_y = float(plan["lane_y"])
        ns = int(nsteps or steps_for(goal_x, max(grid.get("vx", [0.6]))))

        cmds = []
        for c in combos:
            v = list(spec.BASE_VECTOR)
            for k, val in c.items():
                v[spec.index(k)] = float(val)
            cmds.append(v)
        cmds = jnp.asarray(cmds, jnp.float32)                 # (C, DIM)
        keys = jax.random.split(jax.random.PRNGKey(0), int(seeds))

        reset, step = env.reset_at, env.step
        with_command = env.with_command

        def one(cmd, key):
            state = with_command(reset(key, xy=(0.0, 0.0), yaw=0.0), cmd)

            def body(carry, _):
                st, k = carry
                k, sub = jax.random.split(k)
                act, _ = policy_fn(st.obs, sub)
                nxt = with_command(step(st, act), cmd)
                return (nxt, k), (nxt.data.qpos[0], nxt.data.qpos[1], nxt.done)

            _, (xs, ys, dones) = jax.lax.scan(body, (state, key), None,
                                              length=ns)
            # 낙하 · 정착 구간의 done 은 지형 탓이 아니다.
            dones = dones.at[:SETTLE].set(0.0)
            alive = jnp.concatenate([jnp.ones(1),
                                     jnp.cumprod(1.0 - dones)[:-1]])
            crossed = (xs > goal_x) & (alive > 0.5)
            in_lane = jnp.max(jnp.abs(ys) * alive) < lane_y
            fell = jnp.max(dones) > 0.5
            # 넘어지기 **전에** 판정선을 지났는가.
            first_cross = jnp.argmax(crossed)
            reached = jnp.any(crossed)
            fell_first = fell & (~reached)
            return (reached & in_lane & (~fell_first)).astype(jnp.float32)

        run = jax.jit(jax.vmap(jax.vmap(one, in_axes=(None, 0)),
                               in_axes=(0, None)))
        ok = np.asarray(run(cmds, keys))                      # (C, seeds)
        for c, rate in zip(combos, ok.mean(axis=1)):
            rows.append({"높이": round(float(h), 3),
                         "면각도": round(lands.face_degrees(h), 1),
                         **{k: float(v) for k, v in c.items()},
                         "통과율": float(rate)})
        if verbose:
            best = max(ok.mean(axis=1))
            arg = combos[int(np.argmax(ok.mean(axis=1)))]
            print(f"  h={h:.2f} ({lands.face_degrees(h):.0f}도)  "
                  f"최고 통과율 {best:.3f}  {arg}", flush=True)
        del env
    if verbose:
        print(f"  ({time.perf_counter() - t0:.0f}초)")
    return rows


def step_table(rows, axis=None) -> str:
    """`step_sweep` 의 행들을 표로. `axis` 를 주면 그 축을 열로 편다."""
    hs = sorted({r["높이"] for r in rows})
    out = ["", "턱 통과율  (행 = 높이)", ""]
    if axis is None:
        out.append("  높이   면각도   최고 통과율   그 조합")
        for h in hs:
            same = [r for r in rows if r["높이"] == h]
            best = max(same, key=lambda r: r["통과율"])
            c = {k: v for k, v in best.items()
                 if k not in ("높이", "면각도", "통과율")}
            out.append(f"  {h:.2f}  {best['면각도']:5.1f}도    "
                       f"{best['통과율']:.3f}     {c}")
    else:
        vals = sorted({r[axis] for r in rows})
        out.append(f"  높이   " + "  ".join(f"{axis}={v:+.2f}" for v in vals))
        for h in hs:
            cells = []
            for v in vals:
                same = [r for r in rows if r["높이"] == h and r[axis] == v]
                cells.append(f"{max(r['통과율'] for r in same):.3f}"
                             if same else "  -  ")
            out.append(f"  {h:.2f}   " + "  ".join(f"{c:>9}" for c in cells))
    return "\n".join(out)
