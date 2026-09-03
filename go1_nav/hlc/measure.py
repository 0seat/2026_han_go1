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
    "headlight_ambient": 0.45,   # 격자를 넣은 뒤 낮췄다. 0.60 이면 바닥이 날아간다
    "headlight_diffuse": 0.75,
    "headlight_specular": 0.05,
    "light_diffuse": 0.70,
    "light_ambient": 0.35,       # 원래 0.0 이라 그늘이 새까맸다
    "light_specular": 0.05,
    "ground_reflect": 0.0,       # 원래 0.8. 어둡고 번들거렸다
    # **`mat_rgba` 는 무늬가 있으면 안 먹는다** (실측: 값을 바꿔도 픽셀이 동일).
    # 그래서 색은 무늬 픽셀에 직접 넣는다. 아래 `checker` 가 그것이다.
    "ground_rgba": (1.0, 1.0, 1.0, 1.0),
    # 원래 텍스처가 평균 (97, 70, 49) 인 어두운 갈색이라 바닥이 탁했다. 떼면
    # 백지가 되어 굴곡을 읽을 단서가 사라진다. **무늬는 남기고 밝게 다시 그린다.**
    # 밝은 두 톤을 푸르게. 회색 무늬에 회색 로봇이면 서로 안 읽힌다.
    # 대비가 크면 굴곡보다 무늬가 먼저 읽히므로 두 톤을 가깝게 둔다.
    # 바닥을 **흰 바탕에 1 m 격자**로 다시 그린다. 체커는 굴곡을 읽히게 해 주지만
    # 눈금이 없어서 "얼마나 갔나 · 얼마나 벌어졌나" 를 못 읽는다. 격자는 그 둘을
    # 같이 준다 -- 밝아서 지형이 보이고, 칸이 1 m 라 거리가 바로 읽힌다.
    #
    # 1 m 를 맞추는 법 -- 텍스처 한 번 반복이 1 m 가 되도록 `mat_texrepeat` 를
    # hfield 의 실제 크기에서 계산한다 (`brighten` 참고). 상수로 박으면 지형
    # 크기가 바뀔 때마다 눈금이 거짓말을 한다.
    "grid_base": (236, 239, 244),      # 바탕. 완전한 흰색은 형태가 날아간다
    "grid_line": (112, 126, 148),      # 격자선. 로봇(회색)과 구별되게 푸르게
    # 50 cm. 1 m 로 뒀더니 칸이 커서 굴곡이 안 읽혔다 -- 격자선 사이가 비어 있으면
    # 그 안의 높낮이를 눈이 못 잡는다. 반으로 줄이면 선이 지형을 따라 휘는 것이
    # 보인다.
    "grid_metres": 0.5,                # 격자 한 칸 (m)
    "grid_px": 64,                     # 텍스처 한 칸의 픽셀
    "grid_line_px": 2,                 # 선 두께. 64 px 에 2 px = 3 cm
    # **맵 밖의 배경.** 이 씬에는 스카이박스가 없어서 지형 너머가 새까맣다.
    # 발표 화면에서 대비가 너무 세고, 지형의 밝은 부분이 오려낸 종이처럼 뜬다.
    # 안개 색을 옅은 하늘색으로 두면 배경이 그 색으로 채워진다.
    # 안개로는 안 된다 -- 실측으로 `mjRND_FOG` 를 켜도 좌상단 픽셀이 (0,0,0)
    # 그대로다. 안개는 **지오메트리에만** 걸리고 빈 곳은 검정으로 지운다.
    # 스카이박스를 넣으려면 텍스처 슬롯이 더 필요해 모델을 다시 컴파일해야 한다.
    # 그래서 렌더가 끝난 뒤 **정확히 검은 픽셀만** 이 색으로 바꾼다. 지형에서
    # 제일 어두운 것이 절벽 `#1b1b1b` = (27,27,27) 이고 로봇은 회색이라, 순수
    # (0,0,0) 은 배경뿐이다. 그림자도 꺼져 있다.
    "background": (209, 219, 232),
}


def brighten(model, keep_texture: bool = False):
    """렌더용 밝은 사본. **물리에 영향이 없다.** 위 주석 참고.

    `keep_texture` -- 바닥 그림을 **그대로 둔다.** `paint_map` 으로 구운 지도를
    `hlc_env.make(texture=)` 로 이미 깔아 둔 경우다. 이걸 안 주면 아래 코드가
    그 위에 흰 격자를 다시 칠하고 `texuniform` 까지 되돌려서 구운 지도가 통째로
    사라진다 -- 실제로 한 번 그렇게 나왔다. 조명만 손본다.
    """
    import copy
    import mujoco

    m = copy.deepcopy(model)
    # 배경색. 빈 곳을 안개 색이 채운다.
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
        if mat < 0:
            continue
        m.mat_reflectance[mat] = BRIGHT["ground_reflect"]
        m.mat_rgba[mat] = BRIGHT["ground_rgba"]
        if keep_texture:
            continue
        # **`texrepeat` 은 지형 전체의 반복 수가 아니라 미터당 반복 수다.**
        # 이 재질은 `texuniform = 1` 이라 월드 단위로 매핑된다. 지형 폭(448 m)을
        # 넣었더니 미터당 448 번이 되어 화면이 모아레로 덮였다. 한 칸을 1 m 로
        # 두려면 미터당 1 번이다.
        rep = 1.0 / float(BRIGHT["grid_metres"])
        m.mat_texuniform[mat] = 1
        m.mat_texrepeat[mat] = (rep, rep)
        for tex in set(int(t) for t in m.mat_texid[mat]):
            if tex >= 0:
                _repaint_grid(m, tex)
    return m


def _repaint_grid(m, tex):
    """텍스처 하나를 **흰 바탕 + 격자선**으로 다시 그린다. 픽셀만 바꾼다.

    한 번 반복이 격자 한 칸이므로, 여기서는 **가장자리 두 변에만** 선을 긋는다.
    반복이 이어붙으면 그 선들이 격자가 된다. 안쪽에 선을 더 그으면 `texrepeat`
    가 뜻하는 칸 수와 눈에 보이는 칸 수가 어긋난다.
    """
    import numpy as _np

    w, h = int(m.tex_width[tex]), int(m.tex_height[tex])
    nch = int(m.tex_nchannel[tex]) if hasattr(m, "tex_nchannel") else 3
    adr = int(m.tex_adr[tex])
    base = _np.asarray(BRIGHT["grid_base"], _np.uint8)[:nch]
    line = _np.asarray(BRIGHT["grid_line"], _np.uint8)[:nch]
    # 선 두께를 텍스처 크기에 맞춰 준다. 64 px 기준으로 적어 둔 값을 비례로 옮긴다.
    t = max(1, int(round(BRIGHT["grid_line_px"] * min(w, h)
                         / max(1, BRIGHT["grid_px"]))))
    img = _np.broadcast_to(base, (h, w, nch)).copy()
    img[:t, :] = line
    img[:, :t] = line
    m.tex_data[adr:adr + w * h * nch] = img.reshape(-1)


#: 탑뷰 카메라. **모델에 없는 시점이라 그때그때 만든다.**
#:
#: `track` 은 뒤에서 따라가는 시점이라 "지금 무슨 지형을 밟는가" 는 잘 보이는데
#: "경로 어디쯤인가 · 어디로 꺾는가" 가 안 보인다. 위에서 보면 그 둘이 보인다.
#: 발표용으로는 둘을 같이 내는 편이 낫다.
#: 실측으로 고른 값 -- 수직(-89도)이면 벽과 언덕이 평면으로 뭉개져 높낮이가
#: 안 읽히고, -45도면 하늘이 화면의 3분의 1을 먹는다 (8x16 은 세로 16 m 라
#: 6 m 위에서도 지도 밖이 보인다).
TOPDOWN = {"거리": 7.0, "고도": -72.0, "방위": 90.0, "높이보정": 0.3}

#: 경로 표시. **로봇이 받는 길잡이가 어디를 가리키는지** 보이게 한다.
#:
#: hfield 는 칸마다 색을 못 준다 -- 재질이 하나이고 텍스처가 월드 좌표로 반복된다.
#: 그래서 지형을 칠하는 대신 **렌더 장면에 도형을 얹는다.** 물리는 이미 끝난
#: 뒤이고 도형은 `mjvScene` 에만 들어가므로 시뮬레이션에 영향이 없다.
#:
#: **칸마다 판을 까는 방식은 버렸다.** 타일 중심에 1.7 m 판을 놓으면 판끼리
#: 떨어져 보이고, 지형이 판 안에서 기울면 한쪽이 땅에 박히고 반대쪽이 뜬다.
#: 대신 경로를 잘게 다시 뽑아 **이어진 선**으로 긋는다. 촘촘하면 선이 지형을
#: 따라 휘므로 박히지도 뜨지도 않는다.
ROUTE_LINE = {"간격": 0.25, "굵기": 0.055, "띄움": 0.035,
              "색": (0.30, 0.58, 0.95, 0.85)}


def _ground_z(model, hf, x, y):
    """hfield 위 그 자리의 지면 높이 (m). 선을 지형에 붙여 놓으려면 필요하다."""
    import numpy as _np

    rx, ry, elev = (float(model.hfield_size[hf][i]) for i in range(3))
    nrow, ncol = int(model.hfield_nrow[hf]), int(model.hfield_ncol[hf])
    adr = int(model.hfield_adr[hf])
    data = model.hfield_data[adr:adr + nrow * ncol].reshape(nrow, ncol)
    # hfield 는 geom 중심을 기준으로 (-rx, rx) x (-ry, ry) 를 덮는다.
    j = int(_np.clip((x + rx) / (2 * rx) * (ncol - 1), 0, ncol - 1))
    i = int(_np.clip((y + ry) / (2 * ry) * (nrow - 1), 0, nrow - 1))
    return float(data[i, j]) * elev


def _resample_line(route, step):
    """경로를 `step` 간격으로 다시 뽑는다. 꺾임에서도 간격이 유지된다."""
    import numpy as _np

    pts = _np.asarray(route, _np.float64).reshape(-1, 2)
    if len(pts) < 2:
        return pts
    seg = _np.linalg.norm(_np.diff(pts, axis=0), axis=1)
    along = _np.concatenate([[0.0], _np.cumsum(seg)])
    if along[-1] <= 0:
        return pts[:1]
    want = _np.arange(0.0, along[-1], float(step))
    return _np.stack([_np.interp(want, along, pts[:, 0]),
                      _np.interp(want, along, pts[:, 1])], axis=1)


def _draw_route(model, scene, route):
    """정답 경로를 **이어진 선**으로 긋는다. 장면에만 넣는다."""
    import mujoco
    import numpy as _np

    hg = next((g for g in range(model.ngeom)
               if int(model.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_HFIELD)),
              None)
    if hg is None:
        return
    hf, gz = int(model.geom_dataid[hg]), float(model.geom_pos[hg][2])
    pts = _resample_line(route, ROUTE_LINE["간격"])
    if len(pts) < 2:
        return
    z = _np.array([gz + _ground_z(model, hf, x, y) + ROUTE_LINE["띄움"]
                   for x, y in pts])
    rgba = _np.array(ROUTE_LINE["색"], _np.float32)
    for i in range(len(pts) - 1):
        if scene.ngeom >= scene.maxgeom:
            break
        g = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE,
                            _np.zeros(3), _np.zeros(3), _np.eye(3).reshape(9),
                            rgba)
        mujoco.mjv_connector(
            g, int(mujoco.mjtGeom.mjGEOM_CAPSULE), ROUTE_LINE["굵기"],
            _np.array([pts[i][0], pts[i][1], z[i]]),
            _np.array([pts[i + 1][0], pts[i + 1][1], z[i + 1]]))
        scene.ngeom += 1


def _camera(model, kind, data=None):
    """`kind` -> `update_scene` 에 넣을 카메라. 문자열이면 모델의 카메라를 쓴다."""
    import mujoco

    if kind != "탑뷰":
        return kind if kind is not None else -1
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance = float(TOPDOWN["거리"])
    cam.elevation = float(TOPDOWN["고도"])
    cam.azimuth = float(TOPDOWN["방위"])
    if data is not None:
        cam.lookat[:] = (float(data.qpos[0]), float(data.qpos[1]),
                         float(data.qpos[2]) + TOPDOWN["높이보정"])
    return cam


def render_frames(env, states, camera="track", height=480, width=640,
                  bright=True, route=None, keep_texture=False):
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

    m = brighten(env.mj_model, keep_texture) if bright else env.mj_model
    bg = np.asarray(BRIGHT["background"], np.uint8) if bright else None
    # 선은 잘게 나누므로 점 수보다 훨씬 많다. 경로 길이 / 간격 만큼 잡는다.
    n_extra = 0
    if route is not None:
        pts = np.asarray(route).reshape(-1, 2)
        length = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
        n_extra = int(length / ROUTE_LINE["간격"]) + 8
    renderer = mujoco.Renderer(m, height=height, width=width,
                               max_geom=10000 + n_extra)
    d = mujoco.MjData(m)
    out = []
    try:
        for state in _progress(states):
            d.qpos[:] = np.asarray(state.data.qpos)
            d.qvel[:] = np.asarray(state.data.qvel)
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)      # track 카메라가 subtree_com 을 본다
            mujoco.mj_camlight(m, d)
            # **탑뷰는 프레임마다 다시 만든다.** 로봇을 따라가야 하기 때문이다.
            renderer.update_scene(d, camera=_camera(m, camera, d))
            if route is not None:
                _draw_route(m, renderer.scene, route)
            img = renderer.render()
            if bg is not None:
                img = img.copy()
                img[np.all(img == 0, axis=-1)] = bg   # 배경만. 위 주석 참고
            out.append(img)
    finally:
        renderer.close()
    return out


def save_video(env, states, filename, fps=50, stride=1, camera="track",
               height=480, width=640, bright=True, route=None,
               keep_texture=False):
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
                             height=height, width=width, bright=bright,
                             route=route, keep_texture=keep_texture)
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


#: 바닥 그림의 격자선. `plot_map` 의 색과 같은 계열로 맞춘다.
PAINT_GRID = (150, 160, 178)


#: 3D 바닥에 쓸 때 `KIND_COLOR` 를 덮어쓰는 색.
#:
#: `plot_map` 의 색은 **종이 위 지도**용이라 평지가 거의 흰색(#e8e8e8)이다.
#: 그 값을 3D 바닥에 그대로 깔면 조명을 받아 날아가고, 그 위에서 다른 색이
#: 안 읽힌다. 평지만 눌러서 나머지가 살게 한다.
PAINT_OVERRIDE = {maze.FLAT: "#c9cfd6"}


def paint_map(mz, plan=None, *, px_per_tile: int = 16, grid_m: float = 1.0):
    """미로를 **바닥 텍스처 한 장**으로 굽는다. `hlc_env.make(texture=)` 에 넣는다.

    발표용이다. 지금 렌더는 지형 전체가 단색이라 경사도 돌도 다리도 구분이 안
    된다. 랜드마다 색을 주면 영상과 `plot_map` 의 지도가 **같은 색**이 되어,
    보는 사람이 둘을 이어서 읽을 수 있다.

    **물리에는 영향이 없다.** 텍스처는 시각 자산이고 mjx 모델에 안 실린다
    (`brighten` 머리말과 같은 이유). 그래서 이 그림을 켠 판과 안 켠 판의
    측정값이 같아야 한다.

    `px_per_tile` -- 랜드 하나가 몇 픽셀인가. 10 x 224 면 16 픽셀에 160 x 3584 다.
    키우면 격자선이 매끈해지고 파일이 커진다.
    """
    from PIL import Image, ImageDraw

    kind = np.asarray(mz.kind)
    ty, tx = kind.shape
    n = int(px_per_tile)
    img = Image.new("RGB", (tx * n, ty * n), PAINT_OVERRIDE[maze.FLAT])
    px = img.load()
    # 랜드 색. `plot_map` 과 같은 표를 쓴다 -- 두 그림의 색이 어긋나면 안 된다.
    rgb = {k: tuple(int(v.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
           for k, v in {**KIND_COLOR, **PAINT_OVERRIDE}.items()}
    for r in range(ty):
        for c in range(tx):
            col = rgb.get(int(kind[r, c]), (200, 200, 200))
            for i in range(r * n, (r + 1) * n):
                for j in range(c * n, (c + 1) * n):
                    px[j, i] = col

    d = ImageDraw.Draw(img)
    # **경로는 여기 안 그린다.** `_draw_route` 가 3D 선으로 그린다 -- 지형을
    # 따라 휘고 굵기가 원근을 타서 바닥에 인쇄한 것보다 훨씬 잘 읽힌다.
    # 1 m 격자. 거리를 읽는 눈금이라 리본 위에도 얹는다.
    step = grid_m / maze.TILE * n
    for j in range(int(tx * n / step) + 1):
        x = j * step
        d.line([(x, 0), (x, ty * n)], fill=PAINT_GRID, width=1)
    for i in range(int(ty * n / step) + 1):
        y = i * step
        d.line([(0, y), (tx * n, y)], fill=PAINT_GRID, width=1)
    return np.asarray(img, dtype=np.uint8)

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


def plot_scan(filename=None, *, dpi=160):
    """관측 스캔 배치 도식. **보고서 그림 전용이다.**

    측정이 아니라 설명을 위한 그림이라 실제 지형을 쓰지 않는다. 대신 상수는
    전부 `obs` 에서 읽어 온다 -- 손으로 적으면 상수가 바뀌었을 때 그림만 틀린
    채로 남는다.

    두 칸으로 그린다. 위는 평면 배치, 아래는 옆에서 본 단면이다.

    **아래 칸이 이 그림의 요점이다.** 두 스캔이 같은 격자를 쓰면서 기준이
    다르다는 것을 말로 쓰면 길고, 단면 한 컷이면 바로 보인다.

        지형 스캔   발밑 지면이 기준     "땅이 솟았나"
        천장 스캔   몸통 원점이 기준     "머리가 닿나"

    기준을 같이 두면 안 되는 이유 -- 몸통 높이는 HLC 가 직접 명령하는 값이다.
    몸통 기준으로 지형을 재면 "내가 몸을 낮춘 것" 과 "땅이 솟은 것" 이 같은
    숫자가 된다 (`obs.terrain_scan` 주석).

    라벨은 **영어로만** 쓴다. 그래프에 한글을 넣으면 글꼴이 없는 자리에서
    두부가 되고, 보고서는 어느 기계에서 열릴지 모른다.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, FancyArrowPatch

    from . import obs as _obs

    off = np.asarray(_obs.scan_offsets())
    fx = np.unique(off[:, 0])
    fy = np.unique(off[:, 1])

    fig, (ax, bx) = plt.subplots(2, 1, figsize=(9.0, 7.4),
                                 gridspec_kw={"height_ratios": [1.25, 1.0]})

    # ---- 위 칸. 평면 배치 ----
    ax.add_patch(Rectangle((-0.35, -0.16), 0.70, 0.32, facecolor="0.80",
                           edgecolor="0.35", zorder=1))
    # 진행 방향 화살표는 **격자 밖에** 둔다. 격자 안에 그리면 점을 덮는다
    ax.annotate("", xy=(0.55, fy[-1] + 0.20), xytext=(0.05, fy[-1] + 0.20),
                zorder=4, arrowprops=dict(arrowstyle="-|>", lw=1.6, color="0.20"))
    ax.text(0.60, fy[-1] + 0.20, "heading", ha="left", va="center",
            fontsize=9, color="0.20")
    ax.scatter(off[:, 0], off[:, 1], s=13, color="#1f77b4", zorder=3,
               label=f"scan points  {_obs.SCAN_NX} x {_obs.SCAN_NY} = {_obs.SCAN_SIZE}")
    # 랜드 한 칸 경계. 앞으로 2.0 m 를 보는 근거가 이것이다.
    ax.axvline(maze.TILE, color="#d62728", ls="--", lw=1.2, zorder=2)
    ax.text(maze.TILE - 0.04, _obs.SCAN_LAT[1] + 0.10,
            f"next tile edge  {maze.TILE:.1f} m", ha="right", fontsize=8.5,
            color="#d62728")
    ax.annotate("", xy=(fx[0], -0.86), xytext=(0.0, -0.86),
                arrowprops=dict(arrowstyle="<->", lw=1.0, color="0.35"))
    ax.text(fx[0] / 2, -0.96, f"{abs(fx[0]):.1f} m behind", ha="center",
            fontsize=8.5, color="0.35")
    ax.annotate("", xy=(fx[0], fy[0]), xytext=(fx[0], fy[-1]),
                arrowprops=dict(arrowstyle="<->", lw=1.0, color="0.35"))
    ax.text(fx[0] - 0.10, 0.0, f"{fy[-1] - fy[0]:.1f} m", rotation=90,
            va="center", ha="right", fontsize=8.5, color="0.35")
    ax.annotate("", xy=(fx[0], fy[-1] + 0.14), xytext=(fx[1], fy[-1] + 0.14),
                arrowprops=dict(arrowstyle="<->", lw=1.0, color="0.35"))
    ax.text((fx[0] + fx[1]) / 2, fy[-1] + 0.20,
            f"{_obs.SCAN_STEP:.1f} m", ha="center", fontsize=8.5, color="0.35")
    ax.set_xlim(fx[0] - 0.45, fx[-1] + 0.30)
    ax.set_ylim(-1.10, fy[-1] + 0.46)
    ax.set_aspect("equal")
    ax.set_xlabel("forward (m, robot frame)")
    ax.set_ylabel("lateral (m)")
    ax.set_title("Scan grid in robot frame  (terrain and ceiling share it)",
                 fontsize=10.5)
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.9)
    ax.grid(alpha=0.25, lw=0.5)

    # ---- 아래 칸. 옆에서 본 단면 ----
    # 설명용 지형. 앞쪽에 단이 하나 오르고 그 위에 터널이 얹힌다.
    gx = np.linspace(fx[0], fx[-1], 400)
    ground = np.where(gx < 0.9, 0.0, 0.18)
    body_z = 0.30                       # 몸통 원점 높이 (m). 그림용 대표값
    tun_lo, tun_hi = 1.15, 1.95         # 터널이 덮는 구간
    ceil_z = 0.18 + maze.TUNNEL_CLEAR   # 터널 바닥면 = 그 자리 지면 + 여유고

    bx.fill_between(gx, -0.12, ground, color="0.86", edgecolor="0.45", lw=1.0)
    bx.add_patch(Rectangle((tun_lo, ceil_z), tun_hi - tun_lo, 0.10,
                           facecolor="#c9b7dd", edgecolor="0.45", lw=1.0))
    bx.text((tun_lo + tun_hi) / 2, ceil_z + 0.15, "tunnel ceiling",
            ha="center", fontsize=8.5, color="0.30")

    # 몸통과 두 기준선
    bx.add_patch(Rectangle((-0.35, body_z - 0.06), 0.70, 0.12,
                           facecolor="0.80", edgecolor="0.35", zorder=3))
    bx.axhline(0.0, color="#2ca02c", ls="--", lw=1.1)
    bx.text(fx[0] - 0.02, 0.052, "ground under robot  (terrain reference)",
            fontsize=8.5, color="#2ca02c")
    bx.axhline(body_z, color="#d62728", ls="--", lw=1.1)
    bx.text(fx[-1] + 0.25, body_z + 0.015, "body origin  (ceiling reference)",
            fontsize=8.5, color="#d62728", ha="right")

    # 스캔 표본. 중앙 열(lateral 0)만 그린다
    gs = np.interp(fx, gx, ground)
    bx.scatter(fx, gs, s=22, color="#1f77b4", zorder=4)
    inside = (fx >= tun_lo) & (fx <= tun_hi)
    bx.scatter(fx[inside], np.full(inside.sum(), ceil_z), s=22,
               marker="v", color="#9467bd", zorder=4)

    # 무엇을 재는가. 화살표 두 개
    i = int(np.argmin(np.abs(fx - 1.4)))
    bx.add_patch(FancyArrowPatch((fx[i], 0.0), (fx[i], gs[i]),
                                 arrowstyle="<->", mutation_scale=11,
                                 color="#1f77b4", lw=1.3, zorder=5))
    bx.text(fx[i] + 0.06, gs[i] / 2, "terrain scan\nheight above ground",
            fontsize=8.5, color="#1f77b4", va="center")
    bx.add_patch(FancyArrowPatch((fx[i] - 0.42, body_z),
                                 (fx[i] - 0.42, ceil_z),
                                 arrowstyle="<->", mutation_scale=11,
                                 color="#9467bd", lw=1.3, zorder=5))
    bx.text(fx[i] - 0.48, (body_z + ceil_z) / 2,
            "ceiling scan\nclearance above body", fontsize=8.5,
            color="#9467bd", va="center", ha="right")

    bx.set_xlim(fx[0] - 0.45, fx[-1] + 0.30)
    bx.set_ylim(-0.12, 0.86)
    bx.set_xlabel("forward (m, robot frame)")
    bx.set_ylabel("height (m)")
    bx.set_title("Side view  -  same grid, different reference", fontsize=10.5)
    bx.grid(alpha=0.25, lw=0.5)

    fig.suptitle(
        f"Terrain scan {_obs.SCAN_SIZE} + ceiling scan {_obs.CEIL_SIZE}"
        f"   (clipped to +-{_obs.SCAN_CLIP:.1f} m)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    if filename is None:
        from .. import paths
        filename = paths.outputs("그림") / "스캔배치.png"
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filename, dpi=dpi)
    plt.close(fig)
    print(f"  그림 {filename.name}")
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


#: `ramp_test` 와 `ramp_video` 가 함께 쓰는 출발 방향. 경사는 +x 로 오르므로
#: 북(+y)을 보면 오르막이 오른쪽, 남(-y)을 보면 왼쪽이다.
RAMP_HEADING = {"등반": 0.0, "횡단·오른쪽오르막": np.pi / 2,
                "횡단·왼쪽오르막": -np.pi / 2}


def ramp_video(out_dir=None, *, modes=("횡단·왼쪽오르막", "횡단·오른쪽오르막"),
               speed: float = 0.6, deg=None, lands_width: int = 7,
               nsteps: int = 400, seed: int = 0, stride: int = 2,
               px=(640, 480), checkpoint=None):
    """`ramp_test` 와 **같은 지형·같은 명령**으로 굴리고 영상을 남긴다.

    표(`ramp_test`)가 낸 숫자를 눈으로 확인하는 그림이다. 측정과 영상이 다른
    틀에서 나오면 둘을 나란히 놓을 수 없으므로 지형 생성과 출발 조건을 그대로
    맞춘다 -- 다르게 두면 "영상은 넘어지는데 표는 안 넘어진다" 가 생긴다.

    `deg` 를 주면 그 각도로 **경사만** 갈아끼운다. `maze.ELEVATION` 은 고정
    상수라 따라오지 않고, 따라서 `SPAN` 이 그대로다 -- 관측 서명이 안 바뀐다
    (`maze.ELEVATION` 주석).

    **`deg` 는 이 프로세스의 `maze.HIGH` 를 바꾼다.** 끝나면 되돌리므로 이어지는
    호출에는 영향이 없지만, 같은 프로세스에서 다른 측정을 병행하지 말 것.

    반환값 -- 저장한 경로 목록.
    """
    import math

    ckpt = checkpoint
    if ckpt is None:
        from .. import paths
        ckpt = paths.llc()
    if out_dir is None:
        from .. import paths
        out_dir = paths.outputs("그림")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    keep = maze.HIGH
    try:
        if deg is not None:
            maze.HIGH = ((maze.CELLS_PER_TILE - 1) * maze.CELL
                         * math.tan(math.radians(float(deg))))
            assert maze.ELEVATION >= maze.LEVEL_MAX * maze.HIGH + maze.WALL_HEIGHT
            print(f"  경사 {deg}도  단 높이 {maze.HIGH:.4f} m", flush=True)

        height, _, plan = lands.obstacle_corridor(
            maze.RAMP, level_after=1, axis=maze.RUN_X, width=int(lands_width))
        env = hlc_env.make(terrain=height)
        policy_fn = loader.load_policy(ckpt, loader.env_observation_size(env))
        ex, ey = plan["extent"]
        ramp_x = float(plan["obstacle_x"])
        y0 = -ey / 2 + maze.TILE * 1.5

        def ground_at(x, y):
            h, w = height.shape
            j = int(np.clip((x + ex / 2) / ex * w, 0, w - 1))
            i = int(np.clip((y + ey / 2) / ey * h, 0, h - 1))
            return float(height[i, j]) * maze.SPAN - maze.DEPTH

        reset, step = jax.jit(env.reset_at), jax.jit(env.step)
        with_command, infer = jax.jit(env.with_command), jax.jit(policy_fn)

        saved = []
        for mode in modes:
            assert mode in RAMP_HEADING, (
                f"모르는 방식 {mode}. {tuple(RAMP_HEADING)} 중에서 고르세요")
            xy = (0.0, 0.0) if mode == "등반" else (
                ramp_x, y0 if "오른쪽" in mode else -y0)
            command = list(spec.BASE_VECTOR)
            command[spec.index("vx")] = float(speed)
            cmd = jnp.asarray(command, jnp.float32)

            key = jax.random.PRNGKey(int(seed))
            key, sub = jax.random.split(key)
            state = with_command(
                reset(sub, xy=tuple(float(v) for v in xy),
                      yaw=float(RAMP_HEADING[mode]), z_offset=ground_at(*xy)),
                cmd)

            states, fell = [state], -1
            for i in range(nsteps):
                key, sub = jax.random.split(key)
                action, _ = infer(state.obs, sub)
                state = with_command(step(state, action), cmd)
                states.append(state)
                if bool(state.done) and i > SETTLE:
                    fell = i
                    break

            tag = mode.replace("·", "_")
            mark = "넘어짐" if fell >= 0 else "버팀"
            angle = f"{deg:g}도" if deg is not None else "기본"
            name = f"경사{angle}_{tag}_vx{speed:g}_{mark}.mp4"
            print(f"  {mode}  vx {speed}  {mark}  {len(states)}스텝", flush=True)
            saved.append(save_video(env, states, out_dir / name,
                                    fps=50, stride=stride, camera="track",
                                    height=int(px[1]), width=int(px[0])))
        del env
        return saved
    finally:
        maze.HIGH = keep


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
