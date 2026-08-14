"""무작위 미로를 눈으로 보는 것. 산출물은 `outputs/`에 떨어진다.

    python tools/render_maze.py                씨앗 2, 4 x 10, 관문 0.5
    python tools/render_maze.py 11             씨앗 11
    python tools/render_maze.py 11 6 16        세로 6 x 가로 16
    python tools/render_maze.py 11 6 16 0.0    관문 0 (우회로가 많다)

여기에는 **판단도 계산도 없다.** 미로는 `maze`가 만들고 씬은 `env`가 굽는다.
이 파일은 카메라를 놓고 그림을 저장할 뿐이다.

렌더는 mjx가 아니다. mjx 렌더는 warp 백엔드에서만 되고 이 환경은 impl="jax"다.
물리는 mjx가 돌지만 그림은 CPU가 그린다. 둘이 같은 `hfield_data`를 본다는 것은
스크립트가 마지막에 찍어 확인한다.
"""
import os
import sys

import numpy as np
import mujoco
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from go1_nav.hlc import env as hlc_env, maze     # noqa: E402

OUT = os.path.join(ROOT, "outputs")
NAMES = {maze.FLAT: "평지", maze.RAMP: "경사", maze.STEP: "턱  ", maze.GAP: "도랑",
         maze.ROCK: "돌  ", maze.WALL: "벽  ", maze.ROUGH: "거침",
         maze.BRIDGE: "다리", maze.TUNNEL: "터널", maze.PIT: "절벽"}
assert set(NAMES) == set(maze.IMPLEMENTED), (
    f"이름 없는 랜드 종류: {set(maze.IMPLEMENTED) - set(NAMES)}")

args = sys.argv[1:]
if len(args) not in (0, 1, 3, 4):
    raise SystemExit(
        f"인자를 {len(args)}개 주셨습니다. 0 · 1 · 3 · 4개만 됩니다.\n"
        f"  render_maze.py                씨앗 2, 4 x 10, 관문 0.5\n"
        f"  render_maze.py 11             씨앗 11\n"
        f"  render_maze.py 11 6 16        세로 6 x 가로 16\n"
        f"  render_maze.py 11 6 16 0.0    관문 0 (우회로가 많다)"
    )
seed = int(args[0]) if args else 2
ty, tx = (int(args[1]), int(args[2])) if len(args) >= 3 else (4, 10)
gate = float(args[3]) if len(args) == 4 else 0.5

mz = maze.generate(seed, shape=(ty, tx), gate=gate)
ok = maze.reachable(mz)
rt = maze.route(mz)          # 통과 규칙으로 다시 찾은 최단 경로. 3D의 파란 띠

print("=" * 74)
print(f"씨앗 {seed}   관문 {gate}   랜드 {ty} x {tx}   맵 {mz.extent[0]:.0f} x {mz.extent[1]:.0f} m   "
      f"격자 {mz.height.shape}   천장 {len(mz.ceiling)}")
print("아래가 y 작은 쪽. 숫자는 높이 단.")
for r in range(ty - 1, -1, -1):
    print("  " + " ".join(f"{NAMES[int(mz.kind[r, c])]}{int(mz.level[r, c])}"
                          f"{'ㅁ' if rt[r, c] else ('*' if ok[r, c] else ' ')}"
                          for c in range(tx)))
print(f"  * = 걸어서 닿는 랜드 {int(ok.sum())}/{mz.kind.size},  ㅁ = 최단 경로 {int(rt.sum())}칸,  생성기가 판 자국 {int(mz.path.sum())}칸")
m_h = mz.height * maze.SPAN - maze.DEPTH
print(f"높이 {m_h.min():+.3f} ~ {m_h.max():+.3f} m   "
      f"출발 {mz.start}   목표 {mz.goal}")
print("=" * 74)


env = hlc_env.make(terrain=mz.height, ceiling=mz.ceiling,
                   texture=maze.texture(rt))
M = env.mj_model
print("mjx hfield_data == mj:", np.array_equal(
    np.asarray(env.mjx_model.hfield_data).reshape(-1), M.hfield_data.reshape(-1)))

# 씬은 로봇 한 마리 크기로 맞춰져 있다. `statistic extent="0.8"`에 zfar 배수
# 50이 곱해져 **40 m 밖이 잘린다.** 맵이 크면 위에서 내려다볼 때 카메라가 그
# 밖으로 나가 화면이 통째로 검게 나온다. extent를 맵 크기로 올려 같이 늘린다.
M.stat.extent = float(max(mz.extent))

# 조명과 재질도 로봇 크기 기준이라 맵 전체가 어둡게 나온다.
M.vis.headlight.ambient[:] = [0.55, 0.55, 0.55]
M.vis.headlight.diffuse[:] = [0.75, 0.75, 0.75]
M.vis.headlight.specular[:] = [0.1, 0.1, 0.1]
# 바닥 재질은 그대로 둔다. 거기에 랜드 색 텍스처가 깔려 있다.


def shot(name, dist, elev, azim, lookat, robot, w=1400, h=900):
    d = mujoco.MjData(M)
    d.qpos[:] = M.keyframe("home").qpos
    d.qpos[0], d.qpos[1], d.qpos[2] = robot
    mujoco.mj_forward(M, d)
    r = mujoco.Renderer(M, height=h, width=w)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance, cam.elevation, cam.azimuth = dist, elev, azim
    cam.lookat[:] = lookat
    r.update_scene(d, camera=cam)
    p = os.path.join(OUT, f"{name}.png")
    Image.fromarray(r.render()).save(p)
    r.close()
    print("저장", p)


start = (float(mz.start[0]), float(mz.start[1]), 0.42)
span = max(mz.extent)
shot(f"maze_s{seed}_top", span * 1.15, -80, 90, (0, 0, 0), start)
shot(f"maze_s{seed}_wide", span * 1.1, -26, 100, (0, 0, 0.2), start)
shot(f"maze_s{seed}_eye", 5.0, -10, 100, (mz.start[0] + 2, mz.start[1], 0.3), start)

maze.save(mz, os.path.join(OUT, f"maze_s{seed}.npz"))
print("저장", os.path.join(OUT, f"maze_s{seed}.npz"))
