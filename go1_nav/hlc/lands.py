"""시험용 지형 -- **미로가 아니라 자를 만든다.**

미로는 씨앗에서 나오고 상수가 박혀 있다(`STEP_HEIGHT` 같은 것). 그런데 그 상수를
정하려면 상수를 흔들어 봐야 한다. 닭과 달걀이라, 흔들 수 있는 지형을 따로 둔다.

**모양은 `maze.py`와 같아야 한다.** 여기서 만든 턱이 미로의 턱과 다르면 재서 얻은
숫자를 미로에 되먹일 수 없다. 그래서 눈금(`CELL`, `normalize`)과 띠 폭 계산
(`_band`)을 전부 `maze.py`에서 가져다 쓴다. 여기서 새로 정하는 것은 **높이 하나**다.

미로의 턱과 다른 점이 하나 있다. 미로는 `+` 십자라 옆으로 못 돌아가게 하고,
여기는 **폭 전체를 가로지르는 띠**다. 재는 것이 "넘을 수 있나"이므로 마주치는
것을 강제해야 한다. 십자로 두면 로봇이 팔 사이로 빠져나가 통과로 기록된다.
"""

from __future__ import annotations

import numpy as np

from . import maze

#: 여기서 출발시키지 않는 랜드. 통로가 좁아 스폰 지터가 통로 밖으로 나간다.
#: `maze_segments` 가 쓴다 -- 자세한 근거는 그 함수 안 주석에 있다.
NARROW_START = (maze.BRIDGE, maze.TUNNEL, maze.GAP, maze.PIT)


def step_corridor(step_height: float, *, step_ahead: int = 2, after: int = 1,
                  width: int = 3, top_span: float | None = None):
    """턱 하나짜리 직선 복도. **로봇이 서는 자리가 원점이 되게** 만든다.

    ```
        y
        ^   +-----+-----+-----+--@--+-----+=====+-----+
        |   |     |     |     |출발 |     | 턱  |착지 |
        +-> +-----+-----+-----+-----+-----+=====+-----+   -> x
             0     1     2     3     4     5     6
                              x=0
    ```

    `env.reset`은 씬의 home 자세로 로봇을 놓고, 그 자리는 **월드 원점**이다.
    지형을 어디에 그리든 로봇은 원점에서 시작한다. 그래서 시작 랜드의 중심이
    원점에 오도록 랜드 수를 잡는다 -- 로봇을 옮기려면 `qpos`를 고치고 `forward`를
    다시 돌려야 하는데, 그러면 접촉과 센서를 손으로 맞추게 되고 조용히 틀린다.

    중심이 원점에 오려면 랜드 수가 **홀수**여야 한다. 앞쪽에 필요한 칸수를
    정하면 뒤쪽은 대칭으로 따라오고, 뒤쪽 칸은 안 쓰지만 해가 없다.

    step_ahead -- 턱이 출발 랜드에서 몇 칸 앞인가. 2면 조주 거리가
                  `2 * TILE - STEP_SPAN/2 = 3.7 m`다. 명령 속도에 도달하고도 남는다.
    after      -- 턱 뒤에 몇 칸을 둘 것인가. 넘자마자 넘어지는 것을 통과로
                  세지 않기 위해 최소 1칸은 있어야 한다.

    반환값 -- `(height, plan)`. `height`는 `env.make(terrain=...)`에 그대로 넣는
    [0, 1] 격자다.

    폭을 3칸 두는 이유 -- hfield 밖은 바닥 geom(-DEPTH)이라 낭떠러지다. 요각이
    조금만 틀어져도 옆으로 떨어져서, 턱이 아니라 조향을 재게 된다.
    """
    ahead = int(step_ahead) + int(after)
    n_lands = 2 * ahead + 1                  # 홀수. 가운데가 원점
    start_land = ahead
    step_land = start_land + int(step_ahead)

    n = maze.CELLS_PER_TILE
    metres = np.zeros((width * n, n_lands * n), dtype=np.float32)

    span = maze.STEP_SPAN if top_span is None else float(top_span)
    lo, hi = maze._band(n, span)                 # 상판 길이. 미로와 같은 계산
    x0 = step_land * n
    metres[:, x0 + lo:x0 + hi] += float(step_height)

    ex, ey = n_lands * maze.TILE, width * maze.TILE
    def centre(col, row):
        return np.asarray([(col + 0.5) * maze.TILE - ex / 2,
                           (row + 0.5) * maze.TILE - ey / 2], dtype=np.float32)

    start = centre(start_land, width // 2)
    assert abs(float(start[0])) < 1e-6 and abs(float(start[1])) < 1e-6,         f"시작 랜드 중심이 원점이 아닙니다: {start}"

    plan = {
        "start_xy": start,
        # 판정선. 턱 상판을 지나 반 랜드를 더 간 자리다. 넘자마자 무너지는 것을
        # 통과로 세지 않으면서, 맵 끝(x = ex/2)에서는 충분히 안쪽이다.
        "goal_x": (x0 + hi) * maze.CELL - ex / 2 + maze.TILE / 2,
        # 셀 중심이 아니라 **띠의 경계**다. 중심으로 재면 폭이 CELL 하나만큼 짧다.
        "step_x0": (x0 + lo) * maze.CELL - ex / 2,
        "step_x1": (x0 + hi) * maze.CELL - ex / 2,
        "step_height": float(step_height),
        "extent": (ex, ey),
        "shape": (width, n_lands),
        # 옆으로 떨어졌는지 보는 선. 가운데 랜드를 벗어나면 실패로 친다.
        "lane_y": maze.TILE / 2,
    }
    return maze.normalize(metres), plan


def face_degrees(step_height: float) -> float:
    """턱 면의 각도. 지금은 한 셀 안에서 올라가므로 높이가 정하면 따라온다."""
    return float(np.degrees(np.arctan2(step_height, maze.CELL)))

def obstacle_corridor(kind: int, *, level_after: int = 0, axis: int | None = None,
                      ahead: int = 1, after: int = 1, width: int = 3, seed: int = 0):
    """장애물 랜드 하나짜리 직선 복도. **`maze.heightfield`가 만든다.**

    `step_corridor`와 다른 점 -- 저쪽은 턱 높이를 흔들려고 격자를 직접 그린다.
    이쪽은 미로의 상수를 그대로 쓰므로 **생성기를 그대로 부른다.** 그래서 여기서
    학습한 지형과 미로의 지형이 같다는 것이 구조적으로 보장된다.

    ```
        +-----+-----+=====+-----+-----+
        |     |출발 |장애물|도착 |     |     ahead=1, after=1 -> 랜드 5개
        +-----+-----+=====+-----+-----+
                   x=0
    ```

    장애물은 **열 전체**(모든 행)에 놓는다. 가운데 행에만 두면 로봇이 옆으로
    돌아가고, 그러면 "이 장애물을 통과했다"가 아니라 "피했다"를 학습한다.
    돌처럼 랜드 안에서 비켜갈 수 있는 것은 그대로 비켜갈 수 있다 -- 그건
    미로에서도 마찬가지라 맞다.

    level_after -- 장애물 뒤쪽 랜드의 높이 단. `maze.RAMP`를 쓸 때 1로 두면
                   진짜 오르막이 된다. 0이면 `_ramp`가 양끝 높이가 같다고 보고
                   아무것도 안 그린다.

    반환값 -- `(height, ceiling, plan)`. `ceiling`은 터널이 아니면 길이 0이다.
    """
    n_lands = 2 * (int(ahead) + int(after)) + 1        # 홀수. 가운데가 원점
    start_land = int(ahead) + int(after)
    obs_land = start_land + int(ahead)

    kinds = np.full((width, n_lands), maze.FLAT, dtype=np.int8)
    kinds[:, obs_land] = int(kind)
    levels = np.zeros((width, n_lands), dtype=np.int8)
    levels[:, obs_land + 1:] = int(level_after)
    axes = np.full((width, n_lands), maze.RUN_X if axis is None else int(axis),
                   dtype=np.int8)

    height = maze.heightfield(kinds, levels, axes, seed)
    ceiling = maze.ceilings(kinds, levels, axes)

    ex, ey = n_lands * maze.TILE, width * maze.TILE
    def centre(col, row):
        return np.asarray([(col + 0.5) * maze.TILE - ex / 2,
                           (row + 0.5) * maze.TILE - ey / 2], dtype=np.float32)

    start = centre(start_land, width // 2)
    assert abs(float(start[0])) < 1e-6 and abs(float(start[1])) < 1e-6,         f"시작 랜드 중심이 원점이 아닙니다: {start}"

    goal = centre(obs_land + int(after), width // 2)
    plan = {
        "kind": int(kind),
        "start_xy": start,
        "goal_xy": goal,
        # 차선 배열. 여기는 하나뿐이지만 `mixed_corridor` 와 **모양을 맞춘다.**
        # 그래야 `stage1.Task` 가 갈래 없이 한 갈래로 읽는다.
        "lane_start_xy": start[None, :].copy(),
        "lane_goal_xy": goal[None, :].copy(),
        "lane_kind": np.asarray([int(kind)], dtype=np.int64),
        "lane_level": np.asarray([int(level_after)], dtype=np.int64),
        # 길잡이에 넣을 경로. 직선 복도는 목표 한 점이라 `[로봇, 목표]` 가 되어
        # 예전과 값이 같다. 꺾인 복도만 점이 여럿이다.
        "lane_route": goal[None, None, :].copy(),
        "obstacle_x": (obs_land + 0.5) * maze.TILE - ex / 2,
        "level_after": int(level_after),
        "axis": maze.RUN_X if axis is None else int(axis),
        "extent": (ex, ey),
        "shape": (width, n_lands),
        "level_grid": levels.astype(np.float32),
        # 시작 랜드 안에서 자세를 흔들 범위. 진입 조건 무작위화가 이걸 쓴다.
        "jitter_xy": maze.TILE / 2 - 0.35,     # 랜드 경계에서 0.35 m 안쪽까지
        "lane_y": ey / 2 - 0.3,                # 맵 밖으로 나가면 실패
    }
    return height, ceiling, plan


def mixed_corridor(kinds, *, level_after=0, ahead: int = 1,
                   after: int = 1, seed: int = 0,
                   separator: int = maze.WALL):
    """장애물이 **차선마다 다른** 직선 복도. 망각을 막으려고 만들었다.

    ```
        +-----+-----+=====+-----+-----+
        |     |출발 | 터널|도착 |     |   <- 0번 차선
        +#####+#####+#####+#####+#####+   <- 벽. 차선을 가른다
        |     |출발 | 다리|도착 |     |   <- 1번 차선
        +#####+#####+#####+#####+#####+
        |     |출발 | 평지|도착 |     |   <- 2번 차선
        +-----+-----+=====+-----+-----+
    ```

    왜 필요한가
    -----------

    한 배치의 모든 환경이 같은 hfield 와 같은 천장 geom 목록을 쓴다. 천장 geom
    개수가 컴파일 시점에 굳으므로 **배치에 판을 섞을 수 없다.** 그래서 랜드를
    하나씩 이어 학습하면 마지막 판만 남는다. 실측 -- 다리를 학습한 파라미터로
    터널을 재니 도달 0.875 에서 **0.000** 이 되었고, 넘어짐 0.000 에 시간초과
    1.000 이라 입구 앞에 서 있었다. 숙이는 행동만 정확히 사라진 것이다.

    지형 하나에 차선을 나란히 두면 그 제약을 우회한다. 천장 geom 은 터널 차선이
    내는 것 하나뿐이라 개수가 고정이고, 섞이는 것은 **에피소드마다 어느 차선에서
    시작하느냐**다. 모델 무작위화가 전혀 필요 없다.

    왜 벽으로 가르는가
    ------------------

    안 가르면 옆 차선으로 새어 장애물을 **피해서** 지나간다. 다리가 특히 그렇다 --
    파인 곳은 제 랜드 안에서만이라, 옆 차선은 평지 높이 그대로다. 벽은 미로가 이미
    관문에 쓰는 것이라 새로 만드는 모양이 아니다.

    차선마다 단이 다를 수 있다
    --------------------------

    `level_after` 는 정수 하나이거나 차선 수만큼의 순서열이다. 경사만 단 1 이
    필요한데 전부에 걸면 터널 · 다리 · 돌 차선의 목표 랜드까지 0.713 m 올라가서
    **장애물 뒤가 절벽이 된다.** 높이 격자가 타일 단위라 행별로 다른 값을 넣으면
    된다.

    가름벽 행도 같이 올린다. 벽 높이는 `그 타일의 단 x HIGH + WALL_HEIGHT` 라,
    단 0 인 벽이 단 1 인 차선 옆에 서면 **벽이 8.7 cm 만 남아 넘어간다.** 그래서
    가름벽은 맞닿은 두 차선의 단 중 큰 쪽을 따른다.

    출발점이 원점이 아니다
    ----------------------

    `obstacle_corridor` 는 출발 랜드 중심이 원점이라고 확인하지만 여기는 차선마다
    y 가 다르므로 그럴 수 없다. `stage1.Task` 가 `lane_start_xy` 를 읽어 그 자리에
    세운다. 원점을 가정하는 코드가 있으면 거기서 어긋난다.

    반환값 -- `(height, ceiling, plan)`. `obstacle_corridor` 와 키가 같고
    `lane_*` 항목의 길이만 다르다.
    """
    kinds = [int(k) for k in kinds]
    assert len(kinds) >= 1, "차선이 하나는 있어야 합니다"
    n_lands = 2 * (int(ahead) + int(after)) + 1        # 홀수. 가운데가 원점 열
    start_land = int(ahead) + int(after)
    obs_land = start_land + int(ahead)
    n_rows = 2 * len(kinds) - 1                        # 차선 사이마다 벽 한 줄

    grid = np.full((n_rows, n_lands), maze.FLAT, dtype=np.int8)
    grid[1::2, :] = int(separator)                     # 홀수 행이 가름벽
    for lane, kind in enumerate(kinds):
        grid[2 * lane, obs_land] = kind

    if np.isscalar(after_turn):
        after = [bool(after_turn)] * n
    else:
        after = [bool(v) for v in after_turn]
        assert len(after) == n, f"장애물 위치는 구간 수({n})만큼 주세요"
    if np.isscalar(level_after):
        lane_levels = [int(level_after)] * len(kinds)
    else:
        lane_levels = [int(v) for v in level_after]
        assert len(lane_levels) == len(kinds), (
            f"단은 차선 수({len(kinds)})만큼 주거나 하나만 주세요")

    levels = np.zeros((n_rows, n_lands), dtype=np.int8)
    for lane, lv in enumerate(lane_levels):
        levels[2 * lane, obs_land + 1:] = lv
    # 가름벽은 맞닿은 두 차선 중 높은 쪽을 따른다. 안 그러면 벽을 넘어간다.
    for sep in range(1, n_rows, 2):
        levels[sep] = np.maximum(levels[sep - 1], levels[sep + 1])
    axes = np.full((n_rows, n_lands), maze.RUN_X, dtype=np.int8)

    height = maze.heightfield(grid, levels, axes, seed)
    ceiling = maze.ceilings(grid, levels, axes)

    ex, ey = n_lands * maze.TILE, n_rows * maze.TILE
    def centre(col, row):
        return np.asarray([(col + 0.5) * maze.TILE - ex / 2,
                           (row + 0.5) * maze.TILE - ey / 2], dtype=np.float32)

    rows = [2 * lane for lane in range(len(kinds))]
    plan = {
        "kind": kinds[0],
        "start_xy": centre(start_land, rows[0]),
        "goal_xy": centre(obs_land + int(after), rows[0]),
        "lane_start_xy": np.stack([centre(start_land, r) for r in rows]),
        "lane_goal_xy": np.stack([centre(obs_land + int(after), r)
                                  for r in rows]),
        "lane_kind": np.asarray(kinds, dtype=np.int64),
        "lane_route": np.stack([centre(obs_land + int(after), r)
                                for r in rows])[:, None, :],
        "obstacle_x": (obs_land + 0.5) * maze.TILE - ex / 2,
        "level_after": lane_levels[0],
        "lane_level": np.asarray(lane_levels, dtype=np.int64),
        "axis": maze.RUN_X,
        "extent": (ex, ey),
        "shape": (n_rows, n_lands),
        "jitter_xy": maze.TILE / 2 - 0.35,
        "lane_y": ey / 2 - 0.3,
    }
    return height, ceiling, plan


def _raise_walls(grid, levels, separator: int) -> None:
    """벽 타일의 단을 **맞닿은 통로 중 높은 쪽**으로 올린다. 제자리에서 고친다.

    벽 높이는 `단 x HIGH + WALL_HEIGHT` 다. 단 0 인 벽이 단 1 인 통로 옆에 서면
    0.8 - 0.713 = 0.087 m 만 남아 로봇이 넘어간다. 우회로가 생기면 정책은
    장애물을 영영 안 배운다.
    """
    ty, tx = grid.shape
    wall = grid == int(separator)
    out = levels.copy()
    for r in range(ty):
        for c in range(tx):
            if not wall[r, c]:
                continue
            near = [levels[rr, cc]
                    for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))
                    if 0 <= rr < ty and 0 <= cc < tx and not wall[rr, cc]]
            if near:
                out[r, c] = max(int(levels[r, c]), int(max(near)))
    levels[:, :] = out


def bent_corridor(before, after_kind, *, turn: int = 1, run_in: int = 1,
                  run_out: int = 2, width: int = 3, level_before: int = 0,
                  level_after: int = 0, seed: int = 0, mirror: bool = False,
                  separator: int = maze.WALL):
    """**한 번 꺾이는** 복도. 길잡이가 처음으로 진짜 경유점을 받는 판이다.

    ```
        ##########################
        #                        #
        #        [after]  도착   #     <- 꺾은 뒤 (y 가 큰 쪽)
        #           ^            #
        #  출발 [before]         #     <- 꺾기 전
        #                        #
        ##########################
    ```

    왜 필요한가
    -----------

    지금까지 길잡이는 `[로봇, 목표]` 두 점이었다. 목표가 늘 직선 앞이라
    `LOOKAHEAD` 0.5 · 1.5 · 3.0 m 세 점이 **전부 같은 방향**을 가리켰다. 즉
    경유점 셋을 쓰라고 만들어 둔 자리가 한 번도 쓰인 적이 없다.

    미로는 꺾인다. 꺾임을 학습하려면 세 점이 서로 다른 방향을 가리키는 판이
    필요하고, 그것만 격리해서 싸게 확인하는 것이 이 복도다. 미로를 씨앗 고정으로
    학습하는 길은 피한다 -- 지형을 외운 정책은 다음 미로에서 아무것도 못 한다.

    모양
    ----

    `before` 를 지나 오른쪽으로 가다가 `turn` 칸 위로 꺾고, `after_kind` 를 지나
    다시 오른쪽으로 가서 도착한다. 정답 경로 밖은 전부 `separator`(벽)다 -- 미로
    생성기가 관문에 쓰는 것과 같은 방식이고, 우회로를 남기면 정책은 **평지만 밟는
    로얄로드**를 배운다.

    장애물은 꺾기 전후에 하나씩 둔다. 꺾임만 있는 판은 이미 평지 학습으로 되므로
    새로 배울 것이 없다.

    `mirror` 는 꺾는 쪽을 뒤집는다. **한쪽으로만 학습하면 "왼쪽으로 돈다" 를
    외운다.** 두 판을 번갈아 돌리거나, 한쪽으로 학습한 뒤 반대쪽으로 재서 실제로
    경유점을 읽는지 확인한다.

    반환값 -- `(height, ceiling, plan)`. 키는 다른 복도와 같고 `lane_route` 만
    점이 여럿이다.
    """
    tx = int(run_in) + 1 + int(run_out) + 1        # 출발 + 장애물 + ... + 도착
    ty = int(turn) + 1
    obs_c = int(run_in)                            # 꺾기 전 장애물 열
    turn_c = obs_c + 1                             # 여기서 위로 꺾는다
    after_c = turn_c                               # 꺾은 뒤 장애물은 같은 열 위
    r0, r1 = 0, ty - 1

    grid = np.full((ty, tx), int(separator), dtype=np.int8)
    levels = np.zeros((ty, tx), dtype=np.int8)
    axes = np.full((ty, tx), maze.RUN_X, dtype=np.int8)

    route_tiles = [(r0, c) for c in range(0, turn_c + 1)]
    route_tiles += [(r, turn_c) for r in range(r0 + 1, r1 + 1)]
    route_tiles += [(r1, c) for c in range(turn_c + 1, tx)]
    for r, c in route_tiles:
        grid[r, c] = maze.FLAT
    grid[r0, obs_c] = int(before)
    grid[r1, after_c] = int(after_kind)
    # 세로 구간은 y 로 지나간다. 다리 · 터널이 여기 놓이면 통로 방향이 맞아야 한다.
    for r in range(r0, r1 + 1):
        axes[r, turn_c] = maze.RUN_Y
    # 단이 바뀌는 자리는 꺾기 전 장애물 타일이다. `bent_lanes` 와 같은 이유다.
    levels[:, :] = int(level_before)
    for r, c in route_tiles:
        if c > obs_c:
            levels[r, c] = int(level_after)
    _raise_walls(grid, levels, separator)

    if mirror:                       # 행을 뒤집으면 꺾는 쪽이 반대가 된다
        grid = grid[::-1].copy()
        levels = levels[::-1].copy()
        axes = axes[::-1].copy()
        r0, r1 = ty - 1 - r0, ty - 1 - r1

    height = maze.heightfield(grid, levels, axes, seed)
    ceiling = maze.ceilings(grid, levels, axes)

    ex, ey = tx * maze.TILE, ty * maze.TILE
    def centre(col, row):
        return np.asarray([(col + 0.5) * maze.TILE - ex / 2,
                           (row + 0.5) * maze.TILE - ey / 2], dtype=np.float32)

    start = centre(0, r0)
    goal = centre(tx - 1, r1)
    # 출발점과 꺾이는 칸. 직선 구간의 중간 점은 `resample` 이 보간하므로 넣어도
    # 값이 같고 개수만 는다. 출발점은 빼면 안 된다 -- `route_progress` 가 경로
    # 시작부터 재므로, 빼면 출발 구간의 전진 보상이 0 이 된다.
    route = np.stack([start, centre(turn_c, r0), centre(turn_c, r1), goal])

    plan = {
        "kind": int(before),
        "start_xy": start,
        "goal_xy": goal,
        "lane_start_xy": start[None, :].copy(),
        "lane_goal_xy": goal[None, :].copy(),
        "lane_kind": np.asarray([int(before)], dtype=np.int64),
        "lane_level": np.asarray([int(level_after)], dtype=np.int64),
        "lane_route": route[None, :, :].copy(),
        "obstacle_x": (obs_c + 0.5) * maze.TILE - ex / 2,
        "lane_after_turn": np.asarray(after, dtype=bool),
        "level_after": int(level_after),
        "axis": maze.RUN_X,
        "extent": (ex, ey),
        "shape": (ty, tx),
        "level_grid": levels.astype(np.float32),
        "jitter_xy": maze.TILE / 2 - 0.35,
        "lane_y": ey / 2 - 0.3,
    }
    return height, ceiling, plan


def bent_lanes(kinds, *, mirrors=None, level_after=0, run_in: int = 2,
               run_out: int = 2, after_turn=False, gap_after=None,
               seed: int = 0, separator: int = maze.WALL):
    """**차선마다 다른 장애물, 차선마다 다른 회전 방향.**

    ```
        ##########################
        #  출발 [터널]           #
        #              +-> 도착  #   <- 0번 차선. 위로 꺾는다
        ##########################   <- 벽
        #              +-> 도착  #
        #  출발 [다리]           #   <- 1번 차선. 아래로 꺾는다
        ##########################
    ```

    `mixed_corridor` 와 `bent_corridor` 를 합친 것이다. 두 문제를 한 번에 푼다.

        회전 방향   한쪽으로만 학습하면 그쪽만 돈다. 실측 -- 왼쪽으로 꺾는 판을
                    1.000 까지 올린 정책이 거울판에서 0.000 이었다
        망각        꺾인 복도는 단일 판이라 다른 랜드를 잊는다. 실측 -- 네 차선
                    0.938 이 꺾임 학습 뒤 0.734 로 떨어졌다

    한 지형에 차선을 나란히 두면 배치 안에서 둘 다 섞인다. 천장 geom 개수도
    고정이라 컴파일이 안 깨진다.

    행 배치
    -------

    차선 하나가 **두 줄**을 쓴다 (꺾이니까). 차선 사이에는 한 줄을 비워 벽으로
    남긴다. 그래서 차선 L 개에 `3L - 1` 줄이다. 정답 경로가 아닌 칸은 전부 벽이라
    가름벽을 따로 세우지 않는다 -- 우회로를 남기면 정책은 평지만 밟는다.

    `run_in` 이 2 인 이유 -- 1 이면 장애물 바로 다음 칸이 모서리라 **넘는 것과
    도는 것을 동시에** 시킨다. 실측으로 돌이 직선 복도 0.953 에서 꺾인 차선
    0.281 로 떨어졌다. 2 면 사이에 평지 2 m 가 생겨 둘이 나뉜다.

    `after_turn` 은 장애물을 **꺾임 뒤**로 옮긴다. 차선별로 주면 한 지형에서 두
    배치가 섞인다 -- 지금 잘하는 것을 잊지 않으면서 새 상황을 배운다.

    ```
        False   출발 - 평지 - [장애물] - 꺾임 - 평지 - 도착
        True    출발 - 평지 - 꺾임 - 평지 - [장애물] - 도착
    ```

    왜 필요한가 -- S자 복도에서 다섯 판이 전부 같은 자리에서 다리 아래로
    떨어졌다. 꺾어 나온 직후 1.3 m 만에 진입해야 하는 배치였고, 지금 학습에는
    장애물이 꺾임 **앞**에만 있어 "돌고 나서 정렬해 진입" 을 겪은 적이 없다.

    `gap_after` 는 **꺾임과 그 뒤 장애물 사이 칸수**다. None 이면 `run_in` 을
    쓴다. 이것을 `run_in` 에서 떼어낸 이유 -- 2칸(4 m)으로 재보니 0스텝 평가가
    이미 0.977 이었다. 돌고 나서 정렬할 직선이 통째로 주어지면 정렬은 이미
    아는 일이라 배울 것이 없다. **재현하려던 실패는 꺾임 뒤 1.3 m** 였다
    (S자 복도, 다섯 판 전부 다리 아래로). 그 간격을 직접 잡으려고 연다.

    `mirrors` 는 차선별 회전 방향이다. None 이면 번갈아 준다. `level_after` 는
    정수 하나이거나 차선 수만큼의 순서열이다 (경사만 1 이 필요하다).

    반환값 -- `(height, ceiling, plan)`. 키는 다른 복도와 같다.
    """
    kinds = [int(k) for k in kinds]
    n = len(kinds)
    assert n >= 1, "차선이 하나는 있어야 합니다"
    if mirrors is None:
        mirrors = [bool(i % 2) for i in range(n)]
    mirrors = [bool(v) for v in mirrors]
    assert len(mirrors) == n, f"회전 방향은 차선 수({n})만큼 주세요"
    if np.isscalar(after_turn):
        after = [bool(after_turn)] * n
    else:
        after = [bool(v) for v in after_turn]
        assert len(after) == n, f"장애물 위치는 구간 수({n})만큼 주세요"
    if np.isscalar(level_after):
        levels_of = [int(level_after)] * n
    else:
        levels_of = [int(v) for v in level_after]
        assert len(levels_of) == n, f"단은 차선 수({n})만큼 주세요"

    # 꺾임 뒤에 두려면 꺾임과 장애물 사이에도 자리가 있어야 한다. 그 간격이
    # 1칸이면 넘는 것과 도는 것을 **동시에** 시킨다 -- 꺾임 앞에서는 그것이
    # 돌을 0.953 에서 0.281 로 무너뜨렸다. 뒤에서는 그것이 배우려는 것 자체다.
    gap = int(run_in) if gap_after is None else int(gap_after)
    assert gap >= 1, "꺾임 칸과 장애물 칸이 붙으면 모서리가 장애물이 된다"
    tail = gap if any(after) else 0
    tx = int(run_in) + 1 + tail + int(run_out) + 1
    obs_c, turn_c = int(run_in), int(run_in) + 1
    # 차선 하나가 두 줄, 차선 사이에 한 줄, 그리고 **위아래 테두리 한 줄씩.**
    # 테두리가 없으면 맨 바깥 차선이 맵 가장자리에 붙어 로봇이 걸어 나간다.
    # 실측으로 그랬다 -- 꺾인 복도 거울판에서 맵 밖으로 나가 넘어졌다.
    n_rows = 3 * n + 1
    base = 1                                # 0 번 줄은 테두리

    grid = np.full((n_rows, tx), int(separator), dtype=np.int8)
    levels = np.zeros((n_rows, tx), dtype=np.int8)
    axes = np.full((n_rows, tx), maze.RUN_X, dtype=np.int8)

    starts, goals, routes = [], [], []
    for lane in range(n):
        lo, hi = base + 3 * lane, base + 3 * lane + 1
        r_start, r_end = (hi, lo) if mirrors[lane] else (lo, hi)
        tiles = [(r_start, c) for c in range(0, turn_c + 1)]
        step = 1 if r_end > r_start else -1
        tiles += [(r, turn_c) for r in range(r_start + step, r_end + step, step)]
        tiles += [(r_end, c) for c in range(turn_c + 1, tx)]
        # **단이 바뀌는 자리는 장애물 타일 자신이다.** 뒤쪽에서 올리면 `_ramp` 는
        # 양 옆 이웃이 같은 단이라 아무것도 안 그리고, 대신 단이 실제로 바뀌는
        # 모서리에 0.713 m 절벽이 선다. 실측으로 그랬다 -- 로봇이 모서리에서
        # 멈췄다. 장애물 타일 자신의 단은 0 으로 둔다. `_ramp` 가 그 칸의 값을
        # 이웃 단 사이 보간으로 덮어쓴다.
        o_c_lane = (turn_c + gap) if after[lane] else obs_c
        for r, c in tiles:
            grid[r, c] = maze.FLAT
            if c > o_c_lane:
                levels[r, c] = levels_of[lane]
        # 꺾임 앞이면 출발 줄의 `obs_c`, 뒤면 도착 줄의 `turn_c + gap`.
        if after[lane]:
            o_r, o_c = r_end, turn_c + gap
        else:
            o_r, o_c = r_start, obs_c
        grid[o_r, o_c] = kinds[lane]
        for r in (r_start, r_end):
            axes[r, turn_c] = maze.RUN_Y
        starts.append((r_start, 0))
        goals.append((r_end, tx - 1))
        # **출발점을 경로에 넣는다.** 안 넣으면 경로가 첫 꼭짓점에서 시작해
        # 출발부터 거기까지 걷는 동안 `route_progress` 가 안 줄고, 그 구간의
        # 전진 보상이 0 이 된다. `route_polyline` 은 로봇을 앞에 붙이므로
        # 영향이 없었는데 보상 쪽에서 드러났다.
        routes.append(((r_start, 0), (r_start, turn_c),
                       (r_end, turn_c), (r_end, tx - 1)))

    # 벽은 `그 타일의 단 x HIGH + WALL_HEIGHT` 다. 단 0 인 벽이 단 1 인 통로
    # (0.713 m) 옆에 서면 **0.087 m 만 남아 넘어간다.** 벽을 이웃 통로의 최대
    # 단으로 올린다.
    _raise_walls(grid, levels, separator)

    height = maze.heightfield(grid, levels, axes, seed)
    ceiling = maze.ceilings(grid, levels, axes)

    ex, ey = tx * maze.TILE, n_rows * maze.TILE
    def centre(row, col):
        return np.asarray([(col + 0.5) * maze.TILE - ex / 2,
                           (row + 0.5) * maze.TILE - ey / 2], dtype=np.float32)

    return height, ceiling, {
        "kind": kinds[0],
        "start_xy": centre(*starts[0]),
        "goal_xy": centre(*goals[0]),
        "lane_start_xy": np.stack([centre(*s) for s in starts]),
        "lane_goal_xy": np.stack([centre(*g) for g in goals]),
        "lane_kind": np.asarray(kinds, dtype=np.int64),
        "lane_level": np.asarray(levels_of, dtype=np.int64),
        "lane_route": np.stack([np.stack([centre(*t) for t in r])
                                for r in routes]),
        "obstacle_x": (obs_c + 0.5) * maze.TILE - ex / 2,
        "lane_after_turn": np.asarray(after, dtype=bool),
        "level_after": levels_of[0],
        "axis": maze.RUN_X,
        "extent": (ex, ey),
        "shape": (n_rows, tx),
        "level_grid": levels.astype(np.float32),
        "jitter_xy": maze.TILE / 2 - 0.35,
        "lane_y": ey / 2 - 0.3,
    }


def snake_corridor(kinds, *, level_after=0, run: int = 3, after_turn=False,
                   seed: int = 0, mirror: bool = False,
                   separator: int = maze.WALL):
    """**여러 번 꺾이는** 복도. 꺾는 쪽이 번갈아 바뀐다.

    ```
        ############################
        #  출발 [터널]      [돌]   #   <- 0번 줄
        #             +-----+  +-> 도착
        #        [다리]           #   <- 1번 줄
        ############################
    ```

    왜 필요한가
    -----------

    지금까지 정책이 겪은 경로는 **ㄱ 자 한 번**이 전부다. 미로는 여러 번 꺾이고
    방향도 섞인다. 더 어려운 판을 학습시키기 전에, 지금 정책이 **학습 없이**
    얼마나 하는지부터 재야 한다.

        잘 나온다   경로를 읽는 규칙이 이미 일반적이다. 미로로 바로 간다
        무너진다    무엇이 부족한지 이 판이 알려준다

    거울판에서 0.922 가 나온 것이 이 시험을 할 근거다. 회전 방향이 전부 뒤집히고
    지형도 거울인데 유지됐으니, 정책이 지형이 아니라 경로를 읽고 있다.

    모양
    ----

    `kinds` 는 **구간마다 하나씩**이다. 구간이 n 개면 꺾임은 n-1 번이다. 구간은
    두 줄을 번갈아 쓴다 -- 0번 구간이 윗줄, 1번이 아랫줄, 2번이 다시 윗줄. 그래서
    꺾는 쪽이 위 · 아래로 교대한다. 한쪽으로만 꺾으면 계단이 되고, 그건 회전
    방향을 섞지 못한다.

    구간 하나가 `run` 칸이고 장애물이 하나 들어간다. 꺾는 칸과 겹치지 않게
    `run >= 2` 여야 한다.

    `after_turn` 이 구간마다 **장애물을 꺾임의 앞뒤 어디에 둘지** 정한다.
    구간별로 주면 한 판에서 두 배치가 섞인다.

    ```
        False   ... 꺾임 - 평지 - [장애물] - 꺾임 ...   넘고 나서 돈다
        True    ... 꺾임 - [장애물] - 평지 - 꺾임 ...   돌고 나서 넘는다
    ```

    `level_after` 는 구간마다의 **절대 단**이다. 정수 하나면 전 구간 같은 값이고,
    순서열이면 구간별이다. 경사는 자기 타일에서 단이 바뀌어야 그려진다.

    반환값 -- `(height, ceiling, plan)`. 키는 다른 복도와 같다.
    """
    kinds = [int(k) for k in kinds]
    n = len(kinds)
    assert int(run) >= 2, "장애물이 꺾는 칸과 겹칩니다. run 을 2 이상으로"
    assert int(run) >= 2, "장애물이 꺾는 칸과 겹칩니다. run 을 2 이상으로"
    if np.isscalar(after_turn):
        after = [bool(after_turn)] * n
    else:
        after = [bool(v) for v in after_turn]
        assert len(after) == n, f"장애물 위치는 구간 수({n})만큼 주세요"
    if np.isscalar(level_after):
        lv = [int(level_after)] * n
    else:
        lv = [int(v) for v in level_after]
        assert len(lv) == n, f"단은 구간 수({n})만큼 주세요"

    tx = n * int(run) + 1
    n_rows = 4                      # 테두리 + 윗줄 + 아랫줄 + 테두리
    top, bot = 1, 2
    rows = [(top if i % 2 == 0 else bot) for i in range(n)]
    if mirror:
        rows = [(bot if r == top else top) for r in rows]

    grid = np.full((n_rows, tx), int(separator), dtype=np.int8)
    levels = np.zeros((n_rows, tx), dtype=np.int8)
    axes = np.full((n_rows, tx), maze.RUN_X, dtype=np.int8)

    turns, obstacles = [], []
    for i, k in enumerate(kinds):
        c0, c1 = i * int(run), (i + 1) * int(run)
        r = rows[i]
        for c in range(c0, min(c1, tx - 1) + 1):
            grid[r, c] = maze.FLAT
            levels[r, c] = lv[i - 1] if (i > 0 and c <= c0) else lv[i]
        # 구간 안 장애물 자리. **꺾임과의 거리가 이 판의 난이도 전부다.**
        #   after=True   들어오자마자 (앞 꺾임에서 한 칸)  -> 돌고 나서 정렬
        #   after=False  나가기 직전 (다음 꺾임에서 한 칸) -> 넘고 나서 돌기
        obs_c = (c0 + 1) if after[i] else (c1 - 1)
        grid[r, obs_c] = k
        obstacles.append((r, obs_c))
        # 장애물 앞은 이전 단, 뒤는 이 구간의 단. 경사가 그려지려면 단이 바뀌는
        # 자리가 **장애물 타일 자신**이어야 한다.
        for c in range(c0, obs_c + 1):
            levels[r, c] = lv[i - 1] if i > 0 else 0
        if i + 1 < n:               # 꺾는 칸. 두 줄을 세로로 잇는다
            r2 = rows[i + 1]
            lo, hi = sorted((r, r2))
            for rr in range(lo, hi + 1):
                grid[rr, c1] = maze.FLAT
                levels[rr, c1] = lv[i]
                axes[rr, c1] = maze.RUN_Y
            turns.append((r, c1))
            turns.append((r2, c1))

    _raise_walls(grid, levels, separator)
    height = maze.heightfield(grid, levels, axes, seed)
    ceiling = maze.ceilings(grid, levels, axes)

    ex, ey = tx * maze.TILE, n_rows * maze.TILE
    def centre(row, col):
        return np.asarray([(col + 0.5) * maze.TILE - ex / 2,
                           (row + 0.5) * maze.TILE - ey / 2], dtype=np.float32)

    start = centre(rows[0], 0)
    goal = centre(rows[-1], tx - 1)
    route = np.stack([start] + [centre(r, c) for r, c in turns] + [goal])

    return height, ceiling, {
        "kind": kinds[0],
        "start_xy": start,
        "goal_xy": goal,
        "lane_start_xy": start[None, :].copy(),
        "lane_goal_xy": goal[None, :].copy(),
        "lane_kind": np.asarray([kinds[0]], dtype=np.int64),
        "lane_level": np.asarray([lv[-1]], dtype=np.int64),
        "lane_route": route[None, :, :].copy(),
        "obstacle_x": (obstacles[0][1] + 0.5) * maze.TILE - ex / 2,
        "level_after": lv[-1],
        "axis": maze.RUN_X,
        "extent": (ex, ey),
        "shape": (n_rows, tx),
        "level_grid": levels.astype(np.float32),
        "jitter_xy": maze.TILE / 2 - 0.35,
        "lane_y": ey / 2 - 0.3,
    }



#: `maze_segments(reverse=True)` 가 잇는 차선별 배열. 나머지 키는 스칼라라
#: 앞 판의 것을 그대로 쓴다.
LANE_KEYS = ("lane_start_xy", "lane_goal_xy", "lane_kind", "lane_level",
             "lane_yaw", "lane_counts", "lane_tiles", "lane_reverse")


def _route_reversed(mz):
    """경로만 뒤집은 미로 사본.

    지형 배열은 그대로 **공유한다** -- 뒤집는 것은 정답지를 읽는 순서뿐이고,
    hfield 도 천장도 같은 지도다. 복사하면 8x16 에서 1.3 MB, 64x64 에서
    41 MB 를 헛되이 쓴다.
    """
    import dataclasses

    return dataclasses.replace(
        mz, route=np.ascontiguousarray(np.asarray(mz.route)[::-1]))


def _join_lanes(a: dict, b: dict) -> dict:
    """두 판의 차선 배열을 잇는다.

    `lane_route` 만 폭이 다를 수 있어 따로 맞춘다 -- 두 방향의 가장 긴 구간이
    다르면 `np.concatenate` 가 그냥 터진다. 채우는 값은 **그 구간의 목표점**
    이라야 한다 (`maze_segments` 가 쓰는 규칙과 같다). 0 으로 채우면 원점이
    경로에 끼어들어 남은 거리가 튄다.
    """
    # **어느 쪽이 역방향인지 남긴다.** 표에서 차선 번호만 보면 역방향 차선이
    # 0.000 일 때 "못 배웠다"인지 "그 방향으로는 불가능하다"인지 구분이 안 된다.
    b = dict(b, lane_reverse=np.ones(len(b["lane_start_xy"]), dtype=bool))

    out = dict(a)
    for k in LANE_KEYS:
        if k in a and k in b:
            out[k] = np.concatenate([a[k], b[k]])

    ra, rb = a["lane_route"], b["lane_route"]
    width = max(ra.shape[1], rb.shape[1])

    def pad(r):
        if r.shape[1] == width:
            return r
        tail = np.repeat(r[:, -1:, :], width - r.shape[1], axis=1)
        return np.concatenate([r, tail], axis=1)

    out["lane_route"] = np.concatenate([pad(ra), pad(rb)]).astype(np.float32)
    return out


def maze_segments(mz, *, span: int = 6, stride: int = 1, tail: bool = False,
                  reverse: bool = False):
    """**미로 하나 위의 여러 구간.** 조각을 손으로 까는 것을 끝내려고 만들었다.

    ```
        정답지  o--o--[돌]--o--o--[다리]--o--o--@
                |--- 구간 0 ---|
                   |--- 구간 1 ---|
                      |--- 구간 2 ---|          span 칸씩 겹쳐가며 전부
    ```

    구간 하나가 차선 하나다. 출발은 정답지 위 어느 칸이든 되고, 목표는 거기서
    **`span` 칸 앞**이다. 미로의 최종 목표가 아니다.

    왜 목표를 앞으로 옮기는가
    -------------------------

    목표를 최종 목표에 고정하면 차선들이 **같은 코스의 뒷부분끼리 겹친다.**
    실측 -- 20칸짜리 미로에서 목표까지 2~6칸으로 다섯 차선을 뽑았더니, 4 m 짜리가
    12 m 짜리의 꼬리와 완전히 같았다. 차선은 다섯인데 상황은 하나였다. 게다가
    쓰는 구간이 끝의 12 m 뿐이라 **38 m 중 26 m 가 학습에 안 나왔다** -- 그
    미로의 다리와 거침이 전부 그 버려지는 구간에 있었다.

    목표를 앞으로 옮기면 정답지의 모든 자리가 차선이 된다. 길이가 전부 같아
    `MAX_STEPS` 하나로 맞고, 미로가 낸 장애물이 빠짐없이 배치에 들어온다.

    목표가 경유점이 되는 것은 오히려 실제 배치에 가깝다 -- nav 가 HLC 에 주는
    것은 원래 국소 목표점이다.

    난이도
    ------

    축이 **`span`** 하나다. 키우면 거리와 **에피소드당 장애물 개수**가 같이
    는다. S자에서 무너진 축이 정확히 그것이다 -- 꺾임 뒤 다리가 차선 판에서
    1.000 인데 그것이 연속으로 나오는 코스에서는 0.000 이었다.

    `stride` 는 출발 칸 간격이다. 1 이면 정답지의 모든 칸이 출발점이 된다.

    `tail` 은 기본이 False 다. `span` 칸을 다 못 채우는 끝부분을 버린다. True 로
    두면 마지막 구간들이 2 m · 4 m 로 짧아지는데, 목표가 어차피 경유점이라
    **미로의 진짜 목표에 특별한 지위가 없다.** 짧은 판은 쉬운 자리를 배치에
    채워 넣을 뿐이다. 실측 -- 20칸 미로에 span 6 이면 True 는 19차선(그중 5개가
    6 m 미만), False 는 14차선 전부 12 m 다.

    한계 (팔지 않는다)
    ------------------

    미로 하나에 정답 경로는 **한 갈래**다. 경로를 여럿 쓰려면 미로가 여럿이고,
    hfield 는 배치마다 하나라 **한 배치에 못 섞는다.** 차선 판과 똑같은 제약이라
    미로를 바꾸려면 순차로 학습해야 한다. 그리고 이 방식은 "서 있는 것이
    넘어지는 것보다 싸다"는 보상 문제를 전혀 안 고친다.

    반환값 -- `(height, ceiling, plan)`. 키는 다른 복도와 같고 `lane_yaw` 와
    `lane_tiles` 가 는다. 미로는 경로가 +x 로만 가지 않으므로 출발 방향이
    차선마다 다르다.
    """
    if reverse:
        # **정답지를 뒤로도 훑는다.** 같은 칸을 반대쪽에서 만나는 판이 없으면
        # 칸당 진입 방향이 하나로 고정된다. 실측 -- 씨앗 0 은 오르막을 행+ 로만
        # 지나는데, 오르막을 행- 로 지나는 씨앗에서 경사 차선이 전부 0.000 이
        # 됐다 (씨앗 3 은 16 차선 중 14 개, 씨앗 6 은 15 차선 중 13 개).
        #
        # 지도를 키워도 이 축은 안 늘어난다. 경로가 하나면 방향도 하나다.
        # 국소 장면 수로 재면 8x16 에서 33 -> 66 (x2.00), 64x64 에서 280 -> 418
        # (x1.49) 다. 긴 경로는 같은 패턴을 다시 밟아 배수가 준다.
        fwd = maze_segments(mz, span=span, stride=stride, tail=tail)
        bwd = maze_segments(_route_reversed(mz), span=span, stride=stride,
                            tail=tail)
        return fwd[0], fwd[1], _join_lanes(fwd[2], bwd[2])

    route = np.asarray(mz.route, dtype=np.int64)
    assert len(route) >= 2, "route 가 없는 미로입니다. maze.generate 로 만드세요"
    n = len(route)
    span = int(span)
    assert span >= 1, "구간이 한 칸은 되어야 합니다"
    shape = mz.kind.shape

    last = n - 1 if tail else n - 1 - span
    idx = list(range(0, last, max(1, int(stride))))
    # **좁은 지형 칸에서는 출발시키지 않는다.** 스폰 지터가 `TILE/2 - 0.35`
    # = 0.65 m 인데 다리 상판은 폭 0.60 m (좌우 0.30) 다. 그 칸에서 시작하면
    # 리셋하는 순간 상판 밖이다. 실측 -- 그 차선이 도달 0.333 이었고, 실패판의
    # 최저 z 가 -0.736 (도랑 바닥) 인데 넘어짐은 False 였다. 200 스텝을 도랑
    # 안에서 보낸 것이다. 다리 · 터널은 **접근해서 들어가는 것**이지 그 위에서
    # 시작하는 것이 아니다.
    idx = [i for i in idx
           if int(mz.kind[route[i][0], route[i][1]]) not in NARROW_START]
    assert idx, f"출발점이 없습니다. 경로가 {n}칸인데 span 이 {span}입니다"

    def centre(i):
        return maze.tile_center(int(route[i][0]), int(route[i][1]), shape)

    starts = np.stack([centre(i) for i in idx])
    def settle(i, j):
        """목표로 쓸 수 있는 칸으로 옮긴다. **경사면에는 목표를 두지 않는다.**

        경사 타일은 지면이 한쪽 끝 0 에서 반대쪽 끝 `HIGH` 까지 기울어 있고,
        타일 중심은 그 중턱이다. 거기를 목표로 잡으면 **미끄러지는 방향이 목표를
        지나쳐 아래로** 가고, 되돌아오려면 비탈을 거슬러 올라야 한다.

        실측 -- 그런 차선이 512판 중 도달 0.000 이었다. 시간초과 0.875 에 최저 z
        0.259(정상 보행), 보상합 +10.2 로 12 m 중 9 m 를 갔다. 잘 내려온 뒤
        목표에서 2.8 m 아래에 서 있었다.

        경사는 **지나가는 곳이지 멈추는 곳이 아니다.** 다리 · 터널을 출발점에서
        뺀 것과 같은 이유다 (`NARROW_START`).

        **앞뒤 둘 다 한도를 둔다.** 앞으로만 밀면 경사가 이어질 때 구간이 계속
        늘어나고(실측 6칸 -> 12칸 = 24 m, `MAX_STEPS` 300 초과), 뒤로만 당기면
        너무 짧아진다(실측 1칸 = 2 m, 배치를 쉬운 판으로 채운다). 언덕은 대개
        경사 칸이 1~3 개라 앞으로 3 칸이면 거의 덮는다.
        """
        def ramp(t):
            return int(mz.kind[route[t][0], route[t][1]]) == maze.RAMP

        if not ramp(j):
            return j
        for t in range(j + 1, min(n, j + 4)):      # 앞으로 최대 3칸
            if not ramp(t):
                return t
        for t in range(j - 1, i + 1, -1):          # 그래도 안 되면 뒤로
            if not ramp(t):
                return t
        return min(j + 1, n - 1)                   # 전부 경사면 포기하고 한 칸 뒤

    ends = [settle(i, min(i + span, n - 1)) for i in idx]
    goals = np.stack([centre(j) for j in ends])
    # 출발 방향. 정답지가 그 칸에서 **나가는** 방향이다. 안 주면 로봇이 +x 를
    # 보고 서는데, 미로의 경로는 세로로도 간다 -- 첫 스텝부터 벽을 본다.
    yaw = []
    for i in idx:
        d = centre(min(i + 1, n - 1)) - centre(i)
        yaw.append(float(np.arctan2(d[1], d[0])))

    # 그 구간 안에서 만나는 **첫 장애물**. 특권 관측의 `kind_onehot` · `level` 이
    # 이걸 쓴다. 구간 하나에 장애물이 여럿일 수 있어 한 칸으로는 다 못 적는다.
    # 그래서 "다음에 무엇이 오는가"로 뜻을 정한다.
    def ahead(i, j):
        for t in range(i, j + 1):
            k = int(mz.kind[route[t][0], route[t][1]])
            if k in maze.IMPLEMENTED and k != maze.FLAT:
                # **단은 그 장애물 구간이 끝난 다음 칸에서 읽는다.** 경사 칸
                # 자신의 단은 0 이다 (`_ramp` 가 이웃 단 사이를 보간하므로
                # 그래야 한다). 언덕은 경사 칸이 여러 개 이어지므로 바로 다음
                # 칸도 0 이다. 같은 종류가 이어지는 동안 건너뛰고 나서 읽는다.
                e = t
                while (e + 1 < n
                       and int(mz.kind[route[e + 1][0], route[e + 1][1]]) == k):
                    e += 1
                nr, nc = route[min(e + 1, n - 1)]
                return k, int(mz.level[nr, nc])
        er, ec = route[j]
        return int(maze.FLAT), int(mz.level[er, ec])

    kinds, levels, routes, counts = [], [], [], []
    for i, j in zip(idx, ends):
        k, lv = ahead(i, j)
        kinds.append(k)
        levels.append(lv)
        routes.append(np.stack([centre(t) for t in range(i, j + 1)]))
        # **구간 안 지형 구성.** `lane_kind` 는 "처음 만나는 것" 하나뿐이라
        # 밀도를 올리면 어느 구간이든 돌 · 거침만 찍힌다. 표에서 터널이 있는
        # 구간인지 없는 구간인지 구분이 안 되므로 개수를 따로 남긴다.
        # 특권 관측은 `lane_kind` 를 그대로 쓴다 -- 이것은 표시 전용이다.
        row = np.zeros(len(maze.IMPLEMENTED), dtype=np.int64)
        for t in range(i, j + 1):
            kk = int(mz.kind[route[t][0], route[t][1]])
            if kk in maze.IMPLEMENTED:
                row[maze.IMPLEMENTED.index(kk)] += 1
        counts.append(row)
    # 배열 하나로 쌓아야 하므로 길이를 맞춘다. **그 구간의 목표점을 반복해
    # 채운다** -- 길이 0 인 구간이라 누적 거리에 기여하지 않고, `obs._project`
    # 가 이미 그 경우를 막아 두었다 (denom 에 1e-9 검사).
    width = max(len(r) for r in routes)
    padded = np.stack([
        np.concatenate([r, np.repeat(r[-1][None, :], width - len(r), axis=0)])
        if len(r) < width else r for r in routes])

    ex, ey = mz.extent
    return mz.height, mz.ceiling, {
        "kind": kinds[0],
        "start_xy": starts[0],
        "goal_xy": goals[0],
        "lane_start_xy": starts.astype(np.float32),
        "lane_goal_xy": goals.astype(np.float32),
        "lane_kind": np.asarray(kinds, dtype=np.int64),
        "lane_level": np.asarray(levels, dtype=np.int64),
        "lane_route": padded.astype(np.float32),
        "lane_yaw": np.asarray(yaw, dtype=np.float32),
        # (차선, 랜드종류) 개수. 표시 전용이다 -- 위 주석 참고.
        "lane_counts": np.stack(counts),
        # 구간 길이 (칸). 끝쪽 구간은 `span` 보다 짧다.
        "lane_tiles": np.asarray([j - i for i, j in zip(idx, ends)],
                                 dtype=np.int64),
        # 정방향 판이다. `reverse=True` 면 `_join_lanes` 가 뒷쪽을 True 로 채운다.
        "lane_reverse": np.zeros(len(idx), dtype=bool),
        "obstacle_x": float(starts[0][0]),
        "level_after": levels[0],
        "axis": maze.RUN_X,
        "extent": (ex, ey),
        "shape": shape,
        # 단 격자. 빠짐 판정의 기준선이다 (`stage1.PIT_MARGIN` 참고).
        "level_grid": np.asarray(mz.level, dtype=np.float32),
        "jitter_xy": maze.TILE / 2 - 0.35,
        "lane_y": ey / 2 - 0.3,
    }

