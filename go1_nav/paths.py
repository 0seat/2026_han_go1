"""경로 해석 -- 콜랩과 로컬에서 **같은 코드가 돌게** 한다.

체크포인트는 구글 드라이브에 있고 저장소는 깃허브에 있다. 콜랩에서는 드라이브를
마운트해서 읽고, 로컬에서는 드라이브 데스크톱이 만든 폴더를 읽는다. 경로가 두
군데에 하드코딩되면 노트북과 저장소가 어긋나므로 여기 한 곳에 둔다.

    콜랩    /content/drive/MyDrive/go1_walking/...
    로컬    드라이브 데스크톱이 만든 폴더를 **찾아낸다**

로컬 경로를 적어 두지 않는 이유 -- 팀 공유 폴더가 **바로가기**로 걸려 있어서
`.shortcut-targets-by-id/<32자리 id>/go1_walking` 모양이 된다. 그 id 는 공유
설정에 따라 접근 열쇠가 되므로 **공개 저장소에 적지 않는다.** 대신 드라이브
문자와 흔한 자리를 훑어 `go1_walking` 폴더를 찾는다.

못 찾거나 다른 자리를 쓰려면 환경변수 `GO1_DRIVE` 로 뿌리를 지정한다.
그쪽이 언제나 이긴다.

산출물은 드라이브에 쓰지 않는다. 콜랩 세션이 끝나면 사라지는 것이 맞다 --
남길 가치가 있는 것은 표와 영상이고, 그건 사람이 골라서 옮긴다.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

#: 드라이브에서 학습 산출물이 쌓이는 폴더 이름. 팀이 정한 것이라 여기서만 적는다.
WALKING = "go1_walking"

#: `go1_walking` 을 찾아볼 자리들. 드라이브 문자는 실행할 때 채운다.
_LOCAL_GLOBS = (
    ".shortcut-targets-by-id/*/{name}",     # 팀 공유 바로가기 (윈도우)
    "내 드라이브/{name}",
    "My Drive/{name}",
    "{name}",
)


def _local_candidates(name: str):
    """`name` 폴더가 있을 만한 자리를 훑는다. **얕게만 본다** -- 드라이브를
    통째로 뒤지면 느리고, 여기 없으면 `GO1_DRIVE` 로 알려 주는 편이 빠르다."""
    import string

    roots = []
    if sys.platform.startswith("win"):
        roots += [Path(f"{c}:/") for c in string.ascii_uppercase
                  if Path(f"{c}:/").exists()]
    else:
        home = Path.home()
        roots += [home / "Google Drive", home / "GoogleDrive",
                  Path("/Volumes/GoogleDrive"), home]
    for root in roots:
        for pat in _LOCAL_GLOBS:
            try:
                for hit in root.glob(pat.format(name=name)):
                    if hit.is_dir():
                        yield hit.parent
            except OSError:            # 접근 못 하는 드라이브는 건너뛴다
                continue

#: 콜랩에서 마운트되는 자리.
_COLAB_MOUNT = Path("/content/drive/MyDrive")


def on_colab() -> bool:
    """콜랩인가. 모듈 존재로 본다 -- 파일 존재로 보면 마운트 전에 틀린다."""
    return "google.colab" in sys.modules


def mount(force: bool = False) -> Path:
    """콜랩이면 드라이브를 마운트하고 뿌리를 낸다. 로컬이면 아무것도 안 한다.

    노트북 첫 칸에서 한 번 부른다. 이미 마운트돼 있으면 다시 안 한다 --
    `drive.mount`는 두 번 불러도 되지만 인증 창이 떠서 흐름이 끊긴다.
    """
    if on_colab() and (force or not _COLAB_MOUNT.exists()):
        from google.colab import drive
        drive.mount(str(_COLAB_MOUNT.parent), force_remount=force)
    return drive_root()


def drive_root() -> Path:
    """드라이브 뿌리. `GO1_DRIVE`가 있으면 그것이 이긴다."""
    override = os.environ.get("GO1_DRIVE")
    if override:
        return Path(override)
    if on_colab():
        return _COLAB_MOUNT
    for found in _local_candidates(WALKING):
        return found
    return Path.home()


def walking() -> Path:
    """학습 산출물 폴더. 없으면 바로 알려준다 -- 나중에 이상한 곳에서 죽지 않게."""
    p = drive_root() / WALKING
    if not p.is_dir():
        raise FileNotFoundError(
            f"{p} 가 없습니다.\n"
            f"  콜랩    paths.mount() 를 먼저 부르세요.\n"
            f"  로컬    구글 드라이브 데스크톱이 켜져 있는지 보세요."
            f" 그래도 못 찾으면 환경변수 GO1_DRIVE 에\n"
            f"          {WALKING} 의 **부모** 폴더를 지정하세요."
        )
    return p


def checkpoint(phase: str, run: str | None = None, name: str = "final_checkpoint") -> Path:
    """`walking()/<phase>/<run>/<name>`. `run`을 비우면 **가장 최근 것**을 고른다.

    가장 최근을 고르는 규칙은 폴더 이름의 사전순이다. 이 저장소들이 전부
    `YYYYMMDD_HHMMSS`라 사전순이 곧 시간순이다.
    """
    base = walking() / phase
    if not base.is_dir():
        raise FileNotFoundError(f"{base} 가 없습니다.")
    if run is None:
        runs = sorted(p.name for p in base.iterdir()
                      if p.is_dir() and p.name[:8].isdigit())
        if not runs:
            raise FileNotFoundError(f"{base} 안에 실행 폴더가 없습니다.")
        run = runs[-1]
    path = base / run / name
    if not path.is_dir():
        raise FileNotFoundError(f"{path} 가 없습니다.")
    return path


#: 지금 기준 LLC. 바뀌면 여기만 고친다 -- 노트북에 경로를 적지 말 것.
LLC_PHASE = "phase18_speed_fwd"
LLC_RUN = "20260815_115644"


def llc() -> Path:
    """지금 쓰는 LLC 체크포인트. `llc/spec.py`의 `SOURCE`와 같은 것을 가리킨다."""
    return checkpoint(LLC_PHASE, LLC_RUN)


def params_file(spec) -> Path:
    """체크포인트 pkl 을 찾는다. **드라이브가 없어도 되게 한다.**

    전에는 `walking() / spec` 하나였다. 그러면 구글 드라이브 공유 폴더가 붙어
    있는 사람만 쓸 수 있다 -- pkl 파일 하나 받아서 돌려 보려는 사람이 첫
    명령에서 막힌다. 순서대로 본다.

        1  그대로 있으면 그것          절대경로도 상대경로도 된다
        2  GO1_PARAMS/<spec>          pkl 만 모아 둔 폴더가 따로 있을 때
        3  walking()/<spec>           지금까지의 방식

    2번이 있는 이유 -- 팀원에게 pkl 을 몇 개 건네줄 때 폴더 구조를 드라이브와
    똑같이 맞추라고 하는 것보다 환경변수 하나로 가리키게 하는 편이 낫다.
    """
    spec = Path(spec)
    if spec.exists():
        return spec

    tried = [str(spec)]
    base = os.environ.get("GO1_PARAMS")
    if base:
        cand = Path(base) / spec
        if cand.exists():
            return cand
        tried.append(str(cand))

    try:
        cand = walking() / spec
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"{spec} 를 못 찾았습니다."
            f"\n  본 자리   " + "  ".join(tried) +
            f"\n  그리고 드라이브도 못 찾았습니다 -- {exc}"
            f"\n  pkl 을 직접 가리키거나(절대경로), GO1_PARAMS 에 그 부모 "
            f"폴더를 지정하세요."
        ) from None
    if cand.exists():
        return cand
    tried.append(str(cand))
    raise FileNotFoundError(
        f"{spec} 를 못 찾았습니다. 본 자리:\n  " + "\n  ".join(tried))


def outputs(sub: str = "") -> Path:
    """산출물 폴더. 저장소 옆(`outputs/`)에 만든다. git에 안 올라간다."""
    root = Path(__file__).resolve().parent.parent / "outputs"
    p = root / sub if sub else root
    p.mkdir(parents=True, exist_ok=True)
    return p
