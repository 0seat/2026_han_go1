"""11차원 명령 Go1 환경 — playground `Go1JoystickFlatTerrain`의 최소 파생.

설계 원칙: **`_get_obs`도 `_get_reward`도 건드리지 않는다.**

playground의 Go1 joystick은 관측을 이렇게 조립한다 (명령이 맨 뒤).

    state = jp.hstack([linvel, gyro, gravity, qpos-default, qvel, last_act,
                       info["command"]])

즉 `info["command"]`의 길이만 3에서 11로 바꾸면 관측이 48에서 56으로 **저절로**
늘어난다. `privileged_state`도 `state`를 앞에 물고 있으므로 123에서 131이 된다.
체크포인트의 실제 shape가 정확히 56/131이므로, phase12가 한 일도 이것일 가능성이
매우 높다. 관측 조립 코드를 베껴 쓰지 않는 편이 훨씬 안전하다 -- WTW 이식에서
관측 순서를 손으로 옮기다 DOF 순서를 틀려 며칠을 날렸다.

이 가설이 틀렸다면 `gate()`가 첫 줄에서 잡는다. 통과하지 못하면 스윕을 돌리지 말 것.

주의 사항 두 가지

    1. PD를 거는 자리가 WTW 이식과 반대다. playground는 Kd를 `dof_damping[6:]`에
       넣고 `actuator_biasprm[:, 2]`는 건드리지 않는다. `docs/01_llc.md` 8.2절은
       정반대로 적혀 있으니 습관대로 옮기면 감쇠가 이중으로 걸린다.
       이 모듈은 playground의 기본 모델을 그대로 쓰므로 문제되지 않지만,
       나중에 지형을 붙일 때 다시 만난다.

    2. 부모는 `steps_until_next_cmd`가 0이 되면 명령을 스스로 재샘플한다.
       스윕은 명령을 고정해야 하므로 `hold()`가 매 스텝 되돌린다.
"""

from __future__ import annotations

import contextlib
import io
import os
import re
import shutil
from pathlib import Path

import jax
import jax.numpy as jp
import numpy as np

from ..llc import spec
from . import maze


_HFIELD_RE = re.compile(r"<hfield\b[^>]*?/>")


#: 천장 전용 충돌 그룹. 왜 새 그룹이 필요한가 --
#:
#: `go1_mjx_feetonly.xml`은 `class="go1"` 기본값에서 `contype=0 conaffinity=0`으로
#: **몸 전체의 충돌을 꺼두고**, 발 구만 `conaffinity=1`로 되살린다. 바닥은
#: `contype=1`이라 발만 바닥과 부딪힌다. 몸통 충돌 geom은 XML에 있지만 죽어 있다.
#: 그래서 천장을 그냥 놓으면 **로봇이 그대로 통과한다.**
#:
#: 몸통을 되살리되 기존 물리를 건드리지 않으려면 비트를 나눠야 한다.
#:
#:     바닥    contype=1  conaffinity=0
#:     발      contype=0  conaffinity=1     ->  바닥과 부딪힘 (그대로)
#:     천장    contype=2  conaffinity=0
#:     몸통    contype=0  conaffinity=2     ->  천장과만 부딪힘
#:
#: 여기서 "몸통"은 trunk의 **박스 geom 하나**다. 다리까지 살리면 mjx가 실린더 x
#: 박스 충돌을 구현하지 않아 모델 변환에서 죽는다.
#:
#: 몸통과 바닥은 `1 & 2 = 0`이라 여전히 안 부딪힌다. phase14가 학습된 조건이
#: 그대로 유지된다. 발과 천장도 안 부딪히는데(2 & 1 = 0), 발을 천장까지 드는
#: 상황은 드물어 감수한다.
_CEILING_BIT = 2


@contextlib.contextmanager
def _hfield_resolution(nrow: int, ncol: int, ceiling=None, texture=None):
    """씬 XML의 hfield 선언을 `maze`의 해상도로 바꿔 끼운다.

    playground의 rough 씬은 `assets/hfield.png`(256 x 256, 20 x 20 m)로 격자를
    정한다. 해상도도 맵 크기도 컴파일 때 확정되므로 모델을 만든 뒤에는 못 바꾼다
    -- `hfield_data`의 크기가 이미 고정돼 있다. 그래서 **XML 단계에서** 갈아끼운다.
    맵이 정사각형이 아니어도 여기서 따라간다.

    씬 파일을 복사해 오지 않는 이유 -- 그 XML은 `<include>`와 상대경로로
    menagerie 메시를 물고 있어서, 옮기면 경로가 전부 깨진다. playground의
    자산 로더를 그대로 태우는 편이 안전하다.

    치환이 안 되면 조용히 256으로 도는 대신 여기서 멈춘다. 해상도가 다르면
    턱의 각도가 달라지고, 그것은 "정책이 턱을 못 넘는다"로 나타나서 원인을
    찾기 어렵다.
    """
    import mujoco

    original = mujoco.MjModel.from_xml_string
    replaced = []

    half_x = ncol * maze.CELL / 2
    half_y = nrow * maze.CELL / 2

    def patched(xml, assets=None, *args, **kwargs):
        new, n1 = _HFIELD_RE.subn(
            f'<hfield name="hfield" nrow="{nrow}" ncol="{ncol}" '
            f'size="{half_x} {half_y} {maze.SPAN} {maze.BASE}"/>',
            xml,
        )
        # 격자 값 0이 -DEPTH를 뜻하므로 바닥을 그만큼 내려야 평지가 z=0에 온다.
        # **컴파일 뒤에 `geom_pos`를 고쳐서는 안 된다** -- worldbody에 붙은 정적
        # geom은 `geom_xpos`가 컴파일 때 구워지고 `mj_forward`가 다시 계산하지
        # 않는다. 여기서 한 번 걸렸다. XML에서 내려야 CPU와 mjx가 같이 움직인다.
        new, n2 = re.subn(r'<geom name="floor"',
                          f'<geom name="floor" pos="0 0 {-maze.DEPTH}"', new)

        # 바닥 색. 씬은 바위 텍스처를 5 x 5로 반복해 깐다. 그 자산 바이트를
        # 우리 그림으로 갈아 끼우고 반복을 1 x 1로 바꾸면, XML에 요소를 더하지
        # 않고도 맵 전체에 한 장이 정확히 덮인다.
        if texture is not None:
            from PIL import Image
            buf = io.BytesIO()
            # 텍스처의 v축은 위에서 아래로 가고 hfield의 행은 -y에서 +y로 간다.
            # 뒤집지 않으면 색만 남북이 바뀐 채로 깔린다.
            Image.fromarray(np.asarray(texture, dtype=np.uint8)[::-1]).save(
                buf, format="PNG")
            assets = dict(assets or {})
            assets["rocky_texture.png"] = buf.getvalue()
            new = new.replace('texuniform="true" texrepeat="5 5"',
                              'texuniform="false" texrepeat="1 1"')

        if ceiling is not None and len(ceiling):
            boxes = "\n".join(
                f'    <geom name="ceiling_{i}" type="box" '
                f'pos="{b[0]:.6f} {b[1]:.6f} {b[2]:.6f}" '
                f'size="{b[3]:.6f} {b[4]:.6f} {b[5]:.6f}" '
                f'contype="{_CEILING_BIT}" conaffinity="0" priority="1" '
                f'friction="1.0" rgba="0.7 0.66 0.6 1"/>'
                for i, b in enumerate(ceiling)
            )
            new, n3 = re.subn(r"</worldbody>", boxes + "\n  </worldbody>", new)

            # **먼 천장은 계산하지 않는다.** mjx 는 충돌 쌍을 모델 컴파일 때
            # 정적으로 펼치므로, 천장이 75 개면 로봇이 어디에 있든 매 스텝
            # 몸통 x 천장 75 쌍을 전부 돌린다. 게다가 박스 x 박스라 mjx 충돌
            # 중 제일 비싸다. 실측 (64x64, 로컬 CPU 1환경)
            #
            #     천장 5개  44.1 스텝/초        천장 75개  25.9 스텝/초
            #
            # `max_geom_pairs` 는 경계구 거리로 싸게 추린 뒤 **가까운 N 쌍에만**
            # 충돌 함수를 돌린다 (`mjx/_src/collision_driver.py` 의 `collision`).
            # 실측 -- 75 개를 그대로 두고 25.9 -> 38.9 스텝/초, **1.50 배**다.
            # 터널은 하나도 안 잃는다.
            #
            # 천장이 적어도 손해가 없어 조건 없이 켠다.
            #
            #     천장  5개(64x64)  끔 39.6  켬 40.4
            #     천장 10개         끔 39.2  켬 41.5
            #     천장 20개         끔 44.1  켬 45.9
            #     8x16 천장 5개     끔 40.8  켬 40.2
            #
            # **주의 — 이 값들은 정책을 태워 잰 것이다.** 0 행동으로 밟으면
            # 로봇이 넘어지고, 그 뒤로는 지형이 아니라 **누운 자세**가 접촉
            # 개수를 정한다. 그 자로 재서 "지도 크기가 2.31 배" "작은 판이
            # 1.5 배 느려짐" 같은 틀린 결론을 여러 번 냈다. 스텝 비용을 잴 때는
            # 정책을 태우고 넘어진 판을 버릴 것.
            #
            # **지형은 안 잘린다.** `_GEOM_NO_BROADPHASE` 가 hfield 와 plane 을
            # 빼 준다. 이 모델의 충돌 그룹은 (박스, 박스) 75 쌍과 (hfield, 구)
            # 4 쌍뿐이라, 잘리는 것은 천장 그룹 하나다.
            #
            # 4 인 이유 -- 터널 75 칸 중 67 칸이 외딴 한 칸이고 붙은 것도 2 칸이라
            # 로봇 근처에 천장이 둘을 넘지 않는다. 4 는 여유분이다.
            #
            # **주의 —** 비트 단위로 같지는 않다. `top_k` 가 접촉 배열의 순서를
            # 바꾸고 그것이 솔버의 합 순서를 바꾼다. 잘린 쌍은 멀어서 접촉이
            # 없었으므로 물리가 아니라 반올림이 달라지는 것이지만, 넘어지는
            # 중인 로봇은 그 차이를 증폭한다. 실측 3 판에서 도달 · 넘어짐은 모두
            # 같고 스텝 수가 24 -> 27 로 갈린 판이 하나 있었다.
            # **예전 영상은 그대로 재현되지 않는다.**
            new, n5 = re.subn(
                r"</mujoco>",
                '<custom><numeric name="max_geom_pairs" data="4"/></custom>'
                "</mujoco>", new)

            # 몸통 충돌을 천장 그룹에만 되살린다. 이 줄은 include 되는 로봇 XML에
            # 있으므로 자산 dict 쪽을 고쳐야 한다.
            #
            # **몸통 박스 하나만 살린다.** `class="go1"` 기본값을 통째로 바꾸면
            # 다리의 실린더 · 캡슐까지 살아나는데, mjx가 실린더 x 박스 충돌을
            # 구현하지 않아 `put_model`이 NotImplementedError로 죽는다.
            # 천장에 닿아야 하는 것은 몸통이므로 박스 하나로 충분하다.
            assets = dict(assets or {})
            key = "go1_mjx_feetonly.xml"
            body = assets[key].decode("utf-8")
            body, n4 = re.subn(
                r'<geom class="collision" size="0\.125 0\.04 0\.057" type="box"/>',
                f'<geom class="collision" size="0.125 0.04 0.057" type="box" '
                f'conaffinity="{_CEILING_BIT}"/>', body)
            assets[key] = body.encode("utf-8")
            replaced.append(min(n1, n2, n3, n4, n5))
        else:
            replaced.append(min(n1, n2))
        return original(new, assets, *args, **kwargs)

    mujoco.MjModel.from_xml_string = patched
    try:
        yield replaced
    finally:
        mujoco.MjModel.from_xml_string = original


#: 저장소가 들고 다니는 로봇 에셋. `mujoco_playground` 가 여기를 보게 만든다.
#:
#: `<repo>/assets/mujoco_menagerie/unitree_go1` 12 MB 다. 통째 menagerie 는 훨씬
#: 크지만 Go1 XML 이 참조하는 것은 이 폴더 하나뿐이라 그것만 담는다.
ASSETS = Path(__file__).resolve().parents[2] / "assets" / "mujoco_menagerie"


def _ensure_menagerie():
    """로봇 메시를 제자리에 놓는다. **매 세션 clone 하지 않기 위해서다.**

    `mujoco_playground` 는 메시를 패키지에 담지 않고 `mujoco_menagerie` 를 고정
    커밋으로 clone 해서 쓴다. 그런데 clone 을 자동으로 하지 않아서, 없는 환경
    (새 콜랩 세션 등)에서는 XML 의 상대경로가 허공을 가리키며 이렇게 죽는다.

        ValueError: Error opening file
        '../../../../../../mujoco_menagerie/unitree_go1/assets/trunk.stl'

    메시지가 경로만 말하고 무엇을 해야 하는지는 안 말해 준다.

    `ensure_menagerie_exists()` 를 부르면 되지만 **세션마다 다시 받는다.** 콜랩은
    런타임이 초기화될 때마다 site-packages 가 날아가므로 매번 수 분이다. 그래서
    저장소가 에셋을 들고 다니고, 여기서 심볼릭 링크만 건다.

        1  이미 있으면 아무것도 안 한다            로컬은 여기서 끝난다
        2  저장소 사본이 있으면 링크 (없으면 복사)  콜랩은 여기서 끝난다. 즉시
        3  둘 다 없으면 playground 에게 맡긴다     clone. 마지막 수단
    """
    from mujoco_playground._src import mjx_env

    target = Path(str(mjx_env.MENAGERIE_PATH))
    if target.exists():
        return
    if ASSETS.is_dir():
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.symlink(ASSETS, target, target_is_directory=True)
        except OSError:
            # 윈도우는 심볼릭 링크에 권한이 필요하다. 12 MB 라 복사해도 싸다.
            shutil.copytree(ASSETS, target)
        return
    mjx_env.ensure_menagerie_exists()


def _import_playground():
    """무거운 import를 함수 안으로. 로컬(JAX 없음)에서 spec만 읽을 수 있게."""
    from mujoco_playground import registry
    _ensure_menagerie()
    from mujoco_playground._src.locomotion.go1 import go1_constants as go1_consts
    from mujoco_playground._src.locomotion.go1.joystick import Joystick
    return registry, go1_consts, Joystick


def make(noise_level: float = 0.0, command_dim: int = spec.DIM, terrain=None,
         ceiling=None, texture=None):
    """11차원 명령 env를 만든다.

    terrain -- `maze.heightfield()`가 낸 (NROW, NCOL) 격자. None이면 기존대로
    평면 위에서 돈다.

    ceiling -- `maze.ceilings()`가 낸 (N, 6) 천장 박스. 터널이 없으면 None.
    **개수가 모델에 박히므로 배치 전체가 같아야 한다.**

    texture -- `maze.texture()`가 낸 (NROW, NCOL, 3) 색 배열. 바닥 재질에 깔아
    3D에서도 랜드 종류를 색으로 본다. 물리에는 영향이 없다.

    평지 격자는 값이 전부 0이라 표면이 정확히 z=0에 놓인다 (CPU 레이캐스트로 확인).

    주의 -- **평면과 평지 hfield의 성적을 그대로 비교하면 안 된다.** 표면
    높이는 같지만 두 씬의 다른 설정이 함께 바뀐다.

        scene_mjx_feetonly_flat_terrain     home z=0.278, friction 0.6, njmax 40
        scene_mjx_feetonly_rough_terrain    home z=0.35,  friction 1.0, njmax 60

    시작 높이가 7 cm 높고 마찰이 다르다. 무제어(액션 0)로 재면 평면에서는
    서 있고 평지 hfield에서는 뒤집히는데, 그것은 지형이 아니라 낙하 높이 탓이다.
    지형만 보려면 두 씬의 keyframe과 friction을 맞춰야 한다.

    noise_level=0.0이 기본인 이유 -- 스윕은 명령 하나당 응답 하나를 재는 것이고,
    관측 노이즈는 그 응답에 분산만 더한다. 시드 간 편차를 노이즈가 아니라 정책의
    상태 의존성으로 읽으려면 여기서는 꺼야 한다. 노이즈 강건성은 별도 측정이다.
    """
    registry, go1_consts, Joystick = _import_playground()

    class Go1Command11(Joystick):
        """명령 슬롯만 3 -> command_dim으로 늘린 joystick."""

        def __init__(self, config):
            self._command_dim = int(command_dim)
            self._default_command = jp.asarray(
                spec.baseline_vector()[:self._command_dim], dtype=jp.float32
            )
            # 지형이 있으면 hfield가 선언된 씬을 연다. playground의 Joystick은
            # task가 "rough"로 시작할 때 naconmax · njmax를 키우는데, hfield는
            # 접촉이 훨씬 많이 생기므로 그 설정이 필요하다.
            if terrain is None:
                super().__init__(task="flat_terrain", config=config)
            else:
                nrow, ncol = np.asarray(terrain).shape
                with _hfield_resolution(nrow, ncol, ceiling, texture) as replaced:
                    super().__init__(task="rough_terrain", config=config)
                if sum(replaced) != 1:
                    raise RuntimeError(
                        f"씬 XML의 hfield 선언을 {sum(replaced)}개 바꿨습니다 (1개여야 합니다). "
                        f"playground의 rough 씬이 바뀌었을 수 있습니다."
                    )
                self._install_terrain(terrain)

            # 접촉 센서 id. 부모 버전에 따라 이름이 달라 직접 만든다.
            self._floor_found_sensor = [
                self._mj_model.sensor(f"{geom}_floor_found").id
                for geom in go1_consts.FEET_GEOMS
            ]
            self._floor_found_adr = np.asarray(
                [self._mj_model.sensor_adr[i] for i in self._floor_found_sensor]
            )

        # ---------- 지형 ----------
        def _install_terrain(self, height):
            """높이 격자를 모델에 굽는다. **`put_model` 전에 끝나야 한다.**

            playground의 `Go1Env.__init__`은 XML을 읽은 직후 `mjx.put_model`을
            부른다(base.py:67). 그 뒤에 `_mj_model`만 고치면 GPU에 올라간
            모델은 안 바뀐다 -- 화면에는 지형이 보이는데 발은 평면을 딛는다.
            그래서 여기서 다시 올린다.

            `hfield_size`와 해상도는 `np.ndarray`라 배치 전체가 공유한다.
            환경마다 다르게 할 수 있는 것은 `hfield_data`뿐이다.
            """
            from mujoco import mjx

            m = self._mj_model
            nrow, ncol = int(m.hfield_nrow[0]), int(m.hfield_ncol[0])
            h = np.asarray(height, dtype=np.float32)
            if h.shape != (nrow, ncol):
                raise ValueError(
                    f"높이 격자 {h.shape}가 모델의 hfield {(nrow, ncol)}와 다릅니다."
                )
            m.hfield_data[:] = h.reshape(-1)
            self._mjx_model = mjx.put_model(m, impl=self._config.impl)

        # 부모의 자동 재샘플을 무력화.
        # playground 0.2.0의 시그니처는 `sample_command(rng, x_k)`이고 본체가
        # `x_k - w*(x_k - y*z)` (y,z,w는 shape (3,))라 11차원 x_k를 넣으면
        # 브로드캐스트에서 터진다. 그대로 돌려주는 것이 유일하게 안전하다.
        def sample_command(self, rng, *args, **kwargs):
            del rng, kwargs
            if args and args[0] is not None:
                return args[0]
            return self._default_command

        def reset(self, rng):
            """부모 reset 뒤 명령을 11차원으로 승격한다.

            playground의 `reset`은 `sample_command`를 부르지 않고 `shape=(3,)`로
            명령을 직접 만든다(joystick.py:208). 그래서 `sample_command`만
            덮어써서는 관측이 48D로 나온다 -- 실제로 여기서 한 번 걸렸다.
            """
            return self.with_command(super().reset(rng), self._default_command)

        def reset_at(self, rng, xy=(0.0, 0.0), yaw=0.0, speed=0.0,
                     z_offset=0.0):
            """**자세를 지정해서** 리셋한다. 부모의 무작위 배치를 덮어쓴다.

            playground의 `reset`은 위치를 U(-0.5, 0.5) m, 요각을 U(-pi, pi),
            그리고 `qvel[0:6]`을 U(-0.5, 0.5)로 흔든다 (joystick.py). 설정으로
            끄는 길이 없다. 그대로 두면

                측정   복도를 따라 걷게 하려는데 로봇이 아무 쪽이나 보고 선다.
                       실제로 여기서 한 번 걸렸다 -- 요각 -142도로 시작해 옆으로
                       걸어 나갔고, 표에는 "턱을 못 넘었다"로 찍혔다
                학습   랜드 A -> B 과제에서 진입 조건을 우리가 못 고른다

            둘 다 자세를 우리가 정해야 한다. 그래서 **끄는 것이 아니라 덮어쓴다** --
            부모의 리셋을 그대로 부르고 `qpos`/`qvel`만 갈아 끼운 뒤 다시
            `mjx.forward`를 돌린다. 접촉과 센서가 새 자세에 맞게 다시 풀린다.

            `speed`는 진행 방향 초기 속도다. 0이면 정지 출발, 0보다 크면 앞
            랜드에서 달려 들어온 상황이 된다. 1단계 학습에서 이 인자를 흔든다.

            `z_offset`은 몸통을 그만큼 띄운다. **keyframe의 z는 평지 기준이라
            높은 자리에서 출발시키면 지형에 박힌다.** 경사 중턱에서 출발시키는
            커리큘럼이 이 인자를 쓴다 -- x 2.6 의 지형이 0.57 m 인데 몸통을
            0.35 에 놓으면 0.22 m 파묻힌 채로 시작한다.

            기본값 0 이라 평지에서 출발하는 기존 호출은 그대로다. 부르는 쪽이
            지형 높이를 알고 넘겨야 한다 -- 여기서는 높이 격자를 안 들고 있다.
            """
            from mujoco import mjx

            state = self.reset(rng)
            half = 0.5 * jp.asarray(yaw, dtype=jp.float32)
            xy = jp.asarray(xy, dtype=state.data.qpos.dtype).reshape(2)

            q = state.data.qpos.at[0:2].set(xy)
            q = q.at[2].add(jp.asarray(z_offset, dtype=q.dtype))
            # keyframe의 자세가 단위 쿼터니언이라 요 회전을 그대로 넣으면 된다.
            q = q.at[3:7].set(jp.array([jp.cos(half), 0.0, 0.0, jp.sin(half)],
                                       dtype=q.dtype))
            v = state.data.qvel.at[0:6].set(0.0)
            v = v.at[0].set(speed * jp.cos(yaw)).at[1].set(speed * jp.sin(yaw))

            data = mjx.forward(self.mjx_model,
                               state.data.replace(qpos=q, qvel=v))
            return self.with_command(state.replace(data=data),
                                     state.info["command"])

        # ---------- 명령 주입 ----------
        def with_command(self, state, command):
            """명령을 바꾸고 **관측을 다시 조립한다.**

            관측 꼬리를 `.at[-11:].set()`으로 덮어쓰는 방법도 있지만 쓰지 않는다.
            그러려면 "명령이 꼬리에 있다"와 "privileged 앞부분이 state와 같다"를
            둘 다 참으로 가정해야 하는데, 여기서는 부모의 `_get_obs`를 그대로
            불러 **가정 없이** 만든다. noise_level=0에서는 재조립이 원본과 같다.
            """
            info = dict(state.info)
            info["command"] = jp.asarray(command, dtype=state.data.qpos.dtype)
            info["steps_until_next_cmd"] = jp.asarray(1_000_000_000, dtype=jp.int32)
            return state.replace(info=info, obs=self._get_obs(state.data, info))

        def hold(self, state, command):
            """스텝 뒤에 명령을 되돌린다. `with_command`와 같지만 의도가 다르다."""
            return self.with_command(state, command)

        # ---------- 측정 ----------
        def probe(self, state):
            """한 스텝의 관측 가능량. 스칼라만 모아 (23,) 벡터로 낸다.

            배열을 그대로 쌓으면 스캔 출력이 수백 MB가 된다. 여기서 압축한다.
            """
            d = state.data
            linvel = self.get_local_linvel(d)          # 몸통 좌표계 3
            gyro = self.get_gyro(d)                    # 몸통 좌표계 3
            gravity = self.get_gravity(d)              # 몸통 좌표계 중력 단위벡터 3

            # 몸통 좌표계 중력에서 피치·롤. 요에 불변이라 이 경로를 쓴다.
            #   g_b = [sin(pitch), -cos(p)sin(r), -cos(p)cos(r)]
            pitch = jp.arcsin(jp.clip(gravity[0], -1.0, 1.0))
            roll = jp.arctan2(-gravity[1], -gravity[2])

            contact = (d.sensordata[self._floor_found_adr] > 0).astype(jp.float32)

            feet_w = d.site_xpos[self._feet_site_id]           # (4,3) 월드
            rot = d.xmat[self._torso_body_id].reshape(3, 3)
            feet_b = (feet_w - d.qpos[0:3]) @ rot              # 몸통 좌표계

            return jp.concatenate([
                linvel,                 # 0:3
                gyro,                   # 3:6
                jp.array([pitch, roll, d.qpos[2]]),   # 6:9
                contact,                # 9:13
                feet_w[:, 2],           # 13:17  발 높이 (평지 floor z=0)
                feet_b[:, 1],           # 17:21  발 좌우 위치
                jp.array([state.done, state.reward]),  # 21:23
            ])

    config = registry.get_default_config("Go1JoystickFlatTerrain")
    config.impl = "jax"
    config.noise_config.level = float(noise_level)
    if hasattr(config, "pert_config"):
        config.pert_config.enable = False   # 외란은 스윕의 신호를 가린다

    return Go1Command11(config=config)


#: `probe()` 출력의 열 이름. 정의는 `spec`이 소유한다 (JAX 없이 읽을 수 있어야 한다).
PROBE_COLUMNS = spec.PROBE_COLUMNS


def gate(env, verbose: bool = True) -> bool:
    """스윕을 돌리기 전에 통과해야 하는 구조 검사.

    여기서 걸리면 `spec.py`의 관측 배치 가설이 틀린 것이다. 스윕을 강행하면
    "정책이 명령에 반응하지 않는다"는 결론이 나오는데, 그것은 슬롯이 죽은 게
    아니라 우리가 엉뚱한 자리에 넣고 있다는 뜻이다. 구별이 불가능하므로 멈춘다.
    """
    state = jax.jit(env.reset)(jax.random.PRNGKey(0))
    obs = state.obs
    checks: list[tuple[str, bool, str]] = []

    s = tuple(obs["state"].shape)
    p = tuple(obs["privileged_state"].shape)
    checks.append(("state 56D", s == (56,), f"{s}"))
    checks.append(("privileged_state 131D", p == (131,), f"{p}"))
    checks.append(("command 11D", tuple(state.info["command"].shape) == (spec.DIM,),
                   f"{tuple(state.info['command'].shape)}"))
    checks.append(("action 12D", env.action_size == 12, f"{env.action_size}"))

    # 명령이 관측 꼬리에 있는가 -- 슬롯을 하나씩 흔들어 직접 확인한다.
    probe_cmd = jp.asarray(spec.baseline_vector()) + 1.0
    moved = jax.jit(env.with_command)(state, probe_cmd)
    delta = np.asarray(moved.obs["state"] - obs["state"])
    changed = np.flatnonzero(np.abs(delta) > 1e-6)
    tail = set(range(56 - spec.DIM, 56))
    checks.append(("명령이 state[45:56]에만 실린다", set(changed.tolist()) <= tail,
                   f"바뀐 인덱스 {changed.tolist()}"))

    mirror = np.allclose(np.asarray(obs["privileged_state"][:56]),
                         np.asarray(obs["state"]), atol=1e-6)
    checks.append(("privileged[:56] == state", mirror, "미러 아님" if not mirror else ""))

    dt = float(env.dt)
    checks.append(("제어 주기 0.02 s (50 Hz)", abs(dt - 0.02) < 1e-9, f"{dt}"))

    # probe가 도는지. 여기서 걸리면 playground 버전이 달라 접근자 이름이 바뀐 것이다.
    try:
        p = jax.jit(env.probe)(state)
        okp = tuple(p.shape) == (len(spec.PROBE_COLUMNS),)
        checks.append((f"probe {len(spec.PROBE_COLUMNS)}D", okp, f"{tuple(p.shape)}"))
    except Exception as exc:
        checks.append(("probe 실행", False, f"{type(exc).__name__}: {exc}"))

    ok = all(c[1] for c in checks)
    if verbose:
        print("=" * 66)
        print("구조 게이트")
        for name, passed, detail in checks:
            mark = "통과" if passed else "실패"
            print(f"  [{mark}] {name}" + (f"   {detail}" if detail else ""))
        print("=" * 66)
        if not ok:
            print("게이트 미통과. spec.py의 관측 배치 가설이 틀렸습니다.")
            print("스윕을 돌리면 '슬롯이 죽었다'와 '자리를 잘못 짚었다'를 구별할 수 없습니다.")
    return ok
