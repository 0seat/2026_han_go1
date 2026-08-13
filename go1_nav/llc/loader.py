"""체크포인트 로드 — Orbax(brax PPO) 3분할 파라미터.

체크포인트 구조 (phase14 `final_checkpoint/_METADATA` 실측)

    params[0]  normalizer   mean/std/count  state(56) privileged_state(131)
    params[1]  policy       56 -> 512 -> 256 -> 128 -> 24   (12 mean + 12 logstd)
    params[2]  value        131 -> 512 -> 256 -> 128 -> 1

출력이 24인 것은 brax의 `NormalTanhDistribution`이다. `deterministic=True`면
tanh(mean)을 쓴다. 12를 그대로 받아 쓰면 안 된다.

관측 정규화는 params[0]에 실려 brax 추론 함수가 자동 적용한다. 직접 MLP를 굴리면
이 단계를 빠뜨리는데, 정규화 없이도 그럴듯하게 걷다가 명령을 바꾸면 무너지는
형태로 나타나 원인을 찾기 어렵다. brax 경로를 벗어나지 말 것.

`inspect()`는 JAX 없이 로컬(CPU)에서도 돈다 -- Colab에 올리기 전에 체크포인트가
기대한 shape인지 확인하는 용도다.
"""

from __future__ import annotations

import json
from pathlib import Path

#: 이 스택이 검증된 버전 조합. phase14 `environment/package_versions.json`.
EXPECTED_VERSIONS = {
    "brax": "0.14.2",
    "mujoco": "3.10.0",
    "playground": "0.2.0",
}


def resolve(checkpoint_root) -> Path:
    """Orbax 체크포인트 디렉토리를 찾는다.

    brax는 실행 중에는 `<root>/<step>/`에 숫자 폴더로 쌓고, 마지막에는
    `final_checkpoint/` 바로 아래에 평평하게 저장하기도 한다. 둘 다 받는다.
    """
    root = Path(checkpoint_root)
    if (root / "_METADATA").is_file() or (root / "config.json").is_file():
        return root
    numeric = [p for p in root.iterdir() if p.is_dir() and p.name.isdigit()]
    if not numeric:
        raise FileNotFoundError(f"체크포인트를 찾지 못했습니다: {root}")
    return max(numeric, key=lambda p: int(p.name))


def inspect(checkpoint_root, verbose: bool = True) -> dict:
    """`_METADATA`에서 텐서 shape만 읽는다. JAX 불필요.

    반환값의 `observation_size`를 그대로 `load_policy`에 넘길 수 있다.
    """
    path = resolve(checkpoint_root) / "_METADATA"
    tree = json.loads(path.read_text(encoding="utf-8"))["tree_metadata"]
    shapes = {k: tuple(v["value_metadata"]["write_shape"]) for k, v in tree.items()}

    def find(needle):
        for key, shape in shapes.items():
            if needle in key:
                return shape
        return None

    obs_state = find("'mean', 'state'")
    obs_priv = find("'mean', 'privileged_state'")
    policy_in = find("'1', 'params', 'hidden_0', 'kernel'")
    value_in = find("'2', 'params', 'hidden_0', 'kernel'")

    layers = sorted(k for k in shapes if "'1', 'params', 'hidden_" in k and "kernel" in k)
    policy_out = shapes[layers[-1]][-1] if layers else None
    hidden = tuple(shapes[k][-1] for k in layers[:-1])

    info = {
        "checkpoint_dir": str(resolve(checkpoint_root)),
        "observation_size": {"state": obs_state, "privileged_state": obs_priv},
        "policy_input": policy_in[0] if policy_in else None,
        "value_input": value_in[0] if value_in else None,
        "policy_output": policy_out,
        "hidden_layer_sizes": hidden,
        "action_size": policy_out // 2 if policy_out else None,
    }

    if verbose:
        print("=" * 66)
        print("체크포인트 shape")
        print(f"  경로            {info['checkpoint_dir']}")
        print(f"  state           {obs_state}")
        print(f"  privileged      {obs_priv}")
        print(f"  policy          {info['policy_input']} -> "
              f"{' -> '.join(map(str, hidden))} -> {policy_out}  "
              f"(action {info['action_size']})")
        print(f"  value           {info['value_input']} -> 1")
        print("=" * 66)
        if obs_state != (56,):
            print("주의: state가 56D가 아닙니다. spec.py의 명령 차원 가정을 다시 볼 것.")
    return info


def load_params(checkpoint_root):
    """(normalizer, policy, value) 3분할 파라미터."""
    from brax.training import checkpoint as brax_checkpoint

    path = resolve(checkpoint_root)
    try:
        params = brax_checkpoint.load(str(path))
    except Exception as exc:                      # config.json이 없는 저장 형태
        print(f"[loader] brax 로더 실패({type(exc).__name__}), orbax로 직접 복원합니다.")
        import orbax.checkpoint as ocp
        raw = ocp.PyTreeCheckpointer().restore(str(path))
        if isinstance(raw, dict):
            params = tuple(raw[k] for k in sorted(raw, key=int))
        else:
            params = tuple(raw)

    if len(params) != 3:
        raise RuntimeError(
            f"파라미터가 3분할이 아닙니다({len(params)}개). "
            "normalizer/policy/value 구조가 아니면 추론 함수를 만들 수 없습니다."
        )
    return params


def load_policy(checkpoint_root, observation_size, action_size=12,
                hidden_layer_sizes=(512, 256, 128), deterministic=True):
    """추론 함수를 만든다.

        policy_fn = load_policy(ckpt, env_observation_size(env))
        action, _ = policy_fn(state.obs, rng)

    `observation_size`는 env에서 뽑아 넘긴다(`env_observation_size`). 체크포인트의
    `inspect()` 값과 다르면 여기서 조용히 통과한 뒤 행동이 이상해지므로,
    호출 전에 두 값을 직접 비교할 것.
    """
    from brax.training.acme import running_statistics
    from brax.training.agents.ppo import networks as ppo_networks

    params = load_params(checkpoint_root)
    networks = ppo_networks.make_ppo_networks(
        observation_size=observation_size,
        action_size=action_size,
        preprocess_observations_fn=running_statistics.normalize,
        policy_hidden_layer_sizes=tuple(hidden_layer_sizes),
        value_hidden_layer_sizes=tuple(hidden_layer_sizes),
        policy_obs_key="state",
        value_obs_key="privileged_state",
    )
    make_inference = ppo_networks.make_inference_fn(networks)
    return make_inference((params[0], params[1]), deterministic=deterministic)


def env_observation_size(env) -> dict:
    """env가 실제로 내는 관측 shape. brax 네트워크 생성에 그대로 쓴다."""
    import jax
    state = jax.eval_shape(env.reset, jax.random.PRNGKey(0))
    return {k: tuple(v.shape) for k, v in state.obs.items()}


def version_check(strict: bool = True) -> dict:
    """설치된 버전이 체크포인트를 만든 조합과 같은지."""
    import importlib.metadata as metadata

    found = {}
    for name in ["brax", "mujoco", "playground", "jax", "jaxlib", "flax",
                 "optax", "orbax-checkpoint", "mediapy"]:
        try:
            found[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            found[name] = "미설치"

    bad = {k: (v, found[k]) for k, v in EXPECTED_VERSIONS.items() if found[k] != v}
    for name, version in found.items():
        mark = " <-- 불일치" if name in bad else ""
        print(f"  {name:18s} {version}{mark}")
    if bad and strict:
        raise RuntimeError(f"버전 불일치: {bad}")
    return found
