from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from unilab.algos.torch.common.normalization import EmpiricalNormalization
from unilab.algos.torch.fast_sac.learner import FastSACLearner, SACActor
from unilab.algos.torch.offpolicy.checkpoint_adapter import (
    G1_HEIGHT_ACTOR_ADAPTER_ID,
    adapt_g1_height_actor_state,
    adapt_g1_height_normalizer_state,
    load_g1_height_actor_warm_start,
    materialize_g1_height_actor_checkpoint,
)


def _randomized_actor(obs_dim: int) -> SACActor:
    torch.manual_seed(7)
    actor = SACActor(
        obs_dim=obs_dim,
        action_dim=3,
        hidden_dim=32,
        use_layer_norm=True,
        device="cpu",
    )
    with torch.no_grad():
        for parameter in actor.parameters():
            parameter.normal_(mean=0.0, std=0.1)
    actor.eval()
    return actor


def _normalizer_state() -> dict[str, torch.Tensor]:
    normalizer = EmpiricalNormalization(shape=98, device="cpu")
    state = normalizer.state_dict()
    state["_mean"] = torch.arange(98, dtype=torch.float32).unsqueeze(0) / 100.0
    state["_var"] = torch.arange(1, 99, dtype=torch.float32).unsqueeze(0) / 10.0
    state["_std"] = torch.sqrt(state["_var"])
    state["count"] = torch.tensor(123, dtype=torch.long)
    return state


def test_g1_height_actor_adapter_inserts_zero_column_at_command_index_96() -> None:
    actor = _randomized_actor(98)
    source = actor.state_dict()

    adapted, metadata = adapt_g1_height_actor_state(source)

    assert metadata["adapter_id"] == G1_HEIGHT_ACTOR_ADAPTER_ID
    assert metadata["first_weight_key"] == "net.0.weight"
    assert adapted["net.0.weight"].shape == (32, 99)
    torch.testing.assert_close(adapted["net.0.weight"][:, :96], source["net.0.weight"][:, :96])
    torch.testing.assert_close(adapted["net.0.weight"][:, 96], torch.zeros(32))
    torch.testing.assert_close(adapted["net.0.weight"][:, 97:], source["net.0.weight"][:, 96:])
    for key in source.keys() - {"net.0.weight"}:
        torch.testing.assert_close(adapted[key], source[key])


def test_g1_height_normalizer_adapter_preserves_prefix_suffix_and_count() -> None:
    source = _normalizer_state()

    adapted = adapt_g1_height_normalizer_state(source)

    for key in ("_mean", "_var", "_std"):
        assert adapted[key].shape == (1, 99)
        torch.testing.assert_close(adapted[key][:, :96], source[key][:, :96])
        torch.testing.assert_close(adapted[key][:, 97:], source[key][:, 96:])
    torch.testing.assert_close(adapted["_mean"][:, 96], torch.tensor([0.754]))
    torch.testing.assert_close(adapted["_var"][:, 96], torch.ones(1))
    torch.testing.assert_close(adapted["_std"][:, 96], torch.ones(1))
    torch.testing.assert_close(adapted["count"], source["count"])


def test_g1_height_actor_adapter_is_output_equivalent_with_normalized_obs() -> None:
    source_actor = _randomized_actor(98)
    target_actor = _randomized_actor(99)
    adapted_actor, _ = adapt_g1_height_actor_state(source_actor.state_dict())
    target_actor.load_state_dict(adapted_actor, strict=True)

    source_normalizer = EmpiricalNormalization(shape=98, device="cpu")
    source_normalizer.load_state_dict(_normalizer_state())
    source_normalizer.eval()
    target_normalizer = EmpiricalNormalization(shape=99, device="cpu")
    target_normalizer.load_state_dict(adapt_g1_height_normalizer_state(_normalizer_state()))
    target_normalizer.eval()

    torch.manual_seed(11)
    source_obs = torch.randn(8, 98)
    target_obs = torch.cat(
        [
            source_obs[:, :96],
            torch.full((8, 1), 0.754),
            source_obs[:, 96:],
        ],
        dim=1,
    )

    source_output = source_actor(source_normalizer(source_obs, update=False))
    target_output = target_actor(target_normalizer(target_obs, update=False))

    for source_value, target_value in zip(source_output, target_output, strict=True):
        torch.testing.assert_close(source_value, target_value, atol=1.0e-6, rtol=1.0e-6)


def test_g1_height_adapter_rejects_wrong_source_shape_and_index() -> None:
    actor = _randomized_actor(97)

    with pytest.raises(ValueError, match="source actor input dim"):
        adapt_g1_height_actor_state(actor.state_dict())
    with pytest.raises(ValueError, match="insertion index"):
        adapt_g1_height_actor_state(_randomized_actor(98).state_dict(), insertion_index=99)


def test_materialized_actor_checkpoint_records_hashes_and_omits_training_state(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "legacy.pt"
    output_path = tmp_path / "adapted.pt"
    torch.save(
        {
            "actor": _randomized_actor(98).state_dict(),
            "obs_normalizer": _normalizer_state(),
            "qnet": {"must_not_copy": torch.ones(1)},
            "actor_optimizer": {"must_not_copy": True},
        },
        source_path,
    )

    result = materialize_g1_height_actor_checkpoint(source_path, output_path)

    payload = torch.load(output_path, map_location="cpu", weights_only=True)
    sidecar = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert set(payload) == {"actor", "obs_normalizer", "actor_obs_adapter"}
    assert "qnet" not in payload
    assert "actor_optimizer" not in payload
    assert sidecar["parent_checkpoint_sha256"] == result.parent_checkpoint_sha256
    assert sidecar["output_checkpoint_sha256"] == result.output_checkpoint_sha256
    assert payload["actor_obs_adapter"]["adapter_id"] == G1_HEIGHT_ACTOR_ADAPTER_ID


def test_actor_only_warm_start_keeps_critic_and_optimizers_fresh(tmp_path: Path) -> None:
    source_path = tmp_path / "legacy.pt"
    torch.save({"actor": _randomized_actor(98).state_dict()}, source_path)
    learner = FastSACLearner(
        obs_dim=99,
        action_dim=3,
        critic_obs_dim=102,
        device="cpu",
        actor_hidden_dim=32,
        critic_hidden_dim=16,
        num_atoms=5,
        use_layer_norm=True,
        use_compile=False,
    )
    qnet_before = {key: value.clone() for key, value in learner.qnet.state_dict().items()}
    qtarget_before = {key: value.clone() for key, value in learner.qnet_target.state_dict().items()}

    metadata = load_g1_height_actor_warm_start(learner, source_path)

    assert metadata["adapter_id"] == G1_HEIGHT_ACTOR_ADAPTER_ID
    assert metadata["parent_checkpoint_sha256"]
    assert learner.actor_warm_start_metadata == metadata
    assert learner.actor_optimizer.state_dict()["state"] == {}
    assert learner.q_optimizer.state_dict()["state"] == {}
    for key, value in qnet_before.items():
        torch.testing.assert_close(learner.qnet.state_dict()[key], value)
    for key, value in qtarget_before.items():
        torch.testing.assert_close(learner.qnet_target.state_dict()[key], value)
    assert learner.get_state_dict()["actor_warm_start"] == metadata
