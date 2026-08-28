from __future__ import annotations

import torch


def test_empirical_normalization_matches_two_batch_population_statistics() -> None:
    from unilab.algos.torch.common.normalization import EmpiricalNormalization

    normalizer = EmpiricalNormalization(shape=1, device="cpu", eps=0.0)
    normalizer.update(torch.zeros(10, 1))
    normalizer.update(torch.full((10, 1), 10.0))

    torch.testing.assert_close(normalizer.mean, torch.tensor([5.0]))
    torch.testing.assert_close(normalizer.std, torch.tensor([5.0]))
    torch.testing.assert_close(
        normalizer(torch.tensor([[10.0]]), update=False),
        torch.tensor([[1.0]]),
    )


def test_empirical_normalization_round_trip_preserves_continuation() -> None:
    from unilab.algos.torch.common.normalization import EmpiricalNormalization

    uninterrupted = EmpiricalNormalization(shape=2, device="cpu")
    uninterrupted.update(torch.tensor([[0.0, 2.0], [2.0, 4.0]]))

    restored = EmpiricalNormalization(shape=2, device="cpu")
    restored.load_state_dict(uninterrupted.state_dict())
    next_batch = torch.tensor([[4.0, 8.0], [6.0, 10.0]])
    uninterrupted.update(next_batch)
    restored.update(next_batch)

    torch.testing.assert_close(restored.mean, uninterrupted.mean)
    torch.testing.assert_close(restored.std, uninterrupted.std)
    assert int(restored.count) == int(uninterrupted.count) == 4
