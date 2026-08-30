from __future__ import annotations


def test_g1_walk_environment_uses_the_reward_binding_owner() -> None:
    from unilab.envs.locomotion.g1.joystick import G1WalkEnv
    from unilab.envs.locomotion.g1.walk_reward_bindings import G1WalkRewardBindings

    assert issubclass(G1WalkEnv, G1WalkRewardBindings)
    assert G1WalkEnv._compute_reward is G1WalkRewardBindings._compute_reward
