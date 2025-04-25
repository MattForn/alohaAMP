from gymnasium.envs.registration import register

register(
    id="gym_aloha/AlohaInsertion-v0",
    entry_point="gym_aloha.env:AlohaEnv",
    max_episode_steps=300,
    # Even after seeding, the rendered observations are slightly different,
    # so we set `nondeterministic=True` to pass `check_env` tests
    nondeterministic=True,
    kwargs={"obs_type": "pixels", "task": "insertion"},
)

register(
    id="gym_aloha/AlohaInsertion-features-v0",
    entry_point="gym_aloha.env:AlohaEnv",
    max_episode_steps=300,
    # Even after seeding, the rendered observations are slightly different,
    # so we set `nondeterministic=True` to pass `check_env` tests
    nondeterministic=True,
    kwargs={"obs_type": "features", "task": "insertion"},
)

register(
    id="gym_aloha/AlohaTransferCube-v0",
    entry_point="gym_aloha.env:AlohaEnv",
    max_episode_steps=300,
    # Even after seeding, the rendered observations are slightly different,
    # so we set `nondeterministic=True` to pass `check_env` tests
    nondeterministic=True,
    kwargs={"obs_type": "pixels", "task": "transfer_cube"},
)

register(
    id="gym_aloha/AlohaSimple",
    entry_point="gym_aloha.env:SimpleAlohaEnv",
    max_episode_steps=300,
    nondeterministic=True,
    kwargs={"obs_type": "agent_pos", "task": "simple"},
)

register(
    id="gym_aloha/SimpleAlohaEndEffector",
    entry_point="gym_aloha.env:SimpleAlohaEnvEndEffector",
    max_episode_steps=200,
    nondeterministic=True,
    kwargs={"obs_type": "ee_pos", "task": "simple_ee"},
)
