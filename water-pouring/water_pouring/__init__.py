from gym.envs.registration import register

register(
    id='Pouring-Nondeterministic-v0',
    entry_point='water_pouring.envs:Pouring_nondeterministic',
    nondeterministic = True
)
register(
    id='Pouring-Deterministic-v0',
    entry_point='water_pouring.envs:Pouring_deterministic',
    nondeterministic = False
)