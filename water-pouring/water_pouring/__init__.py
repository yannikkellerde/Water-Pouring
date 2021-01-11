from gym.envs.registration import register

register(
    id='Pouring-Base-v0',
    entry_point='water_pouring.envs:Pouring_base',
    nondeterministic = True
)
register(
    id='Pouring-Simple-v0',
    entry_point='water_pouring.envs:Pouring_simple',
    nondeterministic = True
)
register(
    id='Pouring-mdp-v0',
    entry_point='water_pouring.envs:Pouring_mdp',
    nondeterministic = True
)
register(
    id='Pouring-mdp-full-v0',
    entry_point='water_pouring.envs:Pouring_mdp_full',
    nondeterministic = True
)