from gym.envs.registration import register

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
register(
    id='Pouring-featured-v0',
    entry_point='water_pouring.envs:Pouring_featured',
    nondeterministic = True
)
register(
    id='Pouring-g2g-mdp-v0',
    entry_point='water_pouring.envs:Pouring_G2G_mdp',
    nondeterministic = True
)
register(
    id='Pouring-g2g-featured-v0',
    entry_point='water_pouring.envs:Pouring_G2G_featured',
    nondeterministic = True
)