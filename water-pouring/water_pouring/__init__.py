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
    id='Pouring-simple-no-fix-v0',
    entry_point='water_pouring.envs:Pouring_simple_no_fix',
    nondeterministic = True
)
register(
    id='Pouring-no-fix-v0',
    entry_point='water_pouring.envs:Pouring_no_fix',
    nondeterministic = True
)