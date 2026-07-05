from mpe import MPEenv
# from functools import partial
# from multiagentenv import MultiAgentEnv

# def env_fn(env, **kwargs) -> MultiAgentEnv:
#     return env(**kwargs)
    
# a = partial(env_fn, env=MPEenv)
# env = a("navigation", 3)
env = MPEenv.MPEenv("shooting", 3)
while True:
    obs, state = env.reset()
    done = False
    reward = 0
    print(f"state: {state}")
    print(f"obs: {obs}")
    
    while not done:
        avail = env.get_avail_actions()
        # print(f"left_occupy: {env.left_occupy}")
        # print(f"occupy_map: {env.occupy_map}")
        # print(f"avail_actions: {avail}")
        
        
        
        input_actions = input()
        actions = [int(x) for x in input_actions.split(',')]
        reward, done, info = env.step(actions)
        obs = env.get_obs()
        state = env.get_state()
        
        # print(f"left_occupy: {env.is_unlock}")
        # print(f"avail_actions: {avail}")
        
        print(f"num_bullets: {env.num_bullets}")
        print(f"avail_actions: {avail}")
        
        print(f"state: {state}")
        print(f"obs: {obs}")
        print(f"reward: {reward}")
        print(f"done: {done}")



