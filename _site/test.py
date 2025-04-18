from collections import defaultdict
from common.gridworld import GridWorld

pi = defaultdict(lambda: {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25})
V = defaultdict(lambda: 0.0)

env = GridWorld()


def eval_onestep(pi, V, env, gamma=0.9):
    for state in env.states():
        if state == env.goal_state:
            V[state] = 1.0
            continue

        action_probs = pi[state]
        new_V = 0
        for action, action_prob in action_probs.items():
            next_state = env.next_state(state, action)
            r = env.reward(state, action, next_state)
            new_V += action_prob * (r + gamma * V[next_state])
        V[state] = new_V
    return V


def policy_eval(pi, V, env, gamma=0.9, threshold=1e-5):
    while True:
        old_V = V.copy()
        V = eval_onestep(pi, V, env, gamma)
        delta = 0
        for state in V.keys():
            t = abs(old_V[state] - V[state])
            if delta < t:
                delta = t
        if delta < threshold:
            break
    return V


V = policy_eval(pi, V, env)
env.render_v(V, pi)
