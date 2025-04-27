import numpy as np
from tqdm import tqdm
from collections import defaultdict
from common.gridworld import GridWorld

reward_map = np.array(
    [[0, 0, 0, -1.0, 0, None],
     [0, 0, 0, 0, -1.0, 0],
     [None, 0, -1.0, 0, 0, 0],
     [0, -1.0, 0, 0, None, 0],
     [0, None, -1.0, 0, 0, 0],
     [None, 0, 0, None, 0, 1.0]]
)

start = (0, 0)
goal = (5, 5)

env = GridWorld(reward_map, start, goal)
env.render_v()


class RandomAgent:
    def __init__(self):
        self.gamma = 0.9
        self.action_size = 4

        random_actions = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
        self.pi = defaultdict(lambda: random_actions)
        self.V = defaultdict(lambda: 0.0)
        self.cnts = defaultdict(lambda: 0.0)
        self.memory = []

    def get_action(self, state):
        action_probs = self.pi[state]
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        return np.random.choice(actions, p=probs)

    def add(self, state, action, reward):
        data = (state, action, reward)
        self.memory.append(data)

    def reset(self):
        self.memory.clear()

    def eval(self):
        G = 0
        for data in reversed(self.memory):
            state, _, reward = data
            G = reward + self.gamma * G
            self.cnts[state] += 1
            self.V[state] += (G - self.V[state]) / self.cnts[state]


agent = RandomAgent()

episodes = int(1e3)
for episode in tqdm(range(episodes), desc="Training Progress"):
    state = env.reset()
    agent.reset()

    while True:
        action = agent.get_action(state)
        next_state, reward, done = env.step(action)
        agent.add(state, action, reward)
        if done:
            agent.eval()
            break
        state = next_state

env.render_v(agent.V)
