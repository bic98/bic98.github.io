import random
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from paddle_env import PaddleEnv

# ----- 하이퍼 파라미터 -----
EPISODES = 1100
GAMMA = 0.99
N_STEP = 3
LR = 1e-3
BATCH_SIZE = 16
MEM_CAPACITY = 2000
EPS_START = 1.0
EPS_END = 0.1
EPS_DECAY = 0.995
TARGET_UPDATE = 5
HIDDEN_DIM = 128
BURN_IN = 5
PRIORITY_EPS = 1e-6
REWARD_SCALE = 1.0
IDLE_PENALTY = -0.05
DEATH_PENALTY = -2.0


class R2D2Net(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, HIDDEN_DIM)
        self.lstm = nn.LSTM(HIDDEN_DIM, HIDDEN_DIM, batch_first=True)
        self.out = nn.Linear(HIDDEN_DIM, out_dim)

    def forward(self, x, hidden=None):
        x = torch.relu(self.fc1(x))
        x, hidden = self.lstm(x, hidden)
        return self.out(x), hidden


class PrioritizedReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.priorities = []
        self.position = 0

    def push(self, episode, priority=1.0):
        if len(self.buffer) < self.capacity:
            self.buffer.append(episode)
            self.priorities.append(priority)
        else:
            self.buffer[self.position] = episode
            self.priorities[self.position] = priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        probs = np.array(self.priorities) + PRIORITY_EPS
        probs = probs / probs.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[i] for i in indices]
        return samples

    def __len__(self):
        return len(self.buffer)


if __name__ == "__main__":
    env = PaddleEnv()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    policy_net = R2D2Net(env.state_dim, env.action_space).to(device)
    target_net = R2D2Net(env.state_dim, env.action_space).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    memory = PrioritizedReplayBuffer(MEM_CAPACITY)

    eps = EPS_START
    rewards_log = []

    def choose_action(state, hidden, eps):
        if random.random() < eps:
            return random.randrange(env.action_space), hidden
        with torch.no_grad():
            state = torch.tensor(np.array(state), dtype=torch.float32).unsqueeze(
                0).unsqueeze(0).to(device)
            q, hidden = policy_net(state, hidden)
            return q[0, -1].argmax().item(), hidden

    def compute_nstep_targets(rewards, q_ns, dones):
        T = rewards.size(1)
        target = torch.zeros_like(rewards)
        for t in range(T):
            G = torch.zeros_like(rewards[:, 0])
            for n in range(N_STEP):
                idx = t + n
                if idx >= T:
                    break
                G += (GAMMA ** n) * rewards[:, idx]
                if dones[:, idx].bool().any():
                    break
            if t + N_STEP < T:
                G += (GAMMA ** N_STEP) * q_ns[:, t + N_STEP]
            target[:, t] = G
        return target

    def train_step():
        if len(memory) < BATCH_SIZE:
            return
        batch = memory.sample(BATCH_SIZE)
        for episode in batch:
            states, actions, rewards, next_states, dones = zip(*episode)
            T = len(states)
            if T <= BURN_IN + N_STEP:
                continue

            s = torch.tensor(np.array(states),
                             dtype=torch.float32).unsqueeze(0).to(device)
            ns = torch.tensor(np.array(next_states),
                              dtype=torch.float32).unsqueeze(0).to(device)
            a = torch.tensor(np.array(actions), dtype=torch.long).unsqueeze(
                0).unsqueeze(-1).to(device)
            r = torch.tensor(np.array(rewards),
                             dtype=torch.float32).unsqueeze(0).to(device)
            d = torch.tensor(
                np.array(dones), dtype=torch.float32).unsqueeze(0).to(device)

            hidden = None
            _, hidden = policy_net(s[:, :BURN_IN].contiguous(), hidden)

            q_s, _ = policy_net(s[:, BURN_IN:].contiguous(), hidden)
            q_s = q_s.gather(2, a[:, BURN_IN:]).squeeze(2)

            with torch.no_grad():
                q_ns_policy, _ = policy_net(ns[:, BURN_IN:].contiguous())
                best_actions = q_ns_policy.argmax(2, keepdim=True)
                q_ns_target, _ = target_net(ns[:, BURN_IN:].contiguous())
                q_ns = q_ns_target.gather(2, best_actions).squeeze(2)

                target = compute_nstep_targets(
                    r[:, BURN_IN:], q_ns, d[:, BURN_IN:])

            loss = nn.MSELoss()(q_s, target)
            td_error = (q_s - target).abs().mean().item()
            memory.push(episode, priority=td_error)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    for ep in range(1, EPISODES + 1):
        state = env.reset()
        total_r = 0
        episode_buffer = []
        hidden = None

        while True:
            action, hidden = choose_action(state, hidden, eps)
            next_state, reward, done, _ = env.step(action)
            if action == 1:
                reward += IDLE_PENALTY
            if done and reward < 0:
                reward = DEATH_PENALTY
            reward *= REWARD_SCALE

            episode_buffer.append(
                (state, action, reward, next_state, float(done)))
            state = next_state
            total_r += reward
            if done:
                break

        memory.push(episode_buffer)
        train_step()

        eps = max(EPS_END, eps * EPS_DECAY)
        rewards_log.append(total_r)

        if ep % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        print(
            f"EP {ep:4d} | Score {env.score:3d} | TotalR {total_r:5.1f} | ε {eps:.3f}")

    torch.save(policy_net.state_dict(), "paddle_r2d2_nstep.pth")
    print("모델 저장 완료: paddle_r2d2_nstep.pth")
