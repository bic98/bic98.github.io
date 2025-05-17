# train_dqn.py
import random
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from paddle_env import PaddleEnv

# ----- 하이퍼 파라미터 -----
EPISODES = 1500
GAMMA = 0.99
LR = 1e-3
BATCH_SIZE = 64
MEM_CAPACITY = 20000
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 0.995
TARGET_UPDATE = 10

# ----- Q-네트워크 정의 -----


class QNet(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(),
            nn.Linear(128, 64),     nn.ReLU(),
            nn.Linear(64, out_dim)
        )

    def forward(self, x): return self.net(x)


# ----- 리플레이 버퍼 -----
Transition = collections.namedtuple(
    'Transition', ['s', 'a', 'r', 'ns', 'done'])


class ReplayBuffer:
    def __init__(self, cap): self.buffer = collections.deque(maxlen=cap)
    def push(self, *args):   self.buffer.append(Transition(*args))
    def sample(self, bsize): return random.sample(self.buffer, bsize)
    def __len__(self): return len(self.buffer)


# ----- 학습 루프 시작 -----
if __name__ == "__main__":
    env = PaddleEnv()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    policy_net = QNet(env.state_dim, env.action_space).to(device)
    target_net = QNet(env.state_dim, env.action_space).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    memory = ReplayBuffer(MEM_CAPACITY)

    eps = EPS_START
    rewards_log = []

    def choose_action(state, eps):
        if random.random() < eps:
            return random.randrange(env.action_space)
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            return policy_net(state).argmax(1).item()

    def train_step():
        if len(memory) < BATCH_SIZE:
            return
        batch = memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*batch))

        s = torch.FloatTensor(batch.s).to(device)
        a = torch.LongTensor(batch.a).unsqueeze(1).to(device)
        r = torch.FloatTensor(batch.r).to(device)
        ns = torch.FloatTensor(batch.ns).to(device)
        d = torch.FloatTensor(batch.done).to(device)

        q_sa = policy_net(s).gather(1, a).squeeze()
        # Double DQN
        with torch.no_grad():
            best_actions = policy_net(ns).argmax(
                1, keepdim=True)       # 선택: policy_net
            q_ns = target_net(ns).gather(
                1, best_actions).squeeze()     # 평가: target_net
            target = r + GAMMA * q_ns * (1 - d)
        # DQN
        # with torch.no_grad():
        #     q_ns = target_net(ns).max(1)[0]
        #     target = r + GAMMA * q_ns * (1-d)
        loss = nn.MSELoss()(q_sa, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    for ep in range(1, EPISODES+1):
        state = env.reset()
        total_r = 0
        while True:
            action = choose_action(state, eps)
            next_state, reward, done, _ = env.step(action)
            memory.push(state, action, reward, next_state, done)
            state = next_state
            total_r += reward
            train_step()
            if done:
                break

        eps = max(EPS_END, eps * EPS_DECAY)
        rewards_log.append(total_r)

        if ep % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        print(
            f"EP {ep:3d} | Score {env.score:3d} | TotalR {total_r:4.1f} | ε {eps:.2f}")

    # 모델 저장
    torch.save(policy_net.state_dict(), "paddle_dqn_model.pth")
    print("🎉 모델 저장 완료: paddle_dqn_model.pth")
