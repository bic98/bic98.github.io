---
# Page settings
layout: default
keywords: 강화학습, 머신러닝, 인공지능, 인찬백, InchanBaek, 리워드, 에이전트, 액션, MDP, 마르코프 결정 과정, Q-러닝, reinforcement learning, machine learning, AI, reward, agent, action, Markov decision process, Q-learning, deep reinforcement learning
comments: true
seo:
  title: Reinforcement Learning from Scratch - Complete Guide | InchanBaek Note
  description: 강화학습의 기초부터 고급 알고리즘까지 배우는 완벽 가이드. 마르코프 결정 과정, Q-러닝, 정책 경사법 등 핵심 개념과 실제 구현 방법을 단계별로 설명합니다.
  canonical: https://bic98.github.io/reinforce/
  image: https://bic98.github.io/images/layout/logo.png

# Hero section
title: Reinforcement Learning from Scratch
description: 강화학습의 기본 개념부터 고급 알고리즘까지, 이론과 실습을 통해 처음부터 차근차근 배우는 강화학습 기초 가이드입니다.

# # Author box
# author:
#     title: About Author
#     title_url: '#'
#     external_url: true
#     description: Author description

# Micro navigation
micro_nav: true

# Page navigation
# page_nav:
#     prev:
#         content: Previous page
#         url: '/deep_learning/'
#     next:
#         content: Next page
#         url: '/qgis/'


# Language setting
---


<script type="text/javascript" async
  src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

## 밴디트 문제 (Bandit Problem)

### 강화학습

- 지도학습 (Supervised Learning) : 입력과 출력 데이터가 주어졌을 때, 입력 데이터와 출력 데이터 사이의 관계를 모델링하는 기계학습의 한 방법

- 비지도학습 (Unsupervised Learning) : 입력 데이터만 주어졌을 때, 입력 데이터의 특징을 찾아내는 기계학습의 한 방법

- 강화학습 (Reinforcement Learning) : 에이전트가 환경과 상호작용하며, 환경에 대한 정보를 받아 보상을 최대화하는 행동을 선택하는 방법

### 밴디트 문제란?

밴디트 == 슬롯머신

슬롯머신은 각각의 확률이 다름. 

처음에는 어떤 슬롯머신이 가장 좋은지 모름. 

실제로 플레이를 해보면서 좋은 머신을 찾아야 함. 

정해진 횟수 안에 최대한 많이 보상을 얻는 것이 목표.

<div align="center">
  <div class = 'mermaid'>
    graph LR
    A[Agent] -->|행동| B[Environment]
    B -->|보상| A
  </div>
</div>


**플레이어인 에이전트는 주어진 환경에서 행동을 선택하고, 환경은 에이전트에게 보상을 제공한다.**

목표 : **보상을 최대화하는 행동을 선택하는 것** -> **코인을 최대한 많이 얻는 것** -> **좋은 슬롯머신을 찾는 것**

### 가치와 행동가치

- 가치 (Value) : 특정 상태에서 얻을 수 있는 보상의 기대값

$$
E[R_t] 
$$

- 행동가치 (Action Value) :행동의 결과로 얻은 보상의 기대값

$$
Q(A) = E[R_t | A] 
$$

(E = Expectation, Q = Quality, A = Action, R = Reward)

슬롯 머신 a와 b의 보상의 기댓값을 구해보자. 

밑에는 슬롯머신 a에 대한 표이다. 

| 슬롯머신 a | 
|:---:|:---:|:---:|:---:|:---:|:---:|
| 얻을 수 있는 코인| 0 | 1 | 5 | 10 |
| 보상 | 0.70 | 0.15 | 0.12 | 0.03 |


슬롯 머신 b에 대한 표이다.

| 슬롯머신 b | 
|:---:|:---:|:---:|:---:|:---:|:---:|
| 얻을 수 있는 코인| 0 | 1 | 5 | 10 |
| 보상 | 0.50 | 0.40 | 0.09 | 0.01 |

두 머신에 대한 기댓값은

- 슬롯머신 a : (0.7 * 0 + 0.15 * 1 + 0.12 * 5 + 0.03 * 10) = 1.05
- 슬롯머신 b : (0.5 * 0 + 0.4 * 1 + 0.09 * 5 + 0.01 * 10) = 0.95

슬롯머신 a가 슬롯머신 b보다 좋다.

### 가치 추정

n번의 플레이를 하면서 얻은 보상을 R1, R2, ..., Rn이라고 하자.
그때 행동 가치의 추정치 Qn은 다음과 같이 계산할 수 있다.

$$
Q_n = \frac{R_1 + R_2 + ... + R_n}{n}
$$

하지만 만약 이렇게 n번 플레이 하면서 가치 추정을 한다면 계산량과 메모리 부하가 커진다.
n - 1번째의 가치 추정치를 이용해서 n번째 가치 추정치를 계산할 수 있다.

$$
Q_{n-1} = \frac{R_1 + R_2 + ... + R_{n-1}}{n-1}
$$

이식의 양변에 (n - 1)을 곱하면

$$
(n - 1)Q_{n-1} = R_1 + R_2 + ... + R_{n-1}
$$

이제 n번째 가치 추정치를 계산할 수 있다.

$$
Q_n = \frac{1}{n} (R_1 + R_2 + ... + R_{n-1} + R_n) 
$$

$$
=\frac{1}{n} (n - 1)Q_{n-1} + \frac{1}{n} R_n
$$


$$
= Q_{n - 1} + \frac{1}{n} (R_n - Q_{n - 1})
$$

### 플레이어의 정책

확실하지 않은 추정치를 전적으로 신뢰하면 최선의 행동을 놓칠 수 있음. 따라서 에이전트는 불확실성을 줄여 추정의 신뢰도를 높여야 한다. 

- 정책 (Policy) : 에이전트가 환경과 상호작용할 때, 에이전트가 선택하는 행동을 결정하는 전략

불확실성을 줄이기 위해서는 두가지 정책을 사용할 수 있다. 

1. **탐험 (Exploration)** : 불확실한 행동을 선택하여 환경에 대한 정보를 얻는 것
2. **활용 (Exploitation)** : 현재까지의 정보를 활용하여 최선의 행동을 선택하는 것

결국 강화학습알고리즘은 '활용과 탐험의 균형'을 맞추는 것!!!!

### 엡실론-그리디 정책
탐험과 활용의 균형을 맞추기 위한 방법의 알고리즘 중 하나이다. 
예를 들어, $$\epsilon$$ = 0.1로 설정하면 10%의 확률로 무작위 행동을 선택하고, 90%의 확률로 가장 좋은 행동을 선택한다.

### 밴디트 문제의 해결

- **행동가치 추정** : 행동가치를 추정하고, 가장 좋은 행동을 선택한다.
- **정책** : 엡실론-그리디 정책을 사용하여 탐험과 활용의 균형을 맞춘다.

그럼 위의 내용을 코드로 구현해보자.

```python

import numpy as np

class Bandit:
    def __init__(self, arms = 10):
        self.rates = [0.38991635, 0.5937864,  0.55356798, 0.46228943, 0.48251845, 0.47595196, 0.53560295, 0.43374032, 0.55913105, 0.57484477]

    def play(self, arm):
        rate = self.rates[arm]
        if rate > np.random.rand():
            return 1
        else : 
            return 0


class Agent:
    def __init__(self, epslion, action_size = 10):
        self.epslion = epslion
        self.Qs = np.zeros(action_size)
        self.Ns = np.zeros(action_size)

    def update(self, action, reward):
        self.Ns[action] += 1
        self.Qs[action] += (reward - self.Qs[action]) / self.Ns[action]

    def get_action(self):
        if np.random.rand() < self.epslion:
            return (np.random.randint(len(self.Qs)), 0)
        return (np.argmax(self.Qs), 1)

steps = 10000
agent = Agent(0.1)
bandit = Bandit()

total_reward = 0
total_rewards = []
rates = []
actions = []

for i in range(steps):
    act = agent.get_action()
    action = act[0]
    reward = bandit.play(action)
    agent.update(action, reward)
    total_reward += reward
    total_rewards.append(total_reward)
    rates.append(total_reward / (i + 1))
    if act[1] == 1:
        actions.append(action)

import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))

plt.subplot(3, 1, 1)
plt.plot(total_rewards, label='Total Reward')
plt.xlabel('Steps')
plt.ylabel('Total Reward')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(actions, label='Actions')
plt.xlabel('Steps')
plt.ylabel('Action')
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(rates, label='Average Reward')
plt.xlabel('Steps')
plt.ylabel('Average Reward')
plt.legend()

plt.tight_layout()
plt.show()

```

<div align="center">
  <img src="/images/bandit.png" alt="bandit" width="100%">
</div>

약 10000번의 플레이를 하면 인덱스 1번의 슬롯머신을 액션으로 선택하는 것이 최선임을 아직을 알지 못한다. 
더 많은 스텝을 주어보자. 

<div align="center">
  <img src="/images/bandit2.png" alt="bandit" width="100%">
</div>

약 30000번의 플레이를 하면 인덱스 1번의 슬롯머신을 액션으로 선택하는 것이 최선임을 알게 된다.
약 2%의 확률 차이를 인지하기 위해 20000번의 플레이가 더 필요했다. 


### 비정상 문제 (Non-stationary Problem)

지금까지 다룬 밴디트 문제를 정상문제에 속한다. 정상문제란 보상의 확률 분포가 변하지 않는 문제이다. 위의 코드를 보면 rates라는 변수에 확률이 고정되어 있다.

하지만 실제로는 보상의 확률 분포가 변하는 경우가 많다. 이런 경우를 비정상 문제라고 한다. 이런 경우에는 어떻게 해야할까?



먼저 정상 문제에서는 다음과 같은 식으로 행동가치 추정치를 업데이트 했다.

$$
Q_n = Q_{n - 1} + \frac{1}{n} (R_n - Q_{n - 1})
$$

하지만 비정상 문제에서는 다음과 같은 식으로 행동가치 추정치를 업데이트 한다.

$$
Q_n = Q_{n - 1} + \alpha (R_n - Q_{n - 1})
$$

오래전에 얻은 보상에 대한 가중치를 줄이고, 최근에 얻은 보상에 대한 가중치를 높이는 방법이다. 이때 $$\alpha$$는 학습률이라고 한다.


<div style="overflow-x: auto;">

$$
= Q_{n - 1} + \alpha (R_n - Q_{n - 1})
$$
$$
= (1 - \alpha) Q_{n - 1} + \alpha R_n
$$
$$
= \alpha R_n + (1 - \alpha) {(\alpha R_{n - 1} + (1 - \alpha) Q_{n - 2})}
$$
$$
= \alpha R_n + (1 - \alpha) \alpha R_{n - 1} + (1 - \alpha)^2 Q_{n - 2}
$$
$$
= \alpha R_n + (1 - \alpha) \alpha R_{n - 1} + (1 - \alpha)^2 \alpha R_{n - 2} + (1 - \alpha)^3 Q_{n - 3}
$$
$$
= \alpha R_n + (1 - \alpha) \alpha R_{n - 1} + (1 - \alpha)^2 \alpha R_{n - 2} +
$$
$$
(1 - \alpha)^3 \alpha R_{n - 3} + ... + (1 - \alpha)^{n - 1} \alpha R_1 + (1 - \alpha)^n Q_0
$$

</div>

$$Q_0$$는 초기값이다. 우리가 설정한 값에 따라서  학습결과에 편향이 생긴다. 하지만 표본 평균을 사용하면 편향이 사라진다.


위의 방식을 지수이동평균, 지수가중이동평균이라고 한다.


- **지수 가중 이동 평균 (Exponential Weighted Moving Average)** : 최근에 얻은 보상에 더 많은 가중치를 주고, 오래전에 얻은 보상에는 적은 가중치를 주는 방법


python 코드로 구현해보자.

```python
import numpy as np

class Bandit:
    def __init__(self, arms = 10):
        self.rates = [0.38991635, 0.7837864,  0.55356798, 0.46228943, 0.48251845, 0.47595196, 0.53560295, 0.43374032, 0.55913105, 0.57484477]

    def play(self, arm):
        rate = self.rates[arm]
        self.rates += 0.1 * np.random.randn(len(self.rates))
        if rate > np.random.rand():
            return 1
        else : 
            return 0

class Agent:
    def __init__(self, epslion, action_size = 10):
        self.epslion = epslion
        self.Qs = np.zeros(action_size)

    def update(self, action, reward, alpha = 0.8):
        self.Qs[action] += alpha * (reward - self.Qs[action])

    def get_action(self):
        if np.random.rand() < self.epslion:
            return (np.random.randint(len(self.Qs)), 0)
        return (np.argmax(self.Qs), 1) 

steps = 50000
agent = Agent(0.1)
bandit = Bandit()

total_reward = 0
total_rewards = []
rates = []
actions = []

for i in range(steps):
    act = agent.get_action()
    action = act[0]
    reward = bandit.play(action)
    agent.update(action, reward)
    total_reward += reward
    total_rewards.append(total_reward)
    rates.append(total_reward / (i + 1))
    if act[1] == 1:
        actions.append(action)

import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))

plt.subplot(3, 1, 1)
plt.plot(total_rewards, label='Total Reward')
plt.xlabel('Steps')
plt.ylabel('Total Reward')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(actions, label='Actions')
plt.xlabel('Steps')
plt.ylabel('Action')
plt.ylim(0, 9)
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(rates, label='Average Reward')
plt.xlabel('Steps')
plt.ylabel('Average Reward')
plt.ylim(0, 1)
plt.legend()

plt.tight_layout()
plt.show()
```

<div align="center">
  <img src="/images/bandit3.png" alt="bandit" width="100%">
</div>

고정값 $$\alpha$$ = 0.8로 설정하면 표본 평균을 사용한 경우보다 더 결과가 빨리 수렴하는 것을 볼 수 있다.

### 정리

- **밴디트 문제** : 강화학습의 기초 문제로, 여러 개의 슬롯머신 중에서 최대 보상을 얻는 방법을 찾는 문제
- **행동가치** : 행동의 결과로 얻은 보상의 기대값
- **정책** : 에이전트가 환경과 상호작용할 때, 에이전트가 선택하는 행동을 결정하는 전략
- **엡실론-그리디 정책** : 탐험과 활용의 균형을 맞추기 위한 방법의 알고리즘 중 하나
- **비정상 문제** : 보상의 확률 분포가 변하는 문제
- **지수 가중 이동 평균** : 최근에 얻은 보상에 더 많은 가중치를 주고, 오래전에 얻은 보상에는 적은 가중치를 주는 방법

## 마르코프 결정 과정 (Markov Decision Process)

에리전트의 행동에 따라 환경의 상태가 변하는 문제를 다루어보자. 

### 마르코프 결정 과정이란?

- **마르코프 결정 과정 (Markov Decision Process, MDP)** : 에이전트가 환경과 상호작용하며, 환경의 상태가 마르코프 성질을 만족하는 환경을 모델링하는 방법

- **마르코프 성질** : 미래의 상태가 현재의 상태에만 의존하는 성질

MDP에는 시간 개념이 필요하다. 특정 시간에 에이전트가 행동을 취하고, 그 결과 새로우 상태로 전이한다. 이때의 시간단위를 time step이라고 한다.

<div align="center">
  <div class = 'mermaid'>
    graph LR
    A[Agent] -->|행동| B[Environment]
    B -->|보상, 상태| A
  </div>
</div>

- **상태전이** : 상태는 어떻게 전이되는가?
- **보상** : 보상은 어떻게 주어지는가?
- **정책** : 에이전트는 행동을 어떻게 결정하는가?

위의 세가지 요소를 수식으로 표현해야한다. 

상태전이가 결정적일 경우 다음 상태 s'는 현재 상태 s와 행동 a에만 의존한다.

상태전이함수 => 
$$
s' = f(s, a)
$$

상태전이가 확률적일 경우 다음 상태 s'는 현재 상태 s와 행동 a에만 의존한다.

상태전이확률 =>
$$
P(s' | s, a)
$$

### 보상함수

보상함수는 상태 s와 행동 a에 대한 보상을 반환한다.에이전트가 상태 s에서 행동 a를 취하여 다음 상태 s'로 이동했을 때 받는 보상을 반환한다.

보상함수 =>
$$
r(s, a, s')
$$


### 에이전트의 정책

에이전트의 정책은 에이전트가 행동을 결정하는 방식을 말한다. 에이전트는 '현재 상태' 만드오 행동을 결정한다.
왜냐하면 '환경에 대해 필요한 정보는 모두 현재 상태에 담겨' 있기 때문이다.

에이전트가 확률적으로 결정되는 정책을 다음과 같이 표현할 수 있다.

정책 =>
$$
\pi(a | s) = P(a | s)
$$
    
### MDP의 목표

MDP의 목표는 보상을 최대화하는 정책을 찾는 것이다. 에이전는 정책 $$ \pi(a | s) $$ 에 따라 행동한다. 그 행동과 상태 전이 확률 $$ P(s' | s, a) $$ 에 따라 다음 상태가 결정된다. 그리고 보상함수 $$ r(s, a, s') $$ 에 따라 보상을 받는다.
