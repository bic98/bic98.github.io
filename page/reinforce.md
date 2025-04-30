---
# Page settings
layout: default
keywords: 강화학습, 머신러닝, 인공지능, 인찬백, InchanBaek, 리워드, 에이전트, 액션, MDP, 마르코프 결정 과정, Q-러닝, reinforcement learning, machine learning, AI, reward, agent, action, Markov decision process, Q-learning, deep reinforcement learning
comments: true
seo:
  title: Reinforcement Learning from Scratch - Complete Guide | InchanBaek Note
  description: A complete guide to learning reinforcement learning from basics to advanced algorithms. It explains key concepts such as Markov decision processes, Q-learning, and policy gradient methods, along with step-by-step implementation techniques.
  canonical: https://bic98.github.io/reinforce/
  image: https://bic98.github.io/images/layout/logo.png

# Hero section
title: Reinforcement Learning from Scratch
description: A complete guide to learning reinforcement learning from basics to advanced algorithms. It explains key concepts such as Markov decision processes, Q-learning, and policy gradient methods, along with step-by-step implementation techniques.

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

## Bandit Problem

### Reinforcement Learning

- Supervised Learning : When the input and output data are given, it is a method of modeling the relationship between input data and output data. 

- Unsupervised Learning : When the input data is given, it is a method of finding the characteristics of the input data. 


- **Reinforcement Learning** : **Agent** is an entity that interacts with the **environment** and receives information about the environment to choose **actions** that **maximize rewards**.

### What is Bandit problem?

Bandit == Slot machine

Each slot machine has a different probability.

At first, we don't know which slot machine is the best.

We need to find the good machine by actually playing.

The goal is to get as much reward as possible within a limited number of plays.

<div align="center">
  <div class = 'mermaid'>
    graph LR
    A[Agent] -->|action| B[Environment]
    B -->|reward| A
  </div>
</div>


**The agent as a player selects actions in a given environment, and the environment provides rewards to the agent.**

**Goal**: **Select actions that maximize rewards** -> **Get as many coins as possible** -> **Find the best slot machine**

### Value and Action Value

- **Value**: Expected reward that can be obtained in a specific state

$$
E[R_t] 
$$

- **Action Value**: Expected reward obtained as a result of an action

$$
Q(A) = E[R_t | A] 
$$

(E = Expectation, Q = Quality, A = Action, R = Reward)

Let's calculate the expected rewards for slot machines a and b.

Below is a table for slot machine a.

| Slot machine a | 
|:---:|:---:|:---:|:---:|:---:|:---:|
| Coins obtainable | 0 | 1 | 5 | 10 |
| Reward probability | 0.70 | 0.15 | 0.12 | 0.03 |


Here's a table for slot machine b.

| Slot machine b | 
|:---:|:---:|:---:|:---:|:---:|:---:|
| Coins obtainable | 0 | 1 | 5 | 10 |
| Reward probability | 0.50 | 0.40 | 0.09 | 0.01 |

The expected values for the two machines are:

- Slot machine a: (0.7 * 0 + 0.15 * 1 + 0.12 * 5 + 0.03 * 10) = 1.05
- Slot machine b: (0.5 * 0 + 0.4 * 1 + 0.09 * 5 + 0.01 * 10) = 0.95

**Slot machine a is better than slot machine b.**

### Value Estimation

Let's say the rewards obtained during n plays are R1, R2, ..., Rn.
Then the action value estimate Qn can be calculated as follows:

$$
Q_n = \frac{R_1 + R_2 + ... + R_n}{n}
$$

However, if we estimate the value this way after n plays, the computational and memory load becomes large.
We can calculate the nth value estimate using the (n-1)th value estimate.

$$
Q_{n-1} = \frac{R_1 + R_2 + ... + R_{n-1}}{n-1}
$$

If we multiply both sides of this equation by (n-1):

$$
(n - 1)Q_{n-1} = R_1 + R_2 + ... + R_{n-1}
$$

Now we can calculate the nth value estimate:

$$
Q_n = \frac{1}{n} (R_1 + R_2 + ... + R_{n-1} + R_n) 
$$

$$
=\frac{1}{n} (n - 1)Q_{n-1} + \frac{1}{n} R_n
$$


$$
= Q_{n - 1} + \frac{1}{n} (R_n - Q_{n - 1})
$$

### Player's Policy

If we completely trust uncertain estimates, we might miss the best action. Therefore, the agent needs to reduce uncertainty and increase the reliability of estimation.

- **Policy**: The strategy that determines the actions an agent selects when interacting with the environment

There are two policies that can be used to reduce uncertainty:

1. **Exploration**: Selecting uncertain actions to gain information about the environment
2. **Exploitation**: Selecting the best action based on information available so far

**Ultimately, reinforcement learning algorithms are about finding the right 'balance between exploitation and exploration'!!!!**

### Epsilon-Greedy Policy
This is one of the algorithms used to balance exploration and exploitation.
For example, if $$\epsilon$$ = 0.1, it selects a random action with 10% probability and selects the best action with 90% probability.

### Solving the Bandit Problem

- **Action Value Estimation**: Estimate the action value and select the best action.
- **Policy**: Use the **epsilon-greedy policy** to balance **exploration** and **exploitation**.

Let's implement the above content in code.

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

After about 10,000 plays, it still doesn't know that selecting the slot machine at index 1 as an action is optimal.
Let's try with more steps.

<div align="center">
  <img src="/images/bandit2.png" alt="bandit" width="100%">
</div>

After about 30,000 plays, it learns that selecting the slot machine at index 1 as an action is optimal.
It took an additional 20,000 plays to recognize a probability difference of about 2%.


### Non-stationary Problem

The bandit problem we've covered so far belongs to the category of **stationary problems**. A stationary problem is one where the probability distribution of rewards **does not change**. In the code above, you can see that the probabilities are fixed in the variable called rates.

However, in reality, the probability distribution of rewards often changes. This is called a **non-stationary problem**. How should we handle this?



First, in stationary problems, we updated the action value estimate with the following equation:

$$
Q_n = Q_{n - 1} + \frac{1}{n} (R_n - Q_{n - 1})
$$

But in **non-stationary problems**, we update the action value estimate with the following equation:

$$
Q_n = Q_{n - 1} + \alpha (R_n - Q_{n - 1})
$$

This method **reduces the weight of rewards obtained long ago** and **increases the weight of recently obtained rewards**. Here, $$\alpha$$ is called the **learning rate**.


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

$$Q_0$$ is the initial value. Depending on the value we set, bias can occur in the learning results. However, when using sample averages, the bias disappears.


This method is called **exponential moving average** or **exponentially weighted moving average**.


- **Exponential Weighted Moving Average**: A method that gives **more weight to recently obtained rewards** and **less weight to rewards obtained long ago**


Let's implement this in Python code.

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

When we set the fixed value $$\alpha$$ = 0.8, we can see that the results converge faster than when using sample averages.

### Summary

- **Bandit Problem**: A fundamental problem in reinforcement learning where the goal is to find a method that maximizes rewards among multiple slot machines
- **Action Value**: The expected reward obtained as a result of an action
- **Policy**: The strategy that determines the actions an agent selects when interacting with the environment
- **Epsilon-Greedy Policy**: One of the algorithms used to balance **exploration and exploitation**
- **Non-stationary Problem**: A problem where the probability distribution of rewards changes
- **Exponential Weighted Moving Average**: A method that gives **more weight to recently obtained rewards** and **less weight to rewards obtained long ago**

## Markov Decision Process

Let's examine problems where the state of the environment changes according to an agent's actions.

### What is a Markov Decision Process?

- **Markov Decision Process (MDP)**: A method of modeling an environment where the agent interacts with the environment, and the environment's state satisfies the Markov property

- **Markov Property**: The property where the **future state depends only on the current state**

MDPs require the concept of time. At a specific time, the agent takes an action, and as a result, transitions to a new state. The time unit in this case is called a time step.

<div align="center">
  <div class = 'mermaid'>
    graph LR
    A[Agent] -->|action| B[Environment]
    B -->|reward, state| A
  </div>
</div>

- **State Transition**: How does the state transition?
- **Reward**: How is the reward given?
- **Policy**: How does the agent determine its actions?

The above three elements must be expressed in formulas.

If the state transition is **deterministic**, the next state s' depends only on the current state s and action a.

**State transition function** => 
$$
s' = f(s, a)
$$

If the state transition is **probabilistic**, the next state s' depends only on the current state s and action a.

**State transition probability** =>
$$
P(s' | s, a)
$$

### Reward Function

The **reward function** returns the reward for state s and action a. It returns the reward received when the agent takes action a in state s and moves to the next state s'.

**Reward function** =>
$$
r(s, a, s')
$$


### Agent's Policy

The agent's **policy** refers to how the agent determines its actions. The agent determines its actions based solely on the **'current state'**.
This is because **'all the information needed about the environment is contained in the current state'**.

A policy that the agent decides probabilistically can be expressed as follows:

**Policy** =>
$$
\pi(a | s) = P(a | s)
$$
    
### Goal of MDP

The goal of MDP is to find a policy that maximizes rewards. The agent behaves according to the policy 
$$ 
\pi(a | s) 
$$
The next state is determined according to that action and the state transition probability $$ P(s' | s, a) $$. And the agent receives rewards according to the reward function $$ r(s, a, s') $$.

### Return

The state at time t is $$ S_t $$, according to the policy $$ \pi $$, the action is $$ A_t $$, the reward is $$ R_t $$, and this leads to a flow that transitions to the new state $$ S_{t+1} $$. The return at this time can be defined as follows:

$$
G_t = R_t + rR_{t+1} + r^2R_{t+2} + ... = \sum_{k=0}^{\infty} r^k R_{t+k}
$$

As time passes, the reward decreases exponentially due to $$ \gamma $$.

### State Value Function

The agent's goal is to maximize returns. Even if an agent starts in the same state, the returns can vary for each episode. To respond to such stochastic behavior, we use the expectation, i.e., the expected return, as an indicator.

The state value function is a function that represents the expected value of rewards that can be received in the future, starting from a specific state in reinforcement learning. It is generally represented as $$V(s)$$, where $$s$$ represents the state. The state value function is calculated according to policy $$\pi$$, and is defined by the following formula:

<div style="overflow-x: auto;">
$$
V_{\pi}(s) = \mathbb{E} \left[ \sum_{t=0}^\infty \gamma^t R_{t+1} \mid S_t = s, \pi \right]
$$
</div>

$$
= \mathbb{E}_\pi \left[G_t \mid S_t = s \right]
$$

Where:
- $$\mathbb{E}_\pi$$: Expected value according to policy $$\pi$$
- $$\gamma$$: Discount rate (0 ≤ $$\gamma$$ < 1)
- $$R_{t+1}$$: Reward at time $$t+1$$
- $$S_0 = s$$: Initial state

In other words, the state value function is used to predict the total rewards that will be received in the long term, starting from a specific state, when following a given policy. This plays an important role in evaluating the quality of a policy or finding the optimal policy.

### Optimal Policy and Optimal Value Function


In reinforcement learning, the optimal policy $$\pi^*$$ is a policy that maximizes the expected reward in all states. If the agent follows the optimal policy, it can obtain the maximum possible reward.

The optimal value function $$V^*(s)$$ is the sum of expected rewards that can be obtained when starting from state $$s$$ and following the optimal policy:

<div style="overflow-x: auto;">
$$
V^*(s) = \max_{\pi} V^{\pi}(s) = \max_{\pi} \mathbb{E}_{\pi} \left[ \sum_{t=0}^{\infty} \gamma^t R_{t+1} \mid S_0 = s \right]
$$
</div>

Similarly, the optimal action-value function $$Q^*(s,a)$$ is the sum of expected rewards that can be obtained when taking action $$a$$ in state $$s$$ and thereafter following the optimal policy:

<div style="overflow-x: auto;">
$$
Q^*(s,a) = \max_{\pi} Q^{\pi}(s,a) = \max_{\pi} \mathbb{E}_{\pi} \left[ \sum_{t=0}^{\infty} \gamma^t R_{t+1} \mid S_0 = s, A_0 = a \right]
$$
</div>

The optimal policy and optimal value function can be defined through the Bellman Optimality Equation:


<div style="overflow-x: auto;">
$$
V^*(s) = \max_{a} \left[ R(s,a) + \gamma \sum_{s'} P(s'|s,a) V^*(s') \right]
$$
</div>

<div style="overflow-x: auto;">
$$
Q^*(s,a) = R(s,a) + \gamma \sum_{s'} P(s'|s,a) \max_{a'} Q^*(s',a')
$$
</div>

The goal of reinforcement learning is to find such an optimal policy or optimal value function.

## Bellman equation

First, Summary of the Above. 

- **❓ What is an MDP?**

An MDP is a mathematical framework used to model decision-making in environments where outcomes are partly random and partly under the control of an agent. 

It consists of: 

- A set of **states (S)**
- A set of **actions (A)**
- A **transition probability function (P)**
- A **reward function (R)**
- A **discount factor (γ)**

So, MDP is the **foundation of reinforcement learning**, where an agent learns to choose actions that maximize cumulative reward over time. 


- **❓Why is important Bellman equation in MDP?**

The **Bellman equation** is important in Markov Decision Processes (MDPs) because it provides a **recursive decomposition of the value function**, which represents the expected return starting from a given state. It serves as the **foundation for many reinforcement learning algorithms**, enabling **efficient computation of optimal policies** by breaking down complex problems into smaller subproblems.

🔑 **Bellman Equation – Easy Explanation (with Keywords)**
- **The Bellman equation expresses**
"**What kind of future reward can I expect if I act well in this state?**"

- **It uses recursion to break down a complex problem into smaller subproblems.**

- **This allows us to efficiently and systematically optimize the overall policy.**

- **Many reinforcement learning algorithms like Q-learning and Value Iteration**
are based on the Bellman equation.

### Derivation of Bellman Equation. 

First, let's define 'Return at time t' as the sum of rewards from time 't'

<div style="overflow-x: auto;">
$$
G_t = R_t + \gamma R_{t+1} + \gamma^2 R_{t+2} + ... = \sum_{k=0}^{\infty} \gamma^k R_{t+k}
$$
</div>

Second, What is 'Return at time t + 1'?
<div style="overflow-x: auto;">
$$
G_{t+1} = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + ... = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}
$$
</div>

So, we can rearrange the equation as above two equatios. 

<div style="overflow-x: auto;">
$$
G_t = R_t + \gamma G_{t+1}
$$
</div>

We know the relation between $$G_t$$ and $$G_{t+1}$$.  

Based on the state-vlaue function $$V_\pi(s)$$ we obtained earlier, we can derived the following conclusion. 

<div style="overflow-x: auto;">
$$
V_\pi(s) = \mathbb{E}_\pi[G_t | S_t = s]
    = \mathbb{E}_\pi[R_t + \gamma G_{t+1} | S_t = s]
    = \mathbb{E}_\pi[R_t | S_t = s] + \gamma \mathbb{E}_\pi[G_{t+1} | S_t = s]
$$
</div>

(Since Linearity of Expectation 👉 $$\mathbb{E}[X + Y] = \mathbb{E}[X] + \mathbb{E}[Y]$$)


<div style="text-align: center;">
    <div class="mermaid">
    graph TD
        s((s)) --> A((A))
        s((s)) --> B((B))
        s((s)) --> C((C))
        A --> A1((A1))
        A --> A2((A2))
        B --> B1((B1))
        B --> B2((B2))
        C --> C1((C1))
        C --> C2((C2))
        style s fill:#ffffff,stroke:#000000,stroke-width:2px
        style A fill:#ffffff,stroke:#000000,stroke-width:2px
        style B fill:#ffffff,stroke:#000000,stroke-width:2px
        style C fill:#ffffff,stroke:#000000,stroke-width:2px
        style A1 fill:#ffffff,stroke:#000000,stroke-width:2px
        style A2 fill:#ffffff,stroke:#000000,stroke-width:2px
        style B1 fill:#ffffff,stroke:#000000,stroke-width:2px
        style B2 fill:#ffffff,stroke:#000000,stroke-width:2px
        style C1 fill:#ffffff,stroke:#000000,stroke-width:2px
        style C2 fill:#ffffff,stroke:#000000,stroke-width:2px
    </div>
</div>


we define $$
\pi(a | s) 
$$ as the probability of taking action $$a$$ in state $$s$$.



<div style="text-align: center;">
    <div class="mermaid">
    graph TD
        s((s)) --> A((A))
        s((s)) --> B((B))
        s((s)) --> C((C))
        style s fill:#ffffff,stroke:#000000,stroke-width:2px
        style A fill:#ffffff,stroke:#000000,stroke-width:2px
        style B fill:#ffffff,stroke:#000000,stroke-width:2px
        style C fill:#ffffff,stroke:#000000,stroke-width:2px
    </div>
</div>

so 
$$
\pi(a_1 | s) = A
$$, 
$$
\pi(a_2 | s) = B
$$, 
$$
\pi(a_3 | s) = C
$$

and we choose the action along with the policy $$\pi$$. we move $$s$$ to $$s'$$ with the probability 
$$
P(s' | s, a)
$$. (P is the transition probability function)


<div style="text-align: center;">
    <div class="mermaid">
    graph TD
        A((A)) --> A1((A1))
        A --> A2((A2))
        B((B)) --> B1((B1))
        B --> B2((B2))
        C((C)) --> C1((C1))
        C --> C2((C2))
        style A fill:#ffffff,stroke:#000000,stroke-width:2px
        style B fill:#ffffff,stroke:#000000,stroke-width:2px
        style C fill:#ffffff,stroke:#000000,stroke-width:2px
        style A1 fill:#ffffff,stroke:#000000,stroke-width:2px
        style A2 fill:#ffffff,stroke:#000000,stroke-width:2px
        style B1 fill:#ffffff,stroke:#000000,stroke-width:2px
        style B2 fill:#ffffff,stroke:#000000,stroke-width:2px
        style C1 fill:#ffffff,stroke:#000000,stroke-width:2px
        style C2 fill:#ffffff,stroke:#000000,stroke-width:2px
    </div>
</div>

According to above graph, 

$$
A_1 = P(s' | s, a_1) * \pi(a_1 | s)
$$

$$
A_2 = P(s' | s, a_2) * \pi(a_2 | s)
$$

$$
B_1 = P(s' | s, a_1) * \pi(a_1 | s)
$$

$$
B_2 = P(s' | s, a_2) * \pi(a_2 | s)
$$

$$
C_1 = P(s' | s, a_1) * \pi(a_1 | s)
$$

$$
C_2 = P(s' | s, a_2) * \pi(a_2 | s)
$$

Let's generalize the above equation.

<div style="overflow-x: auto;">

$$
\mathbb{E}_\pi[R_t | S_t = s] = \sum_{a} \pi(a | s) \sum_{s'} P(s' | s, a) R(s, a, s')
$$

</div>


<div style="overflow-x: auto;">

$$
V_\pi(s) = \mathbb{E}_\pi[R_t | S_t = s] + \gamma \mathbb{E}_\pi[G_{t+1} | S_t = s]
$$


$$
= \sum_{a} \pi(a | s) \sum_{s'} P(s' | s, a) R(s, a, s') + \gamma \sum_{a} \pi(a | s) \sum_{s'} P(s' | s, a) V_\pi(s')
$$

$$
= \sum_{a} \pi(a | s) \sum_{s'} P(s' | s, a) \left[ R(s, a, s') + \gamma V_\pi(s') \right]
$$

</div>

This is the **Bellman euqation** for the state value function. 


### State Value Function and Action Value Function(Q-function)

The state value function $$V_\pi(s)$$ is the expected return starting from state $$s$$ and following policy $$\pi$$. It can be expressed as:

<div style="overflow-x: auto;">
$$
V_\pi(s) = \sum_{a} \pi(a | s) \sum_{s'} P(s' | s, a) \left[ R(s, a, s') + \gamma V_\pi(s') \right]
= \mathbb{E}_\pi \left[ G_t | S_t = s \right]
$$
</div>

The Q-function represents the expected return when taking action a in state s at time t, and thereafter following policy π.


<div style="overflow-x: auto;">
$$
q_\pi(s, a) = \mathbb{E}_\pi[G_t | S_t = s, A_t = a]
$$
</div>

In short, 

- policy 
$$\pi$$
determines how to act in a given state $$s$$

- Value function $$V_\pi(s)$$ evalutes how good it is to be in a specific state under policy $$\pi$$

- Action value function(Q-function) $$q_\pi(s, a)$$ evalutes how good it is to take a specific action in a given state under policy $$\pi$$

<div style="overflow-x: auto;">
$$
q_\pi(s, a) = \sum_{s'} P(s' | s, a) \left[ R(s, a, s') + \gamma V_\pi(s') \right] = \sum_{s'} P(s' | s, a) \left[ R(s, a, s') + \gamma \sum_{a'} \pi(a' | s') q_\pi(s', a') \right]
$$
</div>


### optimal Action Value Function

The optimal action value function $$q^*(s, a)$$ is the maximum expected return when taking action $$a$$ in state $$s$$ and thereafter following the optimal policy:

<div style="overflow-x: auto;">
$$
q^*(s, a) = \max_{\pi} q_\pi(s, a) = \mathbb{E}_{\pi} \left[ G_t | S_t = s, A_t = a \right] = \sum_{s'} P(s' | s, a) \left[ R(s, a, s') + \gamma \max_{a'} q^*(s', a') \right]
$$
</div>

### optimal Policy

We assume that the optimal action value function $$q^*(s, a)$$ is known. Then the optimal policy at state $$s$$ is defined as follows. 

<div style="overflow-x: auto;">
$$
\mu^*(s) = \arg \max_a q^*(s, a)
$$
</div>


## Dynamic Programming

Dynamic programming is a method used to solve complex problems by breaking them down into simpler subproblems. It is particularly useful in reinforcement learning for solving Markov Decision Processes (MDPs).

### 3x4 grid world

```
+---+---+---+---+
| S |   |   | G |
+---+---+---+---+
|   | # |   | B |
+---+---+---+---+
|   |   |   |   |
+---+---+---+---+
```

- `S`: Start position
- `G`: Goal position
- `#`: Obstacle or blocked cell
- `B`: Bomb location with a reward of `-1.0`
- Blank cells are navigable spaces.

```python
import numpy as np
import common.gridworld_render as render_helper


class GridWorld:
    def __init__(self):
        self.action_space = [0, 1, 2, 3]  # 행동 공간(가능한 행동들)
        self.action_meaning = {  # 행동의 의미
            0: "UP",
            1: "DOWN",
            2: "LEFT",
            3: "RIGHT",
        }

        self.reward_map = np.array(  # 보상 맵(각 좌표의 보상 값)
            [[0, 0, 0, 1.0],
             [0, None, 0, -1.0],
             [0, 0, 0, 0]]
        )
        self.goal_state = (0, 3)    # 목표 상태(좌표)
        self.wall_state = (1, 1)    # 벽 상태(좌표)
        self.start_state = (2, 0)   # 시작 상태(좌표)
        self.agent_state = self.start_state   # 에이전트 초기 상태(좌표)

    @property
    def height(self):
        return len(self.reward_map)

    @property
    def width(self):
        return len(self.reward_map[0])

    @property
    def shape(self):
        return self.reward_map.shape

    def actions(self):
        return self.action_space

    def states(self):
        for h in range(self.height):
            for w in range(self.width):
                yield (h, w)

    def next_state(self, state, action):
        # 이동 위치 계산
        action_move_map = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        move = action_move_map[action]
        next_state = (state[0] + move[0], state[1] + move[1])
        ny, nx = next_state

        # 이동한 위치가 그리드 월드의 테두리 밖이나 벽인가?
        if nx < 0 or nx >= self.width or ny < 0 or ny >= self.height:
            next_state = state
        elif next_state == self.wall_state:
            next_state = state

        return next_state  # 다음 상태 반환

    def reward(self, state, action, next_state):
        return self.reward_map[next_state]

    def reset(self):
        self.agent_state = self.start_state
        return self.agent_state

    def step(self, action):
        state = self.agent_state
        next_state = self.next_state(state, action)
        reward = self.reward(state, action, next_state)
        done = (next_state == self.goal_state)

        self.agent_state = next_state
        return next_state, reward, done

    def render_v(self, v=None, policy=None, print_value=True):
        renderer = render_helper.Renderer(self.reward_map, self.goal_state,
                                          self.wall_state)
        renderer.render_v(v, policy, print_value)

    def render_q(self, q=None, print_value=True):
        renderer = render_helper.Renderer(self.reward_map, self.goal_state,
                                          self.wall_state)
        renderer.render_q(q, print_value)
```


### Class: `GridWorld`
This class represents a simple grid-based environment for reinforcement learning. It defines the grid's structure, the agent's movement, and the rewards associated with each state.

---

#### **`__init__` Method**
```python
def __init__(self):
    self.action_space = [0, 1, 2, 3]  # up, down, left, right
    self.action_meaning = {
        0: 'up',
        1: 'down',
        2: 'left',
        3: 'right'
    }
    self.reward_map = np.array([
        [0, 0, 0, 1.0],
        [0, None, 0, -1.0],
        [0, 0, 0, 0]
    ])
    self.start_state = (2, 0)
    self.wall_state = (1, 1)
    self.goal_state = (0, 3)
    self.agent_state = self.start_state
```

1. **`action_space`**: Defines the possible actions the agent can take:
   - `0`: Move up
   - `1`: Move down
   - `2`: Move left
   - `3`: Move right

2. **`action_meaning`**: Maps action indices to human-readable directions.

3. **`reward_map`**: A 2D NumPy array representing the grid. Each cell contains:
   - `0`: Neutral reward.
   - `1.0`: Positive reward (goal state).
   - `-1.0`: Negative reward (bomb state).
   - `None`: Represents an obstacle (wall).

4. **`start_state`**: The agent's starting position `(2, 0)` (row 2, column 0).

5. **`wall_state`**: The position of the wall `(1, 1)` (row 1, column 1), which the agent cannot pass through.

6. **`goal_state`**: The position of the goal `(0, 3)` (row 0, column 3).

7. **`agent_state`**: Tracks the agent's current position, initialized to the start state.

---

#### **Properties**
These properties provide useful information about the grid.

1. **`height`**
```python
@property
def height(self):
    return len(self.reward_map)
```
- Returns the number of rows in the grid.

2. **`width`**
```python
@property
def width(self):
    return len(self.reward_map[0])
```
- Returns the number of columns in the grid.

3. **`shape`**
```python
@property
def shape(self):
    return self.reward_map.shape
```
- Returns the grid's dimensions as a tuple `(rows, columns)`.

4. **`actions`**
```python
@property
def actions(self):
    return self.action_space
```
- Returns the list of possible actions.

---

#### **`state` Method**
```python
def state(self):
    for h in range(self.height):
        for w in range(self.width):
            yield (h, w)
```
- A generator that iterates over all possible states (grid cells) in the environment.
- Each state is represented as a tuple `(row, column)`.

---

#### **`next_state` Method**
```python
def next_state(self, state, action):
    action_move_map = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    move = action_move_map[action]
    next_state = (state[0] + move[0], state[1] + move[1])
    ny, nx = next_state

    if ny < 0 or ny >= self.height or nx < 0 or nx >= self.width:
        next_state = state

    if next_state == self.wall_state:
        next_state = state

    return next_state
```

1. **`action_move_map`**: Maps actions to their corresponding movements:
   - `(-1, 0)`: Move up (decrease row index).
   - `(1, 0)`: Move down (increase row index).
   - `(0, -1)`: Move left (decrease column index).
   - `(0, 1)`: Move right (increase column index).

2. **`next_state`**: Calculates the agent's next position based on the current state and action.

3. **Boundary Check**:
   - If the next state is outside the grid's boundaries, the agent stays in the current state.

4. **Wall Check**:
   - If the next state is a wall, the agent stays in the current state.

5. **Returns**: The valid next state after applying the action.

---

#### **`reward` Method**
```python
def reward(self, state, action, next_state):
    return self.reward_map[next_state]
```

1. **Inputs**:
   - `state`: The current state.
   - `action`: The action taken.
   - `next_state`: The resulting state after the action.

2. **Returns**: The reward associated with the `next_state`, as defined in the `reward_map`.

---

### Summary
The `GridWorld` class provides a simple environment for reinforcement learning:
- It defines the grid layout, including walls, rewards, and penalties.
- It allows the agent to move within the grid while handling boundaries and obstacles.
- It provides rewards based on the agent's position.


```python
env = GridWorld()
env.render_v()
```

<div align="center">
  <img src="/images/gridworld1.png" alt="gridworld" width="100%">
</div>

### Implementation of Iterative Policy Evaluation

First, let's implement a function that performs a single step of the update.

 - pi(difaultdict) : `policy`
 - V (defaultdict) : `value function`
 - env(GridWorld) : `environment`
 - gamma (float) : `discount factor`

```python
from collections import defaultdict
from common.gridworld import GridWorld

pi = defaultdict(lambda: {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25})
V = defaultdict(lambda: 0.0)

env = GridWorld()

def eval_onestep(pi, V, env, gamma = 0.9):
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

eval_onestep(pi, V, env)
env.render_v(V, pi)
```

If we try one step of the update, we can see the result below.

<div align="center">
  <img src="/images/gridworld2.png" alt="gridworld" width="100%">   
</div>

```python
        action_probs = pi[state]
        new_V = 0
        for action, action_prob in action_probs.items():
            next_state = env.next_state(state, action)
            r = env.reward(state, action, next_state)
            new_V += action_prob * (r + gamma * V[next_state])
        V[state] = new_V
```

this code is defined as follows. 

$$
s' = f(s, a)
$$

and, 

$$
V_{k + 1}(s) = \sum_{a} \pi(a | s) \left[r(s, a, s') + \gamma V_k(s') \right]
$$

Therefore, we continue repeating this process until the threshold is reached.

```python
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
```

<div align="center">
  <img src="/images/gridworld3.png" alt="gridworld" width="100%">
</div>


### Policy Iteration Method

- `Optimal Policy`: $$ \pi^*(s) $$
- `Optimal Value Function`: $$ V^*(s) $$
- `Optimal Action Value Function`: $$ Q^*(s, a) $$
- `Optimal Action`: 
<div style="overflow-x: auto;">

$$ \mu^*(s) = \arg \max_a Q^*(s, a) $$

$$
= \arg \max_a \sum_{s'} P(s' | s, a) \left[ R(s, a, s') + \gamma V^*(s') \right]
$$
</div>

What do we call the process of finding the optimal policy through repeated evaluation and greedification?

Policy Iteration is an algorithm for findings the optmal policy in a MDPs by alternating between two phase. 

- 1 **Policy Evaluation**: calculate the value function for the current policy by iteratively applying the Bellman expectation equation until convergence. 

- 2 **Policy Improvement**: update the policy to be greedy with respect to the current value function. This means for each state, selecting the action that maximizes expected value. 


By repeating these two steps until the policy no longer changes, we can find the optimal policy. This approach is guaranteed to converge to the optimal policy in finite MDPs. 

In gridworld, since states transition uniquely, we can define greedification as follows. 

<div style="overflow-x: auto;">
$$
\mu^*(s) = \arg \max_a \sum_{s'} P(s' | s, a) \left[ R(s, a, s') + \gamma V^*(s') \right]
= \arg \max_a \left[ R(s, a, s') + \gamma V^*(s') \right]
$$
</div>

```python
def greedy_policy(V, env, gamma=0.9):
    pi = {}
    for state in env.states():
        action_values = {}
        for action in env.actions():
            next_state = env.next_state(state, action)
            r = env.reward(state, action, next_state)
            action_values[action] = r + gamma * V[next_state]
        max_action = max(action_values, key=lambda a: action_values[a])
        action_probs = {a: 0.0 for a in env.actions()}
        action_probs[max_action] = 1.0
        pi[state] = action_probs
    return pi

def policy_iter(env, gamma, threshold = 1e-4, is_render = False):
    pi = defaultdict(lambda: {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25})
    V = defaultdict(lambda: 0.0)
    while True:
        V = policy_eval(pi, V, env, gamma, threshold)
        new_pi = greedy_policy(V, env, gamma)
        if is_render:
            env.render_v(V, pi)
        if new_pi == pi:
            break
        pi = new_pi
    return pi, V

if __name__ == "__main__":
    env = GridWorld()
    gamma = 0.9
    pi = policy_iter(env, gamma, is_render=True)
```

this result is two step of the policy iteration.

<div align="center">
  <img src="/images/gridworld4.png" alt="gridworld" width="100%">
</div>

Four steps of the policy iteration are as follows.

<div align="center">
  <img src="/images/gridworld5.png" alt="gridworld" width="100%">
</div>

By implementing it this way, the value function of all states is updated multiple times. It's too slow. Is there a way to update only one state's value function and proceed?


### Value Iteration Method

#### Why Value Iteration Works

Policy Iteration has two separate steps - policy evaluation (which runs until convergence) and policy improvement. This is computationally expensive because we're repeatedly evaluating the entire state space multiple times before making a single policy improvement.

Value Iteration addresses this inefficiency by recognizing that:

- **Similar Calculations** - Both policy evaluation and improvement use the Bellman equation structure
- **Partial Convergence** - We can improve the policy before the value function fully converges
- **Combined Steps** - We can directly incorporate the max operation into the value update

#### How Value Iteration Works

Value Iteration combines policy evaluation and improvement into a single update:

$$V(s) \leftarrow \max_a \left[R(s,a) + \gamma \sum_{s'} P(s'|s,a) \cdot V(s') \right]$$

This update directly finds the value of the best action for each state, effectively:

- Assuming a greedy policy at each step
- Skipping the explicit policy representation
- Performing only one sweep through the state space per iteration

#### Value Iteration Algorithm

1. Initialize $$V(s) = 0$$ for all states
2. Repeat until convergence:
   - For each state $$s$$:
    <div style="overflow-x: auto;">
         $$V(s) \leftarrow \max_a \left[R(s,a) + \gamma \sum_{s'} P(s'|s,a) \cdot V(s') \right]$$
    </div>

3. Extract the final policy:
<div style="overflow-x: auto;"> 
   $$\pi(s) = \arg\max_a \left[R(s,a) + \gamma \sum_{s'} P(s'|s,a) \cdot V(s') \right]$$
</div>

#### Advantages Over Policy Iteration

- **Computational Efficiency** - No need to perform full policy evaluation at each step
- **Fewer Iterations** - Usually converges in fewer sweeps through the state space
- **Simplicity** - Only need to maintain a value function, not an explicit policy
- **Direct Optimization** - Works towards optimal values from the start

For deterministic environments like our grid world example, the update becomes even simpler:

<div style="overflow-x: auto;">
$$V(s) \leftarrow \max_a \left[R(s,a) + \gamma \cdot V(\text{next_state}(s,a)) \right]$$
</div>

This makes Value Iteration particularly efficient for deterministic problems.

```python
def value_iter_onestep(V, env, gamma=0.9):
    for state in env.states():
        if state == env.goal_state:
            V[state] = 0.0
            continue
        action_values = []
        for action in env.actions():
            next_state = env.next_state(state, action)
            r = env.reward(state, action, next_state)
            value = r + gamma * V[next_state]
            action_values.append(value)
        V[state] = max(action_values)
    return V

def value_iter(V, env, gamma, threshold=1e-3, is_render=True):
    while True:
        if is_render:
            env.render_v(V)
        old_V = V.copy()
        V = value_iter_onestep(V, env, gamma)
        delta = 0
        for state in V.keys():
            t = abs(old_V[state] - V[state])
            if delta < t:
                delta = t
        if delta < threshold:
            break
    return V

if __name__ == "__main__":
    env = GridWorld()
    V = defaultdict(lambda: 0.0)
    gamma = 0.9
    V = value_iter(V, env, gamma, threshold=1e-3, is_render=True)
    pi = greedy_policy(V, env, gamma)
    env.render_v(V, pi)
```

1. one step of the value iteration

<div align="center">
  <img src="/images/gridworld6.png" alt="gridworld" width="100%">
</div>

2. two step of the value iteration

<div align="center">
  <img src="/images/gridworld7.png" alt="gridworld" width="100%">
</div>


3. three step of the value iteration

<div align="center">
  <img src="/images/gridworld8.png" alt="gridworld" width="100%">
</div>
4. four step of the value iteration

<div align="center">
  <img src="/images/gridworld9.png" alt="gridworld" width="100%">
</div>

So, the result of optimal policy is as follows.

<div align="center">
  <img src="/images/gridworld10.png" alt="gridworld" width="100%">
</div>

## Monte Carlo Method

We know the transition probabilities $$ P( s, a) $$ and the reward function $$ R $$, which allows us to apply Dynamic Programming.

Also, using dynamic programming (DP) is too complex to calculate the entire problem.

**What is Monte Carlo method?**

It assumes a value function for the agent to gain experience in an environment.
The experience mentioned here refers to the data (state, action, reward) obtained through the interaction between the environment and the agent.




The following situation can be considered: Think about all possible outcomes when rolling a dice twice.

<div align="center">
  <div class = 'mermaid'>
graph TD
    Start((Start))
    Start --> D1((Die 1: 1))
    Start --> D2((Die 1: 2))
    Start --> D3((Die 1: 3))
    Start --> D4((Die 1: 4))
    Start --> D5((Die 1: 5))
    Start --> D6((Die 1: 6))

    D1 --> D1_1((Die 2: 1))
    D1 --> D1_2((Die 2: 2))
    D1 --> D1_3((Die 2: 3))
    D1 --> D1_4((Die 2: 4))
    D1 --> D1_5((Die 2: 5))
    D1 --> D1_6((Die 2: 6))

    D2 --> D2_1((Die 2: 1))
    D2 --> D2_2((Die 2: 2))
    D2 --> D2_3((Die 2: 3))
    D2 --> D2_4((Die 2: 4))
    D2 --> D2_5((Die 2: 5))
    D2 --> D2_6((Die 2: 6))

    D3 --> D3_1((Die 2: 1))
    D3 --> D3_2((Die 2: 2))
    D3 --> D3_3((Die 2: 3))
    D3 --> D3_4((Die 2: 4))
    D3 --> D3_5((Die 2: 5))
    D3 --> D3_6((Die 2: 6))

    D4 --> D4_1((Die 2: 1))
    D4 --> D4_2((Die 2: 2))
    D4 --> D4_3((Die 2: 3))
    D4 --> D4_4((Die 2: 4))
    D4 --> D4_5((Die 2: 5))
    D4 --> D4_6((Die 2: 6))

    D5 --> D5_1((Die 2: 1))
    D5 --> D5_2((Die 2: 2))
    D5 --> D5_3((Die 2: 3))
    D5 --> D5_4((Die 2: 4))
    D5 --> D5_5((Die 2: 5))
    D5 --> D5_6((Die 2: 6))

    D6 --> D6_1((Die 2: 1))
    D6 --> D6_2((Die 2: 2))
    D6 --> D6_3((Die 2: 3))
    D6 --> D6_4((Die 2: 4))
    D6 --> D6_5((Die 2: 5))
    D6 --> D6_6((Die 2: 6))

    classDef circle fill:#ffffff,stroke:#000000,stroke-width:2px,shape:circle;
    class Start,D1,D2,D3,D4,D5,D6,D1_1,D1_2,D1_3,D1_4,D1_5,D1_6,D2_1,D2_2,D2_3,D2_4,D2_5,D2_6,D3_1,D3_2,D3_3,D3_4,D3_5,D3_6,D4_1,D4_2,D4_3,D4_4,D4_5,D4_6,D5_1,D5_2,D5_3,D5_4,D5_5,D5_6,D6_1,D6_2,D6_3,D6_4,D6_5,D6_6 circle;
  </div>
</div>



If some outcomes represent a probability distribution, we use the sample distribution.
A sample distribution is a method of observing the results of actual sampling.


Let's use the incremental method learned earlier to sample and calculate the expected value of the sum when two dice are rolled.

`incremental method` : $$ V_n = V_{n - 1} + \frac{1}{n} (s_n - V_{n - 1}) $$

```python
trial = 10000
import numpy as np


def Sample():
    x = 0
    for _ in range(2):
        x += np.random.choice([1, 2, 3, 4, 5, 6])
    return x


trial = 1000
V, n = 0, 0


for i in range(trial):
    s = Sample()
    n += 1
    V += (s - V) / n
    if (i + 1) % 100 == 0:
        print(f"Trial {i + 1}: Sample mean = {V}")

# result
# Trial 100: Sample mean = 7.119999999999997
# Trial 200: Sample mean = 6.8199999999999985
# Trial 300: Sample mean = 6.783333333333331
# Trial 400: Sample mean = 6.8575
# Trial 500: Sample mean = 6.844000000000001
# Trial 600: Sample mean = 6.861666666666671
# Trial 700: Sample mean = 6.8885714285714315
# Trial 800: Sample mean = 6.8999999999999995
# Trial 900: Sample mean = 6.948888888888891
# Trial 1000: Sample mean = 6.938000000000002
```

`Value - Function` : $$ V_n = \mathbb{E_{\pi}}[G \mid s] $$

This method applies the Monte Carlo approach to estimate values.

<div style="overflow-x: auto;">
$$
V_{\pi}(s) = \frac{G^{(1)} + G^{(2)}+ G^{(3)} + \cdots + G^{(n)}}{n}
$$
</div>

where $$ G^{(i)} $$ is the return of the $$ i $$-th episode.

Let me explain the first trial episode.

<div align="center">
  <div class="mermaid">
graph TD
    S((S))
    S -->|reward_1| A((A))
    A -->|reward_0| B((B))
    B -->|reward_2| END((END))

    classDef circle fill:#ffffff,stroke:#000000,stroke-width:2px,shape:circle;
    class S,A,B,C,END circle;
  </div>
</div>

<div style="overflow-x: auto;">
$$
G^{(1)} = 1 + 0 + 2 = 3
$$
</div>

The second trial episode is as follows.

<div align="center">
  <div class="mermaid">
graph TD
    S((S))
    S -->|reward_1| A((A))
    A -->|reward_0| B((B))
    B -->|reward_1| C((C))
    C -->|reward_1| END((END))

    classDef circle fill:#ffffff,stroke:#000000,stroke-width:2px,shape:circle;
    class S,A,B,C,END circle;

  </div>
</div>

<div style="overflow-x: auto;">
$$
G^{(2)} = 1 + 0 + 1 + 1 = 3
$$
</div>

As a result, the expected value is as follows. 

<div style="overflow-x: auto;">
$$
V_{\pi}(s) = \frac{G^{(1)} + G^{(2)}}{2} = \frac{3 + 3}{2} = 3
$$
</div>

Let's calculate the value function for all states using the Monte Carlo method. If there are three states (A, B, C), sample data is obtained by performing actual actions.

<div align="center">
  <div class="mermaid">
flowchart TD

    %% A 파이프라인 1
    A1((A)) --> A2([...]) --> A3([...]) --> Aout((○))

    %% A 파이프라인 2
    A1b((A)) --> A2b([...]) --> A3b([...]) --> Aoutb((○))


    %% B 파이프라인 1
    B1((B)) --> B2([...]) --> B3([...]) --> Bout((○))

    %% B 파이프라인 2
    B1b((B)) --> B2b([...]) --> B3b([...]) --> Boutb((○))

    %% C 파이프라인 1
    C1((C)) --> C2([...]) --> C3([...]) --> Cout((○))

    %% C 파이프라인 2
    C1b((C)) --> C2b([...]) --> C3b([...]) --> Coutb((○))

    %% 스타일 지정
    classDef aStyle fill:#b3d9ff,stroke:#3399ff,stroke-width:2px
    classDef bStyle fill:#ffcc99,stroke:#ff9933,stroke-width:2px
    classDef cStyle fill:#99ffcc,stroke:#33cc99,stroke-width:2px

    class A1,A2,A3,Aout,A1b,A2b,A3b,Aoutb aStyle
    class B1,B2,B3,Bout,B1b,B2b,B3b,Boutb bStyle
    class C1,C2,C3,Cout,C1b,C2b,C3b,Coutb cStyle
</div>
</div>

Let's consider starting from state A, taking actions according to policy $$ \pi $$, and reaching the final destination.

<div align="center">
  <div class="mermaid">
graph TD
    A((A))
    A -->|R0| B((B))
    B -->|R1| C((C))
    C -->|R2| END((END))

    classDef aStyle fill:#b3d9ff,stroke:#3399ff,stroke-width:2px
    classDef bStyle fill:#ffcc99,stroke:#ff9933,stroke-width:2px
    classDef cStyle fill:#99ffcc,stroke:#33cc99,stroke-width:2px
    classDef circle fill:#ffffff,stroke:#000000,stroke-width:2px,shape:circle;

    class A aStyle
    class B bStyle
    class C cStyle
    class END circle;
</div>
</div>

The total rewards accumulated from state A to the end are as follows.

<div style="overflow-x: auto;">
$$
G_A = R_0 + \gamma R_1 + \gamma^2 R_2
$$
</div>

Let's consider starting from state B. 

<div align="center">
  <div class="mermaid">
graph TD
    B((B)) -->|R1| C((C))
    C -->|R2| END((END))

    classDef aStyle fill:#b3d9ff,stroke:#3399ff,stroke-width:2px
    classDef bStyle fill:#ffcc99,stroke:#ff9933,stroke-width:2px
    classDef cStyle fill:#99ffcc,stroke:#33cc99,stroke-width:2px
    classDef circle fill:#ffffff,stroke:#000000,stroke-width:2px,shape:circle;

    class B bStyle
    class C cStyle
    class END circle;
</div>
</div>


<div style="overflow-x: auto;">
$$
G_B = R_1 + \gamma R_2
$$
</div>

Let's consider starting from state C. 

<div align="center">
  <div class="mermaid">
graph TD
    C((C)) -->|R2| END((END))

    classDef aStyle fill:#b3d9ff,stroke:#3399ff,stroke-width:2px
    classDef bStyle fill:#ffcc99,stroke:#ff9933,stroke-width:2px
    classDef cStyle fill:#99ffcc,stroke:#33cc99,stroke-width:2px
    classDef circle fill:#ffffff,stroke:#000000,stroke-width:2px,shape:circle;

    class C cStyle
    class END circle;
</div>
</div>

<div style="overflow-x: auto;">
$$
G_C = R_2
$$
</div>

So, the following sequence of calculations can eliminate redundant computations.

<div style="overflow-x: auto;">
$$
G_C = R_2
$$
</div>

<div style="overflow-x: auto;">
$$
G_B = R_1 + \gamma G_C
$$
</div>

<div style="overflow-x: auto;">
$$
G_A = R_0 + \gamma G_B
$$
</div>

### implement

Alright, according to the reading, we can implement this for the agent to interact with the environment.

<div align="center">
  <img src="/images/RL.png" alt="RL" width="100%">
</div>

The start point is (0, 0), the end point is (5, 5), and the black cell represents a wall that the agent cannot pass.

<div align="center">
  <img src="/images/monemap.png" alt="MM" width="80%">
</div>

```python
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
```

The value function (expected value) of each cell obtained through the Monte Carlo method is as follows.

<div align="center">
  <img src="/images/MM1.png" alt="MM" width="80%">
</div>

### Policy Control Using the Monte Carlo Method

The optimal policy alternates between evaluation and improvement.
- **Policy Evaluation**: Calculate the value function for the current policy using the Monte Carlo method.
- **Policy Improvement**: Update the policy to be greedy with respect to the current value function.

State Value Function Evaluation

- `General Method` : $$ V_{\pi}(s) = \frac{G^{(1)} + G^{(2)}+ G^{(3)} + \cdots + G^{(n)}}{n} $$

- `Incremental Method` : $$ V_{\pi}(s) = V_{\pi}(s) + \frac{1}{n} (G - V_{\pi}(s)) $$

Q-Function Evaluation

- `General Method` : $$ Q_{\pi}(s, a) = \frac{G^{(1)} + G^{(2)}+ G^{(3)} + \cdots + G^{(n)}}{n} $$

- `Incremental Method` : $$ Q_{\pi}(s, a) = Q_{\pi}(s, a) + \frac{1}{n} (G - Q_{\pi}(s, a)) $$


### important concept

1. Use an epsilon-greedy policy to give the agent opportunities to explore. 

2. Train the model by applying an exponential moving average with a fixed value `a`, giving greater weight to more recent data.

```python
import numpy as np
import common.gridworld_render as render_helper


class GridWorld:
    def __init__(self, reward_map, start, goal):
        self.action_space = [0, 1, 2, 3]  # 행동 공간(가능한 행동들)
        self.action_meaning = {  # 행동의 의미
            0: "UP",
            1: "DOWN",
            2: "LEFT",
            3: "RIGHT",
        }

        self.reward_map = reward_map if reward_map is not None else np.array([
            [0, 0, 0, 0, 0, 0],
            [0, None, None, None, None, 0],
            [0, None, 0, 0, None, 0],
            [0, None, 0, None, None, 0],
            [0, 0, 0, 0, 0, 10],
        ])

        self.goal_state = goal
        self.wall_state = set(
            zip(*np.where(self.reward_map == None)))  # 벽 상태(좌표)
        self.start_state = start
        self.agent_state = self.start_state   # 에이전트 초기 상태(좌표)

    @property
    def height(self):
        return len(self.reward_map)

    @property
    def width(self):
        return len(self.reward_map[0])

    @property
    def shape(self):
        return self.reward_map.shape

    def actions(self):
        return self.action_space

    def states(self):
        for h in range(self.height):
            for w in range(self.width):
                yield (h, w)

    def next_state(self, state, action):
        # 이동 위치 계산
        action_move_map = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        move = action_move_map[action]
        next_state = (state[0] + move[0], state[1] + move[1])
        ny, nx = next_state

        # 이동한 위치가 그리드 월드의 테두리 밖이나 벽인가?
        if nx < 0 or nx >= self.width or ny < 0 or ny >= self.height:
            next_state = state
        elif next_state in self.wall_state:
            next_state = state

        return next_state  # 다음 상태 반환

    def reward(self, state, action, next_state):
        return self.reward_map[next_state]

    def reset(self):
        self.agent_state = self.start_state
        return self.agent_state

    def step(self, action):
        state = self.agent_state
        next_state = self.next_state(state, action)
        reward = self.reward(state, action, next_state)
        done = (next_state == self.goal_state)

        self.agent_state = next_state
        return next_state, reward, done

    def render_v(self, v=None, policy=None, print_value=True):
        renderer = render_helper.Renderer(self.reward_map, self.goal_state,
                                          self.wall_state)
        renderer.render_v(v, policy, print_value)

    def render_q(self, q=None, print_value=True):
        renderer = render_helper.Renderer(self.reward_map, self.goal_state,
                                          self.wall_state)
        renderer.render_q(q, print_value)
```

Define the grid environment and the way the agent moves (policy).


```python
import os, sys
from tqdm import tqdm
import numpy as np
from collections import defaultdict
from common.gridworld import GridWorld

def greedy_probs(Q, state, epsilon=0.0, action_size=4):
    qs = [Q[(state, action)] for action in range(action_size)]
    max_action = int(np.argmax(qs))

    base_prob = epsilon / action_size
    action_probs = {action: base_prob for action in range(action_size)}  #{0: ε/4, 1: ε/4, 2: ε/4, 3: ε/4}
    action_probs[max_action] += (1 - epsilon)
    return action_probs


class McAgent:
    def __init__(self):
        self.gamma = 0.9
        self.epsilon = 0.1  # (첫 번째 개선) ε-탐욕 정책의 ε
        self.alpha = 0.1    # (두 번째 개선) Q 함수 갱신 시의 고정값 α
        self.action_size = 4

        random_actions = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
        self.pi = defaultdict(lambda: random_actions)
        self.Q = defaultdict(lambda: 0)
        # self.cnts = defaultdict(lambda: 0)
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

    def update(self):
        G = 0
        for data in reversed(self.memory):
            state, action, reward = data
            G = self.gamma * G + reward
            key = (state, action)
            # self.cnts[key] += 1
            # self.Q[key] += (G - self.Q[key]) / self.cnts[key]
            self.Q[key] += (G - self.Q[key]) * self.alpha
            self.pi[state] = greedy_probs(self.Q, state, self.epsilon)


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
agent = McAgent()

episodes = int(1e4)

for episode in tqdm(range(episodes), desc="Training Progress"):
    state = env.reset()
    agent.reset()

    while True:
        action = agent.get_action(state)
        next_state, reward, done = env.step(action)
        agent.add(state, action, reward)
        if done:
            agent.update()
            break
        state = next_state

env.render_q(agent.Q)
```

The agent alternates between exploration and exploitation over 10,000 episodes. Additionally, it reduces the weight of past experiences and assigns higher weight to the rewards obtained through current experiences.

The Q(S, a) for each state is as follows.

<div align="center">
  <img src="/images/Figure_1.png" alt="bandit" width="100%">
</div>

And, The Optimal Policy for each state is as follows. 

<div style="display: flex; justify-content: center; gap: 10px;">
  <img src="/images/Figure_2.png" alt="bandit1" style="width: 48%;">
  <img src="/images/Figure_3.png" alt="bandit2" style="width: 48%;">
</div>

## TD method


