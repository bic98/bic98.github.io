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

강화학습에서 두가지 문제를 해결해야 한다. 바로 정책 평가와 정책제어이다. 정책평가는 정책 $$\pi$$가 주어졌을 때, 그 정책의 value-function or action value function 을 구하는 것이다. 
