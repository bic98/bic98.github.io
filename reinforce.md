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


### 비정상 문제 (Non-stationary Problem)
