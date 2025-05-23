---
# Page settings
layout: default
keywords: 딥러닝, 머신러닝, 인공신경망, 퍼셉트론, 시그모이드, ReLU, 경사하강법, 손실함수, 인찬백, 딥러닝 기초, deep learning, machine learning, neural network, perceptron, sigmoid, gradient descent, loss function, backpropagation, activation function
comments: true
seo:
  title: Deep Learning Basics - From Perceptron to Gradient Descent | InchanBaek Note
  description: 인공신경망의 기본 구조와 학습 원리를 이해하는 완벽 가이드. 퍼셉트론, 활성화 함수, 경사하강법, 역전파 알고리즘까지 딥러닝의 핵심 개념을 단계별로 설명합니다.
  canonical: https://bic98.github.io/deep_learning/
  image: https://bic98.github.io/images/sigmoid_function.png

# Hero section
title: Deep Learning Basics
description: 퍼셉트론부터 경사하강법까지, 딥러닝의 핵심 개념을 처음부터 차근차근 배우는 딥러닝 기초 튜토리얼입니다. 인공신경망의 기본 구조와 학습 과정을 이해하기 쉽게 설명합니다.

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
#         url: '/'
#     next:
#         content: Next page
#         url: '/reinforce/'

# Language setting
---
## Introduction to AI
This document summarizes key concepts from *Deep Learning from Scratch (Volume 1).*
## What is a Perceptron?
<u>퍼셉트론</u>이란 무엇인가?<br>

퍼셉트론은 인공신경망의 한 종류로, 다수의 입력을 받아 하나의 출력을 내보내는 알고리즘이다. 
퍼셉트론은 다수의 신호를 입력으로 받아 하나의 신호를 출력한다. 이때 입력 신호에는 각각 고유한 가중치가 곱해지는데, 이 가중치는 각 신호의 중요도를 조절하는 매개변수이다.
뉴런에서 보내온 신호의 총합이 정해진 한계를 넘어설 때만 1을 출력한다.

1은 신호가 흐른다, 0은 신호가 흐르지 않는다.


<script type="text/javascript" async
  src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

$$
y =
\begin{cases}
0, & w_1 x_1 + w_2 x_2 \leq \theta \\
1, & w_1 x_1 + w_2 x_2 > \theta
\end{cases}
$$



## Simple Logic Circuits
퍼셉트론을 이용하면 간단한 논리회로를 구현할 수 있다.
### AND Gate

입력이 모두 1일 때만 출력이 1이 되는 논리회로이다.


|  x1  |  x2  |  y  |
|:---:|:---:|:---:|
| 0   | 0   | 0   |
| 1   | 0   | 0   |
| 0   | 1   | 0   |
| 1   | 1   | 1   |


```python
def AND(x1, x2):
    w1, w2, theta = 0.5, 0.5, 0.7
    tmp = x1 * w1 + x2 * w2
    if tmp <= theta:
        return 0
    elif tmp > theta:
        return 1
```



### NAND Gate and OR Gate

NAND 게이트는 AND 게이트의 출력을 반전시킨 것이다. 즉, 입력이 모두 1일 때만 출력이 0이 된다.

|  x1  |  x2  |  y  |
|:---:|:---:|:---:|
| 0   | 0   | 1   |
| 1   | 0   | 1   |
| 0   | 1   | 1   |
| 1   | 1   | 0   |

```python
def NAND(x1, x2):
    w1, w2, theta = -0.5, -0.5, -0.7  # 가중치와 임계값 설정
    tmp = x1 * w1 + x2 * w2
    if tmp <= theta:
        return 0
    elif tmp > theta:
        return 1
```

OR 게이트는 입력 신호 중 하나 이상이 1이면 출력이 1이 되는 논리회로이다.

|  x1  |  x2  |  y  |
|:---:|:---:|:---:|
| 0   | 0   | 0   |
| 1   | 0   | 1   |
| 0   | 1   | 1   |
| 1   | 1   | 1   |

```python
def OR(x1, x2):
    w1, w2, theta = 0.5, 0.5, 0.2
    tmp = x1 * w1 + x2 * w2
    if tmp <= theta:
        return 0
    elif tmp > theta:
        return 1
``` 

### Introducing Weights and Bias

세타를 -b로 치환하면 다음과 같이 표현할 수 있다.<br>


$$
y =
\begin{cases}
0, & b + w_1 x_1 + w_2 x_2 \leq 0 \\
1, & b + w_1 x_1 + w_2 x_2 > 0
\end{cases}
$$

여기서 b는 편향이라고 하며, w1과 w2는 가중치이다.


### Implementing Weights and Bias

넘파이를 이용해 가중치와 편향을 도입한 AND 게이트를 구현해보자.<br>
그렇다면 왜 넘파이를 사용하는가? 넘파이를 사용하면 배열을 쉽게 다룰 수 있기 때문이다.<br>
또한 넘파이 모듈은 그래픽카드를 이용한 병렬 계산을 지원하기 때문에 빠르게 계산할 수 있다.

```python
def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1
```

```python
def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])  # AND 게이트와 가중치 반대
    b = 0.7  # 편향 변경
    tmp = np.sum(w * x) + b
    if tmp <= 0:
        return 0
    else:
        return 1
```

```python
def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.2
    tmp = np.sum(w * x) + b
    if tmp <= 0:
        return 0
    else:
        return 1
``` 

가중치는 w1과 w2는 각 입력 신호가 결과에 주는 영향력(중요도)을 조절하는 매개변수<br>
편향은 뉴런이 얼마나 쉽게 활성화(결과로 1을 출력)하느냐를 조정하는 매개변수

위의 단층 퍼셉트론을 그래프로 나타내면 다음과 같다.

<img src="/images/AND_perceptron.png" width="600">

<img src="/images/OR_perceptron.png" width="600">

<img src="/images/NAND_perceptron.png" width="600">

## Limitations of Perceptrons

우리는 지금까지 AND, OR, NAND 게이트를 구현했다. 이제 XOR 게이트를 구현해보자.<br>
XOR 게이트는 배타적 논리합이라는 논리회로이다. 즉, x1과 x2 중 한쪽이 1일 때만 출력이 1이 된다.

|  x1  |  x2  |  y  |
|:---:|:---:|:---:|
| 0   | 0   | 0   |
| 1   | 0   | 1   |
| 0   | 1   | 1   |
| 1   | 1   | 0   |


XOR 게이트는 단층 퍼셉트론으로 구현할 수 없다.  
그 이유는 단층 퍼셉트론이 **직선 하나로 나눌 수 있는 영역**만 표현할 수 있기 때문이다.  
하지만 XOR 게이트는 **단일 직선으로 구분할 수 없는 패턴**을 가진다.  

아래 그림을 보면 이를 직관적으로 이해할 수 있다.

![XOR](/images/xor_gate_plot.png)

퍼셉트론은 직선 하나로 나눈 영역만 표현할 수 있기 때문에 XOR 게이트를 구현할 수 없다.
<br>
<div style="text-align: center;">
<b>이러한 한계를 해결하기 위해 다층 퍼셉트론을 사용한다.</b>
</div>

### Multi-Layer Perceptron

일단 XOR 게이트를 구현하려면 AND, NAND, OR 게이트를 조합해야 한다.<br>

<div align="center">
    <img src="/images/basic.png" alt="basic logic gates" width="300">
</div>

<br>
<b>어떻게 조합하면 XOR 게이트를 구현할 수 있을까?</b>


<div align="center">
    <img src="https://upload.wikimedia.org/wikipedia/commons/a/a2/254px_3gate_XOR.jpg" alt="XOR 게이트" width="300">
</div>

<br>
위 그림을 보면 XOR 게이트는 AND, NAND, OR 게이트를 조합해 만들 수 있다.


|  A  | B  |  NAND  |  OR  |  AND  |
|:---:|:---:|:---:|:---:|:---:|
| 0   | 0   | 1   | 0   | 0   |
| 1   | 0   | 1   | 1   | 1   |
| 0   | 1   | 1   | 1   | 1   |
| 1   | 1   | 0   | 1   | 0   |

```python
def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    return y
``` 






<div align="center">
    <img src="https://miro.medium.com/v2/resize:fit:1100/format:webp/1*qA_APGgbbh0QfRNsRyMaJg.png" alt="XOR 게이트" width="500">
</div>

위의 그림과 같이 다층구조의 퍼셉트론을 이용해 XOR 게이트를 구현할 수 있다.

## From Perceptron to Neural Networks

우리는 지금까지 퍼셉트론을 이용해 AND, OR, NAND, XOR 게이트를 구현했다.
신경망은 가중치의 매개변수를 적절한 값으로 자동으로 학습하는 능력이다.

### Examples of Neural Networks

신경망은 입력층, 은닉층, 출력층으로 구성되어 있다.

<div style="text-align: center;">
    <div class="mermaid">
    graph LR;
        subgraph Input Layer
            I1["input1"] 
            I2["input2"]
            I3["input3"]
        end

        subgraph Hidden Layer
            H1["Hidden1"]
            H2["Hidden2"]
            H3["Hidden3"]
        end

        subgraph Output Layer
            O1["output1"]
            O2["output2"]
        end

        %% 가중치를 포함한 엣지 연결
        I1 -- w1 --> H1
        I2 -- w2 --> H1
        I3 -- w3 --> H1
        I1 -- w4 --> H2
        I2 -- w5 --> H2
        I3 -- w6 --> H2
        I1 -- w7 --> H3
        I2 -- w8 --> H3
        I3 -- w9 --> H3

        H1 -- w10 --> O1
        H2 -- w11 --> O1
        H3 -- w12 --> O1
        H1 -- w13 --> O2
        H2 -- w14 --> O2
        H3 -- w15 --> O2
    </div>
</div>

신경망은 위에서 볼 수 있듯이 가중치를 곱한 입력 신호의 총합이 활성화 함수를 거쳐 출력값을 내보낸다.
가중치는 사람이 직접 설정하는 것이 아니라, 데이터를 학습하여 자동으로 설정된다. 이것이 신경망의 중요한 특징이다.





### Emergence of Activation Functions

신경망의 활성화 함수는 입력 신호의 총합을 출력 신호로 변환하는 함수이다.
활성화 함수는 h(x)로 표현하며, 입력 신호의 총합이 활성화를 일으키는지를 정하는 역할을 한다.


$$ y = h(b + w_1 x_1 + w_2 x_2) $$



<div style="text-align: center;">
    <div class="mermaid">
    graph LR;
        subgraph h_x
            style h_x fill:#f9f9f9,stroke:#333,stroke-width:2px,rx:50,ry:50;
            A[h]
            Y[y]
        end

        X1[x₁] -- w₁ --> A[h]
        X2[x₂] -- w₂ --> A[h]
        B[b] -- b --> A[h]
        A -- h(b + w1x1 + w2x2) --> Y[y]
    </div>
</div>

위의 그림은 활성화 함수 처리 과정을 나타낸다.



### step Function
활성화 함수에 대해 알아보자. 먼저 계단 함수를 소개한다.

$$
h(x) =
\begin{cases} 
0, & x \leq 0 \\
1, & x > 0
\end{cases}
$$

위의 활성화 함수는 계단 함수라고 한다. 계단 함수는 입력이 0을 넘으면 1을 출력하고, 그 외에는 0을 출력한다.

<div align="center">
    <img src="/images/step_function.png" alt="step function" width="400">
</div>

### Sigmoid Function

두번째로 소개할 활성화 함수는 시그모이드 함수이다. 시그모이드 함수는 입력이 커지면 1에 가까워지고, 작아지면 0에 가까워진다.

신경망에서는 활성화 함수로 시그모이드 함수를 이용해 신호를 변환하고, 그 변환된 신호를 다음 뉴런에 전달한다.

$$ h(x) = \frac{1}{1 + \exp(-x)} $$


<div align="center">
    <img src="/images/sigmoid_function.png" alt="sigmoid function" width="400">
</div>
### Comparison of Sigmoid and Step Functions
두개의 활성화 함수를 비교해보자.
시그모이드 함수는 곡선이며, 입력에 따라 출력이 연속적으로 변화한다. 반면 계단 함수는 0을 경계로 출력이 갑자기 변화한다.<br><br>
두 활성화 함수의 공통점은 입력이 작을 때는 0에 가깝고, 입력이 커지면 1에 가까워진다는 것이다.
또한 두 함수는 모두 비선형 함수이다.(선형함수의 예시 : f(x) = ax + b, 비선형함수의 예시 : f(x) = x^2)

### Non-Linear Functions

왜 비선형 함수를 사용해야 하는가? 비선형 함수를 사용하지 않으면 신경망의 층을 깊게 하는 의미가 없어진다.
선형함수의 문제는 다음과 같다. 층을 깊게 쌓아도 은닉층이 없는 네트워크로 표현할 수 있다.<br> <br>
예를 들어, h(x) = cx라는 선형함수가 있다고 하자. 이 함수를 사용한 신경망은 y(x) = h(h(h(x)))로 표현할 수 있다. 이는 y(x) = c * c * c * x로 표현할 수 있으며, 이는 y(x) = ax로 표현할 수 있다. 따라서 선형함수를 사용하면 층을 깊게 쌓는 것이 의미가 없어진다.(a = c^3 으로 표현할 수 있기 때문)
그래서 층을 쌓기 위해서는 비선형 함수를 사용해야 한다.


### ReLU Function

신경망 분야에서는 최근에 ReLU 함수를 주로 사용한다. ReLU 함수는 입력이 0을 넘으면 그 입력을 그대로 출력하고, 0 이하이면 0을 출력한다.

$$ h(x) = \max(0, x) $$

<div align="center">
    <img src="/images/relu_function.png" alt="relu function" width="400">
</div>

왜 시그모이드 함수 대신 ReLU 함수를 사용하는가? ReLU 함수는 시그모이드 함수보다 계산이 간단하다. 또한, 신경망의 학습 속도를 빠르게 하고, 효율적으로 학습할 수 있다.
<br><br>
어떻게 ReLU 함수가 학습 속도를 빠르게 하는가? 시그모이드 함수는 입력이 작을 때 기울기가 0에 가까워지는 문제가 있다. 이는 역전파에서 기울기가 사라지는 문제를 야기한다. 하지만 ReLU 함수는 입력이 0 이상이면 기울기가 1이므로, 역전파에서 기울기가 사라지는 문제를 해결할 수 있다. 기울기가 사라진다는 의미가 무엇인가? 기울기가 사라지면 가중치가 제대로 갱신되지 않는다는 의미이다. 따라서 ReLU 함수를 사용하면 학습 속도가 빨라진다.

## Computation with Multi-Dimensional Arrays

넘파이를 이용하면 다차원 배열을 쉽게 다룰 수 있다. 넘파이를 이용해 행렬의 곱셈을 계산해보자.

-1차원 배열
```python
import numpy as np
A = np.array([1, 2, 3, 4])
print(A)
# [1 2 3 4]
```


-2차원 배열
```python
B = np.array([[1, 2], [3, 4], [5, 6]])
print(B)
# [[1 2] [3 4] [5 6]]
```

-행렬의 곱셈
```python
C = np.dot(A, B)
print(C)
# [22 28]
``` 
행렬의 곱셈은 다음과 같이 계산된다.
하지만 행렬의 곱셈은 행렬의 대응하는 차원의 원소 수가 일치해야 한다. 즉, 앞의 행렬의 열 수와 뒤의 행렬의 행 수가 일치해야 한다. 또한 행렬의 곱셈은 교환법칙이 성립하지 않는다.

<div align="center">
    <img src="/images/matrix_multiplication.png" alt="matrix multiplication" width="400">
</div>
<br>

```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
C = np.dot(A, B)
print(C)
# [[19 22] [43 50]]
C = np.dot(B, A)
print(C)
# [[23 34] [31 46]]
```





## Implementing a Three-Layer Neural Network

이제 3층 신경망을 구현해보자. 이 신경망은 입력층, 은닉층, 출력층으로 구성되어 있다.

<div style = "text-align: center;">
    <div class="mermaid">
        graph LR;
    %% 입력층
    X1["x₁"] --> H1
    X1 --> H2
    X1 --> H3
    X2["x₂"] --> H1
    X2 --> H2
    X2 --> H3

    subgraph Input
        style Input fill:#add8e6,stroke:#333,stroke-width:2px,rx:20,ry:20;
        X1["x₁"]
        X2["x₂"]
    end

    %% 첫 번째 은닉층 (1층)
    subgraph Hidden1
        style Hidden1 fill:#f9f9f9,stroke:#333,stroke-width:2px,rx:20,ry:20; 
        H1["●"]
        H2["●"]
        H3["●"]
    end

    %% 첫 번째 은닉층에서 두 번째 은닉층으로 연결
    H1 --> I1
    H1 --> I2
    H2 --> I1
    H2 --> I2
    H3 --> I1
    H3 --> I2

    %% 두 번째 은닉층 (2층)
    subgraph Hidden2
        style Hidden2 fill:#f9f9f9,stroke:#333,stroke-width:2px,rx:20,ry:20;
        I1["●"]
        I2["●"]
    end

    %% 두 번째 은닉층에서 출력층으로 연결
    I1 --> Y1["y₁"]
    I1 --> Y2["y₂"]
    I2 --> Y1
    I2 --> Y2

    %% 출력층
    subgraph Output
        style Output fill:#ffc0cb,stroke:#333,stroke-width:2px,rx:20,ry:20;
        Y1["y₁"]
        Y2["y₂"]
    end

    %% 스타일 적용
    classDef neuron fill:#f9f9f9,stroke:#333,stroke-width:2px,radius:50%;
    class X1,X2,H1,H2,H3,I1,I2,Y1,Y2 neuron;

    </div>
</div>

### Implementing Signal Transmission in Each Layer


<div style="text-align: center;">
    <div class="mermaid">
        graph LR;
    subgraph Input
        style Input fill:#add8e6,stroke:#333,stroke-width:2px,rx:20,ry:20;
        X1["x₁"]
        X2["x₂"]
    end
    %% 입력층
    B["1"] -->|b₁¹| A1
    X1["x₁"] -->|w₁₁¹| A1
    X1 -->|w₁₂¹| A2
    X1 -->|w₁₃¹| A3
    X2["x₂"] -->|w₂₁¹| A1
    X2 -->|w₂₂¹| A2
    X2 -->|w₂₃¹| A3

    %% 첫 번째 은닉층 (1층)
    subgraph Hidden1
        style Hidden1 fill:#f9f9f9,stroke:#333,stroke-width:2px,rx:20,ry:20;
        A1["a₁¹"]
        A2["a₂¹"]
        A3["a₃¹"]
    end

    %% 스타일 적용
    classDef neuron fill:#f9f9f9,stroke:#333,stroke-width:2px,radius:50%;
    class B,X1,X2,A1,A2,A3 neuron;

    %% 강조 표시 (굵은 선)
    linkStyle 0 stroke-width:3px;

    </div>
</div>


1층 뉴런 $$a_1^{(1)}$$ 은 가중치를 곱한 신호 두 개와 편향을 합하여 계산한다

$$
a_1^{(1)} = w_{11}^{(1)} x_1 + w_{21}^{(1)} x_2 + b_1^{(1)}
$$


행렬의 곱을 이용하면 1층의 가중치 부분을 다음과 같이 간소화할 수 있다.

$$
A^{(1)} = X W^{(1)} + B^{(1)}
$$

이때 행렬 
$$A^{(1)}, X, B^{(1)}, W^{(1)}$$
는 각각 다음과 같다.

$$
A^{(1)} =
\begin{bmatrix}
a_1^{(1)} \\
a_2^{(1)} \\
a_3^{(1)}
\end{bmatrix},
$$

<br>

$$
X =
\begin{bmatrix}
x_1 & x_2
\end{bmatrix},
$$

<br>

$$
B^{(1)} =
\begin{bmatrix}
b_1^{(1)} \\
b_2^{(1)} \\
b_3^{(1)}
\end{bmatrix},
$$

<br>

$$
W^{(1)} =
\begin{bmatrix}
w_{11}^{(1)} & w_{12}^{(1)} & w_{13}^{(1)} \\
w_{21}^{(1)} & w_{22}^{(1)} & w_{23}^{(1)}
\end{bmatrix}
$$

입력층에서 1층으로의 신호 전달을 행렬의 곱으로 나타낼 수 있다.
그렇게 나온 결과를 활성화 함수에 넣어 출력값을 계산한다.


<div style="text-align: center;">
    <div class="mermaid">
        graph LR;

            B["1"] -->|b₁¹| A1
            X1["x₁"] -->|w₁₁¹| A1
            X1 -->|w₁₂¹| A2
            X1 -->|w₁₃¹| A3
            X2["x₂"] -->|w₂₁¹| A1
            X2 -->|w₂₂¹| A2
            X2 -->|w₂₃¹| A3

            A1["a₁¹"] -->|h| Z1["z₁¹"]
            A2["a₂¹"] -->|h| Z2["z₂¹"]
            A3["a₃¹"] -->|h| Z3["z₃¹"]

        %% 스타일 적용
        classDef neuron fill:#f9f9f9,stroke:#333,stroke-width:2px,radius:50%;
        class B,X1,X2,A1,A2,A3,Z1,Z2,Z3 neuron;
    </div>
</div>


왜 비선형 활성화함수의 출력값을 다음 레이어의 입력값으로 사용해야 하는지 알 수 있다. 만약 활성화 함수가 없다면, 신경망은 선형함수가 되어버린다. 즉, 층을 깊게 쌓는 것이 의미가 없어진다. 따라서 비선형 활성화 함수를 사용해야 한다. 또한 비선형 활성화 함수를 사용하면 신경망이 더 복잡한 문제를 풀 수 있다.


<div style="text-align: center;">
    <div class="mermaid">
     graph LR
    %% 입력층 (Input Layer)
    subgraph Input["Input Layer"]
        style Input fill:#add8e6,stroke:#333,stroke-width:2px,rx:20,ry:20;
        X1["x₁"]
        X2["x₂"]
    end

    %% 편향 뉴런 (Bias)
    B1["1"] 
    B2["1"] 
    B3["1"] 

    %% 1층 (Hidden Layer 1)
    subgraph Hidden1["Hidden Layer 1"]
        style Hidden1 fill:#f9f9f9,stroke:#333,stroke-width:2px,rx:20,ry:20;
        A1["a₁¹"] -->|σ| Z1["z₁¹"]
        A2["a₂¹"] -->|σ| Z2["z₂¹"]
        A3["a₃¹"] -->|σ| Z3["z₃¹"]
    end

    %% 2층 (Hidden Layer 2)
    subgraph Hidden2["Hidden Layer 2"]
        style Hidden2 fill:#f9f9f9,stroke:#333,stroke-width:2px,rx:20,ry:20;
        A4["a₁²"] -->|σ| Z4["z₁²"]
        A5["a₂²"] -->|σ| Z5["z₂²"]
    end

    %% 출력층 (Output Layer)
    subgraph Output["Output Layer"]
        style Output fill:#ffc0cb,stroke:#333,stroke-width:2px,rx:20,ry:20;
        A6["a₁³"] -->|σ| Y1["y₁"]
        A7["a₂³"] -->|σ| Y2["y₂"]
    end

    %% 입력층 → 1층
    X1 -->|w₁₁¹| A1
    X1 -->|w₁₂¹| A2
    X1 -->|w₁₃¹| A3
    X2 -->|w₂₁¹| A1
    X2 -->|w₂₂¹| A2
    X2 -->|w₂₃¹| A3
    B1 -->|b₁¹| A1
    B1 -->|b₂¹| A2
    B1 -->|b₃¹| A3

    %% 1층 → 2층
    Z1 -->|w₁₁²| A4
    Z2 -->|w₂₁²| A4
    Z3 -->|w₃₁²| A4
    Z1 -->|w₁₂²| A5
    Z2 -->|w₂₂²| A5
    Z3 -->|w₃₂²| A5
    B2 -->|b₁²| A4
    B2 -->|b₂²| A5

    %% 2층 → 출력층
    Z4 -->|w₁₁³| A6
    Z5 -->|w₂₁³| A6
    Z4 -->|w₁₂³| A7
    Z5 -->|w₂₂³| A7
    B3 -->|b₁³| A6
    B3 -->|b₂³| A7

    %% 스타일 적용
    classDef neuron fill:#f9f9f9,stroke:#333,stroke-width:2px,radius:50%;
    class B1,B2,B3,X1,X2,A1,A2,A3,A4,A5,A6,A7,Z1,Z2,Z3,Z4,Z5,Y1,Y2 neuron;

    %% 강조 표시 (굵은 선)
    linkStyle 0 stroke-width:3px;
    linkStyle 1 stroke-width:3px;
    linkStyle 2 stroke-width:3px;   

    </div>
</div>


위의 진행식을 파이썬으로 구현해보자.

```python

import numpy as np

def init_network():
    network = {
        'W1': np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]]),
        'b1': np.array([0.1, 0.2, 0.3]),
        'W2': np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]]),
        'b2': np.array([0.1, 0.2]),
        'W3': np.array([[0.1, 0.3], [0.2, 0.4]]),
        'b3': np.array([0.1, 0.2])
    }
    return network

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def identity_function(x):
    return x

def forward(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    
    a3 = np.dot(z2, W3) + b3
    y = identity_function(a3)
    
    return y

network = init_network()
x = np.array([1.0, 0.5])
y = forward(network, x)
print(y)  # [0.31682708 0.69627909]
```


## Designing the Output Layer

가중치와 편향은 network 딕셔너리에 저장한다. 이 딕셔너리는 init_network 함수로 초기화한다.
순전파는 forward 함수로 구현한다. 이 함수는 입력 신호를 출력으로 변환하는 처리 과정을 모두 구현한다.
순전파 처리는 다음과 같다.
1. 입력 신호를 1층의 가중치와 편향의 곱을 계산한다.
2. 1층의 출력을 활성화 함수인 시그모이드 함수에 넣어 출력을 계산한다.
3. 2층의 가중치와 편향의 곱을 계산한다.
4. 2층의 출력을 활성화 함수인 시그모이드 함수에 넣어 출력을 계산한다.
5. 3층의 가중치와 편향의 곱을 계산한다.
6. 출력층의 출력을 활성화 함수인 항등 함수에 넣어 최종 출력을 계산한다.

신경망은 분류와 회귀 문제에 모두 사용할 수 있다. 분류는 데이터가 어느 클래스에 속하는지를 구분하는 문제이다. 회귀는 입력 데이터에서 (연속적인) 수치를 예측하는 문제이다.
 
일반적으로 분류에는 소프트맥스 함수를, 회귀에는 항등 함수를 사용한다.

### Implementing Identity and Softmax Functions

**항등함수**는 입력을 그대로 출력한다. 즉, 입력이 1이면 출력도 1이다. 항등 함수는 회귀 문제에 사용된다. 회귀 문제는 입력 데이터에서 연속적인 수치를 예측하는 문제이다. 예를 들어, 입력 데이터에서 주택 가격을 예측하는 문제가 있다. 이때 출력층의 활성화 함수로 항등 함수를 사용한다.

<div style = "text-align: center;">
    <div class="mermaid">
        graph LR;
            A6["a₁³"] -->|σ| Y1["y₁"]
            A7["a₂³"] -->|σ| Y2["y₂"]
    </div>
</div>
<br>
<br>
**소프트맥스 함수**는 다음과 같이 정의된다.

$$
y_k = \frac{\exp(a_k)}{\sum_{i=1}^{n} \exp(a_i)}
$$

```python
def softmax(a):
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y
```


n은 출력층의 뉴런 수이다. 소프트맥스 함수는 출력층의 각 뉴런이 모든 입력 신호에서 영향을 받는다. 이때 소프트맥스 함수의 출력은 0에서 1.0 사이의 실수이며, 출력의 총합은 1이다. 이는 확률로 해석할 수 있다. 즉, 소프트맥스 함수를 이용해 문제를 확률적(통계적)으로 대응할 수 있다.

<div style = "text-align: center;">
    <div class="mermaid">
graph LR;
    A1["a₁"] -->|σ| Y1["y₁"]
    A1 --> Y2
    A1 --> Y3
    A2["a₂"] -->|σ| Y2["y₂"]
    A2 --> Y1
    A2 --> Y3
    A3["a₃"] -->|σ| Y3["y₃"]
    A3 --> Y1
    A3 --> Y2
            
    </div>
</div>




### Considerations When Implementing the Softmax Function

exp(x)에 만약 큰 값이 들어가면 오버플로 문제가 발생할 수 있다. 이를 해결하기 위해 소프트맥스 함수를 다음과 같이 수정할 수 있다.


```python
a = np.array([1010, 1000, 990])
```

이라면, exp(1010)은 너무 큰 값이므로 오버플로 문제가 발생한다. 이를 해결하기 위해 C를 빼주면 다음과 같다.
<br>

지수함수의 성질을 이용하여 오버플로 문제를 해결할 수 있다. 지수함수는 단조증가 함수이므로, 지수함수에 어떤 값을 더하거나 빼도 함수의 형태는 변하지 않는다. 따라서 입력 신호 중 최댓값을 빼주어 오버플로 문제를 해결할 수 있다.
<br>


$$
y_k = \frac{\exp(a_k - C)}{\sum_{i=1}^{n} \exp(a_i - C)}
$$

<br>


```python
c = np.max(a)
exp_a = np.exp(a - c)
exp_a / np.sum(exp_a)
# [9.99954600e-01 4.53978686e-05 2.06106005e-09]
```
C는 입력 신호 중 최댓값이다. 이를 이용해 소프트맥스 함수를 구현해보자.

```python
def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y
```

### Characteristics of the Softmax Function

소프트맥스 함수의 특징은 모든 출력이 0에서 1.0 사이의 실수이며, 출력의 총합은 1이라는 것이다. 이는 소프트맥스 함수의 출력을 '확률'로 해석할 수 있다. 즉, 소프트맥스 함수를 이용해 문제를 확률적(통계적)으로 대응할 수 있다.
<br>
<br>
하지만 소프트맥스 함수를 사용하지 않고 출력층의 각 뉴런의 출력값 중 가장 큰 값을 선택해도 결과는 같다. 이는 신경망의 출력이 가장 큰 뉴런에 해당하는 클래스로만 인식한다는 것이다. 이를 '단일 뉴런의 출력'이라고 한다.현업에서도 지수함수 계산에 드는 자원 낭비를 줄이기 위해 출력층의 소프트맥스 함수는 생략하는 경우가 많다. 이때는 출력층의 뉴런 중에서 가장 큰 값을 선택하면 된다.

### Determining the Number of Neurons in the Output Layer

출력층의 뉴런 수는 문제에 따라 다른다. 예를 들어, 손글씨 숫자 인식에서는 10개의 숫자(0에서 9)를 구분해야 하므로 출력층의 뉴런 수는 10개이다. 이때 소프트맥스 함수를 이용해 출력층의 출력을 계산한다. 소프트맥스 함수의 출력은 각 클래스에 대응하는 확률로 해석할 수 있다. 즉, 소프트맥스 함수를 이용해 문제를 확률적(통계적)으로 대응할 수 있다.

## Learning from Data

딥러닝을 종단간 기계학습(end-to-end machine learning)이라고 한다. 이는 데이터(입력)에서 목표한 결과(출력)를 사람의 개입 없이 얻는다는 뜻이다. 즉, 데이터로부터 학습한다는 것이다.

### Data Learning Approaches

- **Data-Driven Learning**
<br>
기계학습에는 두가지 접근법이 있다. 하나는 사람이 규칙을 만드는 방법이고, 다른 하나는 데이터로부터 규칙을 찾아내는 방법이다. 전자는 전문가 시스템이라고 하며, 후자는 데이터 기반 기계학습이라고 한다. 딥러닝은 데이터 기반 기계학습에 속한다.
- **Training Data and Test Data**
<br>
기계학습 문제는 데이터를 훈련 데이터와 시험 데이터로 나눠 학습과 실험을 수행한다. 훈련 데이터로 학습한 모델을 시험 데이터로 평가한다. 이를 통해 모델이 범용적으로 동작하는지 확인할 수 있다.
<br>
<br>
여기서 범용능력(generalization ability)이란, 아직 보지 못한 데이터(훈련 데이터에 포함되지 않은 데이터)로도 문제를 올바르게 풀어내는 능력을 말한다. 범용능력을 획득하는 것이 기계학습의 최종 목표이다.
<br>
<br>
또한 한 데이터셋에만 지나치게 최적화된 상태를 **과적합(overfitting)**이라고 한다. 이는 훈련 데이터에만 지나치게 적응되어 그 외의 데이터에는 제대로 대응하지 못하는 상태를 말한다. 과적합을 방지하는 것이 범용능력을 획득하는 핵심이다.

## Loss Function

신경망의 학습에서는 현재의 상태를 하나의 지표로 표현한다. 이 지표를 가장 좋게 만들어주는 가중치 매개변수의 값을 탐색하는 것이 학습의 목표이다. 이 지표를 **손실 함수(loss function)**이라고 한다. 손실 함수는 신경망 성능의 '나쁨'을 나타내는 지표이다. 즉, 손실 함수의 결과값이 작을수록 좋은 것이다.

### Sum of Squared Errors
가장 많이 쓰이는 손실 함수는 **평균 제곱 오차(mean squared error, MSE)**이다. 이는 신경망의 출력과 정답 레이블의 차이를 제곱한 후, 그 총합을 구한다. 평균 제곱 오차는 다음과 같이 정의된다.

$$
E = \frac{1}{2} \sum_{k} (y_k - t_k)^2
$$

여기서 $$y_k$$는 신경망의 출력, $$t_k$$는 정답 레이블, k는 데이터의 차원 수를 나타낸다.

예를 들어, 
신경망의 출력이 **y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]**이고, 
<br>
정답 레이블이 **t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]**이라면 평균 제곱 오차는 다음과 같다.

**y는 소프트 맥스 함수의 출력이므로 0과 1 사이의 값이다.**<br>
**정답 레이블은 원-핫 인코딩이므로 정답에 해당하는 인덱스의 원소만 1이고 나머지는 0이다.**

따라서 정답 레이블과 소프트맥스 함수의 출력의 차이가 작을수록 평균 제곱 오차는 작아진다.
(원핫 인코딩이란, 정답의 인덱스만 1이고 나머지는 0인 인코딩 방식이다.)
```python
def mean_squared_error(y, t):
    return 0.5 * np.sum((y - t) ** 2)

y = np.array([0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0])
t = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
mean_squared_error(y, t)
# 0.09750000000000003
```

즉, y는 소프트 맥스의 결과로 2일 확률이 가장 높다고 판단할 수 있고, t는 정답 2를 말한다. 
원소의 출력의 추정값과 정답의 차이가 작을수록 평균 제곱 오차는 작아진다.

만약에 y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]이라면 평균 제곱 오차는 다음과 같다.

```python
y = np.array([0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0])
mean_squared_error(y, t)
# 0.5975
``` 


    
### Cross-Entropy Loss

평균 제곱 오차는 신경망의 출력과 정답 레이블의 차이를 줄이면서 학습하는 것을 목표로 한다. 하지만 신경망의 출력이 확률로 해석될 때는 **교차 엔트로피 오차(cross-entropy error)**를 사용하는 것이 바람직하다. 교차 엔트로피 오차는 다음과 같이 정의된다.

$$
E = - \sum_{k} t_k \log y_k
$$

여기서 $$y_k$$는 신경망의 출력, $$t_k$$는 정답 레이블, k는 데이터의 차원 수를 나타낸다.
교차 엔트로피의 성질은 정답일 때의 출력이 전체 값을 정하게 된다. 

$$
y = log(x)
$$

<div align="center">
    <img src="/images/log_function.png" alt="log" width="400">
</div>


의 그래프를 보면 x가 1일때, y는 0이 되고, x가 0에 가까워질수록 y는음의 무한대로 커진다. 따라서 정답일 때의 출력이 전체 값을 정하게 된다.

```python
def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))
```

delta의 값을 더하는 이유는 np.log() 함수에 0을 입력하면 마이너스 무한대를 뜻하는 -inf가 되어 더 이상 계산을 진행할 수 없기 때문이다. 따라서 아주 작은 값을 더해준다.

예를 들어,
```python
y = np.array([0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0])
t = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
cross_entropy_error(y, t)
# 0.510825457099338
```

```python
y = np.array([0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0])
cross_entropy_error(y, t)
# 2.302584092994546
``` 

첫번째는 정답일 때의 출력이 0.6이고, 두번째는 0.1이다. 따라서 첫번째의 오차가 더 작다.
즉, 정답을 2라고 추정할 수 있다. 

### Mini-Batch Learning

훈련데이터에 대한 손실함수의 값을 구하고, 이 값을 최대한 줄여주는 가중치 매개변수를 찾는 것이다. 이때 손실함수의 값을 가장 작게 만드는 가중치 매개변수를 찾는 것이 목표이다.

$$
E = -\frac{1}{N} \sum_{n} \sum_{k} t_{nk} \log y_{nk}
$$

여기서 N은 데이터의 개수이다. 이때 손실함수의 값을 가장 작게 만드는 가중치 매개변수를 찾는 것이 목표이다.
N으로 나누어 정규화한다. N으로 나눔으로써 "평균 손실함수"를 구할 수 있다. 이는 데이터 개수와 관계없이 통일된 지표를 얻을 수 있다.

모든 데이터를 대상으로 손실함수의 합을 구하면 시간이 오래 걸린다. 따라서 데이터 일부를 추려 전체의 근사치로 이용할 수 있다. 이를 **미니배치 학습**이라고 한다.


먼저 케라스에 있는 데이터 셋을 가지고와서 데이터를 가공해보자. 

```python
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# MNIST 데이터 불러오기
(x_train, t_train), (x_test, t_test) = mnist.load_data()

print("훈련 데이터 크기:", x_train.shape)  # (60000, 28, 28)
print("훈련 라벨 크기:", t_train.shape)  # (60000,)
print("테스트 데이터 크기:", x_test.shape)  # (10000, 28, 28)
print("테스트 라벨 크기:", t_test.shape)  # (10000,)

x_train = x_train.reshape(-1, 28*28)  # (60000, 784)
x_test = x_test.reshape(-1, 28*28)    # (10000, 784)
print("Flatten 후 훈련 데이터 크기:", x_train.shape)  # (60000, 784)

t_train = to_categorical(t_train, num_classes=10)
t_test = to_categorical(t_test, num_classes=10)
print("One-hot 변환 후 레이블 크기:", t_train.shape)  # (60000, 10)
``` 

x_train에서 10개의 데이터를 무작위로 추출해보자.

```python
train_size = x_train.shape[0]
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size)
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]
```

np.random.choice(a, b)는 0 이상 a 미만의 수 중에서 무작위로 b개를 골라낸다. 이를 이용해 미니배치를 뽑아낼 수 있다.


```python
np.random.choice(60000, 10)
# array([8013, 1232, 8549, 12334, 9815, 123,  456,  789,  1234, 5678])
```

원-핫 인코딩으로 변환된 정답 레이블을 사용할 경우

```python
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
        
    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + 1e-7)) / batch_size
```

 2, 7등의 레이블로 주어진 경우는 아래의 코드를 사용한다. 

```python
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
        
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
```

```python
# 예측 확률 분포 (3개의 샘플, 4개의 클래스)
y = np.array([
    [0.1, 0.3, 0.4, 0.2],  # 첫 번째 샘플의 확률 분포
    [0.3, 0.2, 0.5, 0.0],  # 두 번째 샘플의 확률 분포
    [0.25, 0.25, 0.25, 0.25]  # 세 번째 샘플의 확률 분포
])

# 정답 레이블 (정수형, 각 샘플의 정답 클래스 인덱스)
t = np.array([2, 0, 3])  # 첫 번째 샘플은 클래스 2, 두 번째 샘플은 클래스 0, 세 번째 샘플은 클래스 3

# np.arange를 사용하여 정답 클래스의 확률값을 가져오기
batch_size = y.shape[0]
selected_probs = y[np.arange(batch_size), t]  # 정답 클래스에 해당하는 확률만 선택
print(selected_probs)  # [0.4 0.3 0.25]

# y[np.arange(batch_size), t]의 작동방식
y[0, 2] = 0.4   # 첫 번째 샘플의 정답 클래스(2)의 확률값
y[1, 0] = 0.3   # 두 번째 샘플의 정답 클래스(0)의 확률값
y[2, 3] = 0.25  # 세 번째 샘플의 정답 클래스(3)의 확률값
```


### Why Define a Loss Function?

왜 손실함수를 설정하는가? 신경망 학습에서 미분의 역할에 주목하면 해결된다. 신경망의 목적은 손실함의 값을 가능한 한 작게 하는 매개변수를 찾는 것이다. 이때 매개변수의 미분(정확히는 기울기)을 계산하고, 그 미분값을 단서로 매개변수의 값을 서서히 갱신하는 과정을 반복한다. 이때 손실함수의 미분값이 중요하다.

신경망을 학습할 때 정확도를 지표로 삼아서는 안된다. 정확도를 지표로 하면 매개변수의 미분이 대부분의 장소에서 0이 되기 때문이다. 즉, 매개변수의 값을 조금 바꾼다고 해도 정확도는 거의 개선되지 않는다. 이는 계단 함수를 활성화 함수로 사용하지 않는 이유와도 일맥상통한다. 계단 함수는 미분값이 대부분 0이기 때문에 신경망 학습에 사용할 수 없다.

시그모이드 함수의 미분은 어느장소라도 0이 되지 않는다. 따라서 신경망 학습에 사용할 수 있다.

<div align="center">
    <img src="/images/step_and_sigmoid_derivative.png" alt="sigmoid_derivative" width="800">
</div>

### Gradient Descent Method

경사하강법이란 함수의 기울기를 구해 기울기가 낮은 쪽으로 이동시키는 방법이다. 이때 기울기를 구할 때 사용하는 것이 바로 미분이다. 미분은 한순간의 변화량을 나타낸다. 이를 이용해 손실함수의 기울기를 구하고, 그 기울기의 반대 방향으로 매개변수를 갱신한다.

$$
\frac{\partial f(x)}{\partial x} = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}
$$

이때 h는 0에 가까운 아주 작은 값이다. 이를 이용해 수치 미분을 구할 수 있다.

```python   
def numerical_diff(f, x):
    h = 1e-4
    return (f(x+h) - f(x-h)) / (2*h)
```

모든 변수의 편미분을 벡터로 정리한 것을 **기울기(gradient)**라고 한다. 기울기는 다음과 같이 구할 수 있다.

$$
(\frac{\partial f}{\partial x_0}, \frac{\partial f}{\partial x_1})
$$

이때 기울기가 가리키는 쪽은 각 장소에서 함수의 출력 값을 가장 크게 줄이는 방향이다. 

```python
def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)
    
    for idx in range(x.size):
        tmp_val = x[idx]
        
        # f(x+h) 계산
        x[idx] = tmp_val + h
        fxh1 = f(x)
        
        # f(x-h) 계산
        x[idx] = tmp_val - h
        fxh2 = f(x)
        
        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val
        
    return grad
```

이를 이용해 $$f(x0, x1) = x0^2 + x1^2$$의 기울기를 구해보자.

```python
def function_2(x):
    return x[0]**2 + x[1]**2

numerical_gradient(function_2, np.array([3.0, 4.0]))
# array([6., 8.])
```

<div align="center">
    <img src="/images/function_and_gradient.png" alt="gradient" width="900">
</div>



하지만 기울기가 가리키는 곳에 정말 함수의 최솟값이 있는지는 보장할 수 없다. 이는 기울기가 가리키는 방향이 꼭 최솟값이 아닐 수도 있기 때문이다. 이를 해결하기 위해 경사하강법을 사용한다.

경사법을 수식으로 나타내면 다음과 같다.

$$
x_0 = x_0 - \eta \frac{\partial f}{\partial x_0}
$$

$$
x_1 = x_1 - \eta \frac{\partial f}{\partial x_1}
$$

이때 $$\eta$$는 학습률을 의미한다. 이는 매개변수 값을 갱신하는 양을 나타낸다. 즉, 학습률은 매개변수 값을 얼마나 갱신하느냐를 정하는 하이퍼파라미터이다.

```python
def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x
    
    for i in range(step_num):
        grad = numerical_diff(f, x)
        x -= lr * grad
        
    return x
```

경사법으로 $$ f(x0, x1) = x0^2 + x1^2$$의 최솟값을 구해보자.

```python
def function_2(x):
    return x[0]**2 + x[1]**2
    

init_x = np.array([-3.0, 4.0])
gradient_descent(function_2, init_x=init_x, lr=0.1, step_num=100)
# array([-6.11110793e-10,  8.14814391e-10])
``` 

이를 이용해 경사하강법을 구현할 수 있다. 이때 lr은 학습률을 의미한다. 학습률은 매개변수 값을 갱신할 때 얼마나 갱신할지를 정하는 하이퍼파라미터이다. 이 값이 너무 크거나 작으면 좋은 장소를 찾아갈 수 없다.

<div align="center">
    <img src="/images/gradient_descent_convergence.png" alt="learning_rate" width="600">
</div>

### Gradients in Neural Networks

신경망 학습에서도 기울기를 구해야 한다. 여기서 말하는 기울기는 가중치 매개변수에 대한 손실 함수의 기울기이다. 이 기울기는 가중치 매개변수의 값을 갱신하기 위해 사용한다. 이때 가중치 매개변수의 기울기를 구해야 한다. 이를 구현해보자.

가중치가 $$W$$, 손실함수가 $$L$$인 경우, 가중치 매개변수에 대한 기울기는 다음과 같이 구할 수 있다.
각 원소에 대한 편미분을 계산한다.

$$
\frac{\partial L}{\partial W}
$$

형상이 2x3인 가중치 $$W$$, 손실함수 $$L$$인 경우의 기울기를 구해보자.

$$
W = \begin{pmatrix} w_{11} & w_{12} & w_{13} \\ w_{21} & w_{22} & w_{23} \end{pmatrix}
$$

$$
\frac{\partial L}{\partial W} = \begin{pmatrix} \frac{\partial L}{\partial w_{11}} & \frac{\partial L}{\partial w_{12}} & \frac{\partial L}{\partial w_{13}} \\ \frac{\partial L}{\partial w_{21}} & \frac{\partial L}{\partial w_{22}} & \frac{\partial L}{\partial w_{23}} \end{pmatrix}
$$


$$\frac{\partial L}{\partial W}$$ 라는 의미는 손실 함수 $$L$$을 가중치 행렬 $$𝑊$$에 대해 편미분한 것으로,$$𝐿$$의 각 원소에 대한 편미분을 정리한 행렬이다.

예를 들어, 1행 1번째 원소인 $$\frac{\partial L}{\partial w_{11}}$$. 이는 $$w_{11}$$을 조금 변경했을 때 손실함수 $$L$$이 얼마나 변화하느냐를 나타낸다.



이를 구현해보자.

```python
import numpy as np  # Missing import for np

# Define softmax function which is used but not defined
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

# Define cross_entropy_error function which is used but not defined
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    
    # If t is one-hot encoded
    if t.size == y.size:
        return -np.sum(t * np.log(y + 1e-7)) / y.shape[0]
    # If t is label encoded
    else:
        return -np.sum(np.log(y[np.arange(y.shape[0]), t] + 1e-7)) / y.shape[0]

class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3)  # 정규분포로 초기화
        
    def predict(self, x):
        return np.dot(x, self.W)
    
    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)
        
        return loss
```

```python
net = simpleNet()
print(net.W)
# [[ 0.47355232 -1.6420551  -0.4380743 ]
#  [-1.1186056  -0.51709446 -0.99752602]]

x = np.array([0.6, 0.9])
p = net.predict(x)
print(p)
# [-1.06852808 -1.57996397 -1.19312484] 

np.argmax(p)  # 0
t = np.array([0, 0, 1])
net.loss(x, t)  # 1.413822588029725
``` 

이제 손실함수를 구하는 함수를 구현했으니, 이를 이용해 기울기를 구해보자.

```python
def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)
    
    for idx in range(x.size):
        tmp_val = x[idx]
        
        # f(x+h) 계산
        x[idx] = tmp_val + h
        fxh1 = f(x)
        
        # f(x-h) 계산
        x[idx] = tmp_val - h
        fxh2 = f(x)
        
        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val
        
    return grad

def f(W):
    return net.loss(x, t)

dW = numerical_gradient(f, net.W)
print(dW)
# [[ 0.21603469  0.14352979 -0.35956448]
#  [ 0.32405204  0.21529468 -0.53934672]]
```

신경망의 기울기를 구한 다음 경사하강법을 이용해 가중치 매개변수를 갱신한다.



### Implementing Learning Algorithms

**전제**
- 신경망은 적응 가능한 가중치와 편향이 있고, 이 가중치와 편향을 훈련 데이터에 적응하도록 조정하는 과정을 '학습'이라 한다.

**1단계 - 미니배치**
- 훈련 데이터 중 일부를 무작위로 가져온다. 이렇게 선별한 데이터를 미니배치라 하며, 그 미니배치의 손실 함수 값을 줄이는 것이 목표이다.

**2단계 - 기울기 산출**
- 미니배치의 손실 함수 값을 줄이기 위해 각 가중치 매개변수의 기울기를 구한다. 기울기는 손실 함수의 값을 가장 작게 하는 방향을 제시한다.

**3단계 - 매개변수 갱신**
- 가중치 매개변수를 기울기 방향으로 아주 조금 갱신한다.

**4단계 - 반복**
- 1~3단계를 반복한다.

이것이 신경망 학습이 이뤄지는 순서이다. 이때 데이터를 미니배치로 무작위로 선정하기 때문에 이를 **확률적 경사 하강법(Stochastic Gradient Descent, SGD)**라고 한다.


```python
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)  # Prevent overflow
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # Initialize weights and biases
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)
        
        return y

    def _cross_entropy_error(self, y, t):
        if y.ndim == 1:
            t = t.reshape(1, t.size)
            y = y.reshape(1, y.size)
        
        batch_size = y.shape[0]
        
        # Add small value to prevent log(0)
        delta = 1e-7
        
        # Handle both one-hot and label encoded targets
        if t.size == y.size:  # one-hot
            return -np.sum(t * np.log(y + delta)) / batch_size
        else:  # label encoded
            return -np.sum(np.log(y[np.arange(batch_size), t] + delta)) / batch_size

    def loss(self, x, t):
        y = self.predict(x)
        return self._cross_entropy_error(y, t)

    def _numerical_gradient(self, f, x):
        h = 1e-4  # Small value for numerical differentiation
        grad = np.zeros_like(x)
        
        # Vectorized operations where possible
        it = np.nditer(x, flags=['multi_index'], op_flags=[['readwrite']])
        while not it.finished:
            idx = it.multi_index
            orig_val = x[idx]
            
            # Calculate f(x+h)
            x[idx] = orig_val + h
            fxh1 = f(x)
            
            # Calculate f(x-h)
            x[idx] = orig_val - h
            fxh2 = f(x)
            
            # Gradient at this point
            grad[idx] = (fxh1 - fxh2) / (2*h)
            
            # Restore original value
            x[idx] = orig_val
            it.iternext()
        
        return grad

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1) if t.ndim != 1 else t
        
        return np.sum(y == t) / float(x.shape[0])

    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        
        grads = {}
        grads['W1'] = self._numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = self._numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = self._numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = self._numerical_gradient(loss_W, self.params['b2'])
        
        return grads
```

지금까지 배운 내용을 이용해 신경망을 학습시켜보자.


---
```python
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm  # tqdm 임포트

# MNIST 데이터 불러오기
(x_train, t_train), (x_test, t_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

print("훈련 데이터 크기:", x_train.shape)  # (60000, 28, 28)
print("훈련 라벨 크기:", t_train.shape)  # (60000,)
print("테스트 데이터 크기:", x_test.shape)  # (10000, 28, 28)
print("테스트 라벨 크기:", t_test.shape)  # (10000,)

x_train = x_train.reshape(-1, 28*28)  # (60000, 784)
x_test = x_test.reshape(-1, 28*28)    # (10000, 784)
print("Flatten 후 훈련 데이터 크기:", x_train.shape)  # (60000, 784)

t_train = to_categorical(t_train, num_classes=10)
t_test = to_categorical(t_test, num_classes=10)
print("One-hot 변환 후 레이블 크기:", t_train.shape)  # (60000, 10)

import numpy as np

iters_num = 1e4
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1)

for i in tqdm(range(int(iters_num)), ncols=80, mininterval=1.0):  # tqdm 설정
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    
    grad = network.numerical_gradient(x_batch, t_batch)
    
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]
    
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
    
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(f"train acc, test acc | {train_acc}, {test_acc}")
```
---

위와 같이 클래스로 구현하여 학습을 시키면 학습시간이 대략 9시간 정도 걸린다. 수치 미분이 쉽지만 연산과정이 너무나 많기 때문에 대용량의 데이터를 학습시키기에는 무리가 있다. 

우리는 수치 미분을 위해 중앙차분 방식을 사용했다. 딥러닝에서는 매개변수에 대한 미분을 계산해야하므로, 각 매개변수마다 독립적으로 수치미분을 계산해야한다. 

만약 네트워크 구조가 (784, 50, 10)이라면, 총 784 * 50 + 50 * 10 = 39700개의 매개변수에 대해 미분을 계산해야한다. 이는 너무 많은 계산량을 요구한다.

총 학습에 걸리는 시간을 계산해보면, 순전파 연산량 * 수치미분 연산량 * 미니배치 크기 * 에폭 수 만큼의 연산량이 필요하다. 이는 너무 많은 연산량을 요구한다.

따라서 우리는 **오차역전파법**을 사용한다. 이는 기울기를 효율적으로 계산할 수 있다.


---

### Summary

- 손실함수는 신경망 학습에서 사용하는 지표이다. 이 손실함수를 최소화하는 것이 신경망 학습의 목표이다.
- 평균 제곱 오차와 교차 엔트로피 오차를 사용한다.
- 미니배치 학습을 사용하면 훈련 데이터의 일부를 사용해 학습을 수행할 수 있다.
- 수치 미분을 사용해 가중치 매개변수의 기울기를 구할 수 있다.
- 경사하강법을 사용해 가중치 매개변수를 갱신할 수 있다.
- 신경망 학습은 미니배치로 데이터를 무작위로 선정하고, 기울기를 구해 가중치 매개변수를 갱신하는 과정을 반복한다.
- 이를 확률적 경사 하강법(SGD)이라고 한다.

---

# Backpropagation

## Computational Graph

- 계산 그래프는 계산 과정을 그래프로 나타낸 것이다.
- 계산 그래프의 노드는 연산을, 에지는 데이터를 나타낸다.
- 계산 그래프를 이용하면 계산 과정을 시각적으로 파악할 수 있다.

### Solving with a Computational Graph

간단한 문제를 계산 그래프로 풀어보자.

현빈군은 슈퍼에서 사과를 2개 샀습니다. 사과 한 개는 100원이고, 소비세가 10% 부과됩니다. 이때 현빈군이 지불하는 금액을 구해보자.

<div style="text-align: center;">
    <div class = 'mermaid'>
        graph LR
        A[apple] --> |100| B[x2]
        B -->|200| C[x1.1]
        C -->|220| D[mission complete]
    </div>
</div>

원안에 있는 노드 **x** 만을 **multiply** 연산으로 생각할 수 있다. 

<div style="text-align: center;">
    <div class = 'mermaid'>
        graph LR
        A[apple] --> |100| B[x]
        B -->|200| C[x]
        F[apple count] --> |2| B
        E[tax] --> |1.1| C
        C -->|220| D[mission complete]
    </div>
</div>


### Local Computation

계산그래프의 특징은 **국소적 계산**을 전파함으로써 최종 결과를 얻는다는 것이다. 이를 **순전파**라고 한다.

### Why Use a Computational Graph?

그렇다면, 계산그래프의 이점은 무엇일까? 계산그래프의 이점은 **국소적 계산**을 통해 최종 결과를 얻을 수 있다는 것이다. 전체가 아무리 복잡해도 각 노드에서는 단순한 계산에 집중하여 문제를 단순화할 수 있다. 

역전파는 순전파와는 반대로 노드의 미분을 효율적으로 구하는 방법이다. 이를 통해 각 노드의 미분을 효율적으로 구할 수 있다.

<div style="text-align: center;">
    <div class = 'mermaid'>
    graph LR
        A[apple] -->|100| B[x]
        B -->|200| C[x]
        C -->|220| D[mission complete]
        D -->|1| C
        C -->|1.1| B
        B -->|2.2| A
    </div>
</div>

사과 1원이 오른다면 최종금액은 2.2원이 오른다는 것을 알 수 있다.


## Chain Rule

계산 그래프의 순전파의 방향은 왼쪽에서 오른쪽으로, 역전파의 방향은 오른쪽에서 왼쪽으로 진행된다. 이 '국소적미분'을 전달하는 원리는 연쇄법칙에 따른다.

$$
y = f(x)
$$

<div style="text-align: center;">
    <div class = 'mermaid'>
    graph LR
        A[.] -->|x| B[f]
        B -->|y| C[.]
        C -->|E| B
        B -->|E ∂y/∂x| A
        
        style A opacity:0
        style C opacity:0
    </div>
</div>

위의 그림과 같이 **역전파**는 **국소적 미분**을 곱하여 전달한다. 신호 (E)에 노드의 국소적 미분 (E ∂y/∂x)을 곱한 후 다음 노드로 전달한다.

$$
y = f(x) = x^2
$$

이라면 미분은 다음과 같다.

$$
\frac{\partial y}{\partial x} = 2x
$$

상류에서 계산된 값에 국소적 미분을 곱하여 하류로 전달한다.



### What is the Chain Rule?

연쇄법칙을 설명하기 전에 합성함수에 대해 알아보자. 합성함수란 여러함수로 구성된 함수이다. 

$$
z = t^2
$$

$$
t = x + y
$$

**합성함수의 미분은 합성함수를 구성하는 각 함수의 미분의 곱으로 나타낼 수 있다.**


t에 대한 z의 미분

$$
\frac{\partial z}{\partial t} = 2t
$$

y에 대한 t의 미분

$$
\frac{\partial t}{\partial y} = 1
$$

x에 대한 t의 미분

$$
\frac{\partial t}{\partial x} = 1
$$

따라서 x에 대한 z의 미분은 다음과 같다.

<div align="center">
    <img src="/images/propagate2.png" alt="chain_rule" width="400">
</div>

$$
\frac{\partial z}{\partial x} = \frac{\partial z}{\partial t} \frac{\partial t}{\partial x}
$$

<div align="center">
    <img src="/images/propagate1.png" alt="chain_rule_example" width="400">
</div>

$$
\frac{\partial z}{\partial x} = \frac{\partial z}{\partial t} \frac{\partial t}{\partial x} = 2t * 1 = 2t = 2(x + y)
$$


## Backpropagation
### Backpropagation of Addition Nodes

덧셈노드의 역전파는 1을 곱하기만 할 뿐이다.
즉 덧셈노드의 역전파는 입력된 값을 그대로 다음 노드로 전달한다.

### Backpropagation of Multiplication Nodes
곱셈노드의 역전파는 상류의 값에 순전파 때의 입력 신호들을 '서로 바꾼 값'을 곱해서 하류로 전달한다.

$$
z = xy
$$

x에 대한 z의 미분

$$
\frac{\partial z}{\partial x} = y
$$

y에 대한 z의 미분

$$
\frac{\partial z}{\partial y} = x
$$

<div align="center">
    <img src="/images/propagate3.png" alt="multiply_node" width="800">
</div>


## Implementing Activation Function Layers
### ReLU Layer

Relu 함수의 수식은 다음과 같다.

$$
y = \begin{cases} x & (x > 0) \\ 0 & (x \leq 0) \end{cases}
$$

x에 대한 y의 미분은 다음과 같다.

$$
\frac{\partial y}{\partial x} = \begin{cases} 1 & (x > 0) \\ 0 & (x \leq 0) \end{cases}
$$

x가 0보다 크면 역전파는 상류의 값을 그대로 하류로 흘린다. x가 0이하면 신호를 하류로 보내지 않는다. 

<div align="center">
    <img src="/images/ReLu.png" alt="relu" width="800">
</div>

ReLU 클래스를 구현하면 다음과 같다. 


```python
class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx
```


### Sigmoid Layer

시그모이드 함수식은 다음과 같다. 

$$
y = \frac{1}{1 + \exp(-x)}
$$

**이를 계산그래프로 나타내면 다음과 같다.**

<div align="center">
    <img src="/images/sigmoid0.png" alt="sigmoid" width="600">
</div>
<br>
**'/' 노드, 즉 y = 1 / x 의 역전파는 다음과 같다.**

<br>

$$
\frac{\partial y}{\partial x} = - \frac{1}{x^2} = -y^2
$$

<br>


<div align="center">
    <img src="/images/sigmoid1.png" alt="sigmoid" width="600">
</div>

<br>
**'+' 노드의 역전파는 1을 곱하기만 한다.**

<br>
<div align="center">
    <img src="/images/sigmoid2.png" alt="sigmoid" width="600">
</div>

<br>

**'exp' 노드의 역전파는 순전파 때의 출력을 곱한다.**

$$
\frac{\partial y}{\partial x} = \exp(x) = y
$$

<br>
<div align="center">
    <img src="/images/sigmoid3.png" alt="sigmoid" width="600">
</div>

<br>

'X' 노드의 역전파는 순전파 때의 값을 서로 바꿔 곱한다.

$$
\frac{\partial y}{\partial x} = 1
$$

<br>
<div align="center">
    <img src="/images/sigmoid4.png" alt="sigmoid" width="600">
</div>

<br>

따라서 시그모이드 역전파의 최종 출력인

$$
\frac{\partial L}{\partial y} y^2 \exp(-x)
$$

를 하류노드로 전파한다. 


$$
\frac{\partial L}{\partial y} y^2 \exp(-x) = 
$$

$$
\frac{\partial L}{\partial y} \frac{1}{1 + \exp(-x)} (1 - \frac{1}{1 + \exp(-x)}) = 
$$

$$
\frac{\partial L}{\partial y} y (1 - y)
$$

시그 모이드 계층의 역전파는 순전파의 출력만으로 계산할 수 있다.

```python
class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out

        return out

    def backward(self, dout):
        dx = dout * self.out * (1.0 - self.out)

        return dx
```


### Affine Layer

신경망의 순전파 때 수행하는 행렬의 곱은 기하학에서 어파인 변환이라고 한다. 이는 입력 데이터에 가중치를 곱하고 편향을 더하는 것을 의미한다. (**affine transformation**)

$$
Y = XW + B
$$

- **$$X$$**: 입력 데이터 (batch size, 입력 차원)
- **$$W$$**: 가중치 행렬 (입력 차원, 출력 차원)
- **$$B$$**: 편향 벡터 (출력 차원,)
- **$$Y$$**: 출력 데이터 (batch size, 출력 차원)


Affine Layer의 역전파는 다음과 같이 계산할 수 있다.

#### **역전파 시 미분값**

- **입력에 대한 미분**:
  $$
  \frac{\partial Y}{\partial X} = W^T \quad \text{(Shape: (n, k) → (k, n))}
  $$
- **가중치에 대한 미분**:
  $$
  \frac{\partial Y}{\partial W} = X^T \quad \text{(Shape: (m, n) → (n, m))}
  $$
- **편향에 대한 미분**:
  $$
  \frac{\partial Y}{\partial B} = 1
  $$

역전파를 손실함수 L에 대해 미분하면 다음과 같다.

- **입력x에 대한 L의 미분**:

$$
\frac{\partial L}{\partial X} = \frac{\partial L}{\partial Y} W^T
$$

- **가중치w에 대한 L의 미분**:

$$
\frac{\partial L}{\partial W} = X^T \frac{\partial L}{\partial Y}
$$

- **편향B에 대한 L의 미분**:

$$
\frac{\partial L}{\partial B} = \frac{\partial L}{\partial Y}
$$

<br>
<div align="center">
    <img src="/images/affine1.png" alt="affine" width="600">
</div>


```python
class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        out = np.dot(x, self.W) + self.b

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        return dx
```


### Softmax-with-Loss Layer

Softmax-with-Loss Layer는 **소프트맥스(Softmax) 함수**와 **손실(Loss) 함수**를 합친 계층

 **1. Softmax 변환**

Softmax 함수는 입력 값을 확률 분포로 변환하는 역할을 한다. 왜 $${\sum x}$$ 대신 $${\sum e^x}$$를 사용할까? 이는 지수 함수를 사용함으로써 음수를 양수로 변환하고, 지수 함수의 특성으로 인해 큰 값을 더 크게, 작은 값을 더 작게 만들기 위함이다.


$$
S(y_i) = \frac{e^{y_i}}{\sum_{j} e^{y_j}}
$$

이를 통해 각 클래스의 확률 값을 얻을 수 있다.

 **2. Cross-Entropy Loss (교차 엔트로피 손실)**

손실 함수는 예측된 확률과 실제 정답 간의 차이를 측정하는 역할을 한다. 

$$
L = - \sum t_i \log p_i
$$

여기서:

- $$t_i$$ : 실제 정답 레이블 (one-hot encoding)
- $$p_i$$ : softmax를 거친 확률값

역전파에서 Softmax의 미분값은 다음과 같이 계산된다.

$$
\frac{\partial L}{\partial Y} = Y - T
$$

즉, softmax의 출력$$Y$$에서 실제 정답 $$T$$을 빼면 그래디언트가 계산된다.

```python
class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)

        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size

        return dx
```

## Implementing Backpropagation

**전제**
- 신경망은 적응 가능한 가중치와 편향이 있고, 이 가중치와 편향을 훈련 데이터에 적응하도록 조정하는 과정을 '학습'이라 한다.

**1단계 - 미니배치**
- 훈련 데이터 중 일부를 무작위로 가져온다. 이렇게 선별한 데이터를 미니배치라 하며, 그 미니배치의 손실 함수 값을 줄이는 것이 목표이다.

**2단계 - 기울기 산출**
- 미니배치의 손실 함수 값을 줄이기 위해 각 가중치 매개변수의 기울기를 구한다. 기울기는 손실 함수의 값을 가장 작게 하는 방향을 제시한다.

**3단계 - 매개변수 갱신**
- 가중치 매개변수를 기울기 방향으로 아주 조금 갱신한다.

**4단계 - 반복**
- 1~3단계를 반복한다.

수치 미분을 사용하여 기울기를 구하는 방법은 간단하지만 계산이 오래 걸린다. 따라서 역전파를 사용하여 기울기를 효율적으로 구할 수 있다.


```python
from keras.src.ops import normalize, one_hot
import numpy as np
from collections import OrderedDict

class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        out = np.dot(x, self.W) + self.b
        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)  # W.T is correct, but need to ensure W is not None
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        return dx

class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx

def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T 

    x = x - np.max(x)
    return np.exp(x) / np.sum(np.exp(x))

def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    
    # If t is one-hot encoded
    if t.size == y.size:
        return -np.sum(t * np.log(y + 1e-7)) / y.shape[0]
    # If t is label encoded
    else:
        return -np.sum(np.log(y[np.arange(y.shape[0]), t] + 1e-7)) / y.shape[0]

class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        if batch_size == 0:
            raise ValueError("Batch size cannot be zero")
        dx = (self.y - self.t) / batch_size
        return dx

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # Initialize weights and biases
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

        # Create layers
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        self.lastLayer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def gradient(self, x, t):
        # Forward
        self.loss(x, t)

        # Backward
        dout = 1
        dout = self.lastLayer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # Set gradients
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W2'], grads['b2'] = self.layers['Affine2'].dW, self.layers['Affine2'].db

        return grads

import tensorflow as tf
import keras
from tqdm import tqdm  # tqdm 임포트

# MNIST 데이터 불러오기
(x_train, t_train), (x_test, t_test) = keras.datasets.mnist.load_data()

# Normalize the data
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

print("훈련 데이터 크기:", x_train.shape)  # (60000, 28, 28)
print("훈련 라벨 크기:", t_train.shape)  # (60000,)
print("테스트 데이터 크기:", x_test.shape)  # (10000, 28, 28)
print("테스트 라벨 크기:", t_test.shape)  # (10000,)

x_train = x_train.reshape(-1, 28*28)  # (60000, 784)
x_test = x_test.reshape(-1, 28*28)    # (10000, 784)
print("Flatten 후 훈련 데이터 크기:", x_train.shape)  # (60000, 784)

# One-hot encode the labels
t_train = keras.utils.to_categorical(t_train, num_classes=10)
t_test = keras.utils.to_categorical(t_test, num_classes=10)
print("One-hot 변환 후 레이블 크기:", t_train.shape)  # (60000, 10)


iters_num = 1e5
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

network = TwoLayerNet(input_size=784, hidden_size=128, output_size=10)

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1)

for i in tqdm(range(int(iters_num)), ncols=80, mininterval=1.0):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    
    grad = network.gradient(x_batch, t_batch)
    
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]
    
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
    
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(f"train acc, test acc | {train_acc}, {test_acc}")

import pickle

# 모델 저장 경로 설정
model_path = 'mnist_model.pkl'

# 모델 저장
with open(model_path, 'wb') as f:
    pickle.dump(network, f)
    
print(f"모델이 {model_path}에 저장되었습니다.")
```

학습이 잘 완료된 것을 볼 수 있다. 테스트 정확도는 98% 정도이다. 

<div align = "center">
    <img src="/images/mnist_res.png" alt="mnist_result" width="800">
</div>

<br>
오차역전법의 시간 복잡도를 계산해보면 수치 미분보다 훨씬 빠르다. 오차 역전파는 순전파와 역전파를 함께 수행하여 모든 매개변수에 대한 미분을 한번에 구한다. 

**O(F(순전파 + 역전파))**

하지만 수치 미분은 모든 매개변수에 대해 미분을 계산해야하므로 **O(F(순전파) * N(네트워크의 매개변수의 개수))** 이다.

만약, 네트워크가 **(784, 128, 10)** 의 구조를 가진다면, 수치 미분은 **784 * 128 + 128 * 10 = 101632** 개의 매개변수에 대해 미분을 계산해야한다.

따라서 오차 역전법이 수치미분보다 약 **N(총 매개변수)배** 빠르다.


---
### keras model mninst
- 케라스 모델을 사용하면 더 빠르게 학습시킬 수 있다.

```python

```python
import wandb
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint
import keras

# Initialize Wandb
wandb.init(name = 'workshop_1', project="mnist_project-sgd")

# 데이터 생성
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# 모델 생성
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation="sigmoid"),
    keras.layers.Dense(10, activation="softmax"),
])

# 컴파일 및 학습
model.compile(optimizer="sgd", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Define the checkpoint callback
checkpoint_callback = WandbModelCheckpoint(filepath="models/mnins_sgd.keras", save_freq='epoch')

model.fit(
    x_train,
    y_train,
    epochs=128,
    validation_data=(x_test, y_test),
    callbacks=[
        WandbMetricsLogger(),
        checkpoint_callback,
    ],
)
``` 

위와 같이 케라스 모델을 사용하면 더 빠르게 학습시킬 수 있다.
손실함수 그래프와 정확도 그래프를 확인해보자. 128번의 에폭을 돌린 결과이다.

<div align="center">
    <img src="/images/mnist_sgd (2).png" alt="loss_accuracy" width="600">
    <img src="/images/mnist_sgd (3).png" alt="loss_accuracy" width="600">
</div>
<br>
에폭이 증가할수록 손실함수는 감소하고 정확도는 증가하는 것을 확인할 수 있다.

## Parameter Update

### Limitations of SGD
- SGD는 비등방성 함수(방향에 따라 기울기가 달라지는 함수)에서는 비효율적으로 움직인다.
- SGD는 비등방성 함수에서는 지그재그로 움직이면서 수렴하는 데 시간이 오래 걸린다.

예를 들어 다음과 같은 함수를 생각해보자.

$$
f(x, y) = \frac{1}{20}x^2 + y^2
$$

이 함수에서 SGD는 다음과 같이 움직인다.

```python
import numpy as np
import matplotlib.pyplot as plt

# Define the function
def f(x, y):
    return (1/20) * x**2 + y**2

# Gradient of the function
def grad_f(x, y):
    df_dx = (1/10) * x
    df_dy = 2 * y
    return np.array([df_dx, df_dy])

# Stochastic Gradient Descent (SGD) parameters
learning_rate = 0.9
num_iterations = 100

# Initial point
x, y = -5.0, 5.0

# Store the path
path = [(x, y)]

# Perform SGD
for _ in range(num_iterations):
    grad = grad_f(x, y)
    x -= learning_rate * grad[0]
    y -= learning_rate * grad[1]
    path.append((x, y))

# Convert path to numpy array for plotting
path = np.array(path)

# Plot the function and the path taken by SGD
Z = f(X, Y)
plt.figure(figsize=(8, 6))
plt.contour(X, Y, Z, levels=30, cmap='viridis')
plt.plot(path[:, 0], path[:, 1], 'r.-', markersize=10)  # Increase markersize
plt.xlim(-6, 6)  # Set x-axis limits to zoom in
plt.ylim(-6, 6)  # Set y-axis limits to zoom in
plt.title('SGD Path on f(x, y)')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
```


<div align="center">
    <img src="/images/sgd_path.png" alt="sgd" width="800">
</div>

이러한 SGD의 단점을 개선해주는 Momentum, AdaGrad, Adam이라는 세가지 방법이 있다. 


### Momentum

모멘텀은 SGD의 단점을 개선한 방법이다. 모멘텀은 '운동량'을 의미하며, SGD에 관성을 더해준다.

### AdaGrad

신경망 학습에서 학습률이 너무 작으면 학습시간이 길어지고, 반대로 크면 발산하여 제데로 이루어지지 않는다. 
학습률을 정하는 **학습률 감소** 방법이 있다. 이 방법은 학습을 진행하면서 학습률을 점차 줄여가는 방법이다.

**AdagGrad**는 각각의 매개변수에 맞춤형 값을 만들어준다. 개별 매개변수에 적응적으로 학습률을 조정한다.

$$
h \leftarrow h + \frac{\partial L}{\partial W} \odot \frac{\partial L}{\partial W}
$$

$$
W \leftarrow W - \eta \frac{1}{\sqrt{h}} \frac{\partial L}{\partial W}
$$

**h** 는 기존 기울기 값을 제곱하여 계속 더해준다. 그리고 매개변수를 갱신할 때 $$1/sqrt(h)$$ 를 곱해 학습률을 조정한다.

매겨변수의 원소 중에서 많이 움직인 원소는 학습률이 낮아진다. 따라서 학습률 감소가 매개변수의 원소마다 다르게 적용된다.

```python
class AdaGrad:
    def __init__(self, lr=0.01):
        self.lr = lr
        self.h = None

    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)

        for key in params.keys():
            self.h[key] += grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)
```


### Adam

Adam은 모멘텀과 AdaGrad를 융합한 방법이다. Adam은 하이퍼파라미터의 '편향 보정'이 진행된다.

### Which Update Method Should Be Used?

SGD, 모멘텀, AdaGrad, Adam 중 어떤 것을 사용해야 할까? 그 정답은 없다. 각자의 데이터에 맞게 실험을 통해 최적의 방법을 찾아야 한다.

<div align="center">
    <img src="/images/adam.png" alt="sgd_adampng" width="600">
</div>


## Weight Initialization

가중치의 초깃값을 적절히 설정하면 학습이 원활하게 이루어진다. 가중치의 초깃값에 따라 학습이 잘 되거나 잘 되지 않는 경우가 있다.

### What If the Initial Weights Are Set to Zero?

가중치의 값을 모두 0으로 설정하면 학습이 잘 이루어지지 않는다. 이는 오차역전파법에서 모든 가중치의 값이 똑같이 갱신되기 때문이다.

예를 들어, 2층 신경망에서 가중치를 0으로 초기화하고 학습을 진행하면 다음과 같은 문제가 발생한다. 

$$
W = \begin{pmatrix} 0 & 0 \\ 0 & 0 \end{pmatrix}
$$

$$
\frac{\partial L}{\partial W} = \begin{pmatrix} 0 & 0 \\ 0 & 0 \end{pmatrix}
$$

$$
W = W - \eta \frac{\partial L}{\partial W} = \begin{pmatrix} 0 & 0 \\ 0 & 0 \end{pmatrix}
$$

가중치의 표준편차가 너무 크면 각 층의 활성화값 분포가 너무 넓어져서 학습이 잘 이루어지지 않는다. 반대로 표준편차가 너무 작으면 각 층의 활성화값 분포가 너무 좁아져서 학습이 잘 이루어지지 않는다. 따라서 적절한 표준편차를 찾아야 한다.

예들들어 표준편차가 크면, 활성화값이 0과 1에 치우쳐져 있어 기울기 소실이 발생할 수 있다.sigmoid 함수와 relu함수를 생각해보면 0과 1에 치우쳐져 있으면 sigmoid 함수는 기울기 소실이 발생할 수 있고, 너무 큰 초깃값을 사용하게 되면 ReLU일 경우에는 음수의 가중합이 계속 증가하게 되어 뉴런이 죽을 수 있다.(dying ReLU) 반대로 표준편차가 작으면 활성화값이 0.5 부근에 집중되어 (모든 뉴런이 비슷한 출력을 내는) 표현력을 제한할 수 있다.






---
그렇다면 초깃값을 어떻게 설정해야할까? 

### Xavier 초기값

Xavier 초기값은 Glorot & Bengio(2010)가 제안한 방법으로, 각 층의 활성화값들이 적절히 분포되도록 하는 초기화 방식이다. 시그모이드나 tanh같은 활성화 함수의 중심 영역(0 부근)에서는 근사적으로 선형 특성을 보이는 점을 활용한다.

- **적용 대상**: 시그모이드, tanh 같은 대칭적인 활성화 함수에 적합
- **수식**:
  - 균등 분포의 경우: $$W \sim U\left[-\frac{\sqrt{6}}{\sqrt{n_{in} + n_{out}}}, \frac{\sqrt{6}}{\sqrt{n_{in} + n_{out}}}\right]$$
  - 정규 분포의 경우: $$W \sim N\left(0, \frac{2}{n_{in} + n_{out}}\right)$$
  - 여기서 $$n_{in}$$은 입력 뉴런 수, $$n_{out}$$은 출력 뉴런 수
- **특징**: 순전파와 역전파 과정에서 신호의 분산을 유지하여 기울기 소실/폭발 문제 완화

### He 초기값

He 초기값은 Kaiming He 등이 제안한 방법으로, ReLU 활성화 함수를 위해 특별히 설계되었다.

- **적용 대상**: ReLU, Leaky ReLU 등의 비선형 활성화 함수에 최적화
- **수식**:
  - 균등 분포의 경우: $$W \sim U\left[-\sqrt{\frac{6}{n_{in}}}, \sqrt{\frac{6}{n_{in}}}\right]$$
  - 정규 분포의 경우: $$W \sim N\left(0, \frac{2}{n_{in}}\right)$$
- **특징**: ReLU는 음수 입력을 0으로 만들어(입력의 약 절반) 활성화된 뉴런의 수가 감소하므로, 
  Xavier보다 더 큰 초기값을 사용하여 분산을 유지한다. 적절한 크기로 초기화하지 않으면 
  많은 뉴런이 입력값으로 음수를 받아 0을 출력하게 되는 "죽은 ReLU" 문제가 발생할 수 있다.

```python
import numpy as np

def xavier_init(n_inputs, n_outputs, uniform=True):
    """
    Xavier(Glorot) 초기화
    
    Parameters:
    - n_inputs: 입력 뉴런 수
    - n_outputs: 출력 뉴런 수
    - uniform: 균등 분포 사용 여부 (False면 정규 분포)
    
    Returns:
    - 초기화된 가중치 행렬
    """
    if uniform:
        # 균등 분포
        limit = np.sqrt(6. / (n_inputs + n_outputs))
        return np.random.uniform(-limit, limit, size=(n_inputs, n_outputs))
    else:
        # 정규 분포
        stddev = np.sqrt(2. / (n_inputs + n_outputs))
        return np.random.normal(0.0, stddev, size=(n_inputs, n_outputs))

def he_init(n_inputs, n_outputs, uniform=True):
    """
    He 초기화
    
    Parameters:
    - n_inputs: 입력 뉴런 수
    - n_outputs: 출력 뉴런 수
    - uniform: 균등 분포 사용 여부 (False면 정규 분포)
    
    Returns:
    - 초기화된 가중치 행렬
    """
    if uniform:
        # 균등 분포
        limit = np.sqrt(6. / n_inputs)
        return np.random.uniform(-limit, limit, size=(n_inputs, n_outputs))
    else:
        # 정규 분포
        stddev = np.sqrt(2. / n_inputs)
        return np.random.normal(0.0, stddev, size=(n_inputs, n_outputs))

# 사용 예시
input_size = 784   # 28x28 이미지
hidden_size = 50   # 은닉층 뉴런 수

# Xavier 초기화 (시그모이드, tanh 활성화 함수에 적합)
W1_xavier = xavier_init(input_size, hidden_size)
print(f"Xavier 초기화 가중치 범위: {W1_xavier.min():.4f} ~ {W1_xavier.max():.4f}")

# He 초기화 (ReLU 활성화 함수에 적합)
W1_he = he_init(input_size, hidden_size)
print(f"He 초기화 가중치 범위: {W1_he.min():.4f} ~ {W1_he.max():.4f}")

```


대부분의 딥러닝 프레임 워크에서는 정규 분포를 사용하여 초기화한다. 

## Batch Normalization

배치 정규화에 대해서 알아보자. 배치 정규화는 각 층의 활성화값이 적당히 분포되도록 조정하는 방법이다.

### Batch Normalization Algorithm

 - **배치 정규화**는 학습 시 미니배치를 단위로 정규화한다. 구체적으로는 데이터 분포가 평균이 0, 분산이 1이 되도록 정규화를 함. 

$$
\mu_B = \frac{1}{m} \sum_{i=1}^{m} x_i
$$

$$
\sigma_B^2 = \frac{1}{m} \sum_{i=1}^{m} (x_i - \mu_B)^2
$$

$$
\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}
$$

미니배치 B = {x1, x2, ..., xm}에 대해 평균 $$\mu_B$$ 과 분산$$\sigma_B^2$$ 을 구한 뒤, 이를 사용해 정규화한다. 여기서 ε은 0으로 나누는 것을 방지하기 위한 작은 값이다.


배치 정규화 계층마다 이 정규화된 데이터에 확대(scale)와 이동(shift) 변환을 수행한다.

$$
y_i = \gamma \hat{x}_i + \beta
$$

- $$\gamma$$: 확대(scale) 파라미터
- $$\beta$$: 이동(shift) 파라미터

처음에는 $$\gamma = 1$$, $$\beta = 0$$부터 시작하고, 학습하면서 적합한 값으로 조정해간다.

배치 정규화의 효과

- 학습 속도 개선
- 초깃값에 크게 의존하지 않는다.
- 오버피팅 억제

### Overfitting

오버피팅이란? 신경망이 훈련 데이터에만 지나치게 적응되어 그 외의 데이터에는 제대로 대응하지 못하는 상태를 말한다. 


오버피팅은 다음의 두경우에 일어난다. 

- 매개변수가 많고 표현력이 높은 모델
- 훈련데이터가 작을 때


### Weight Decay

오버피팅 억제용으로 가중치 감소를 사용한다. 

신경망의 목적은 손실함수의 값을 줄이는 것이다. 이때 가중치의 제곱 노름(L2 노름)을 손실함수에 더한다. 가중치를 $$W$$라 하면, L2 노름에 따른 가중치 감소는 $$\frac{1}{2} \lambda W^2$$이 된다. 여기서 $$\lambda$$는 정규화의 세기를 조절하는 하이퍼파라미터이다. 

가중치 감소는 모든 가중치 각각의 손실 함수에 $$
\frac{1}{2} \lambda W^2$$를 더한다. 따라서 가중치 감소는 가중치의 값이 작아지도록 학습하는 효과가 있다. 



1. **손실 함수에 추가 항목**: 가중치 감소는 손실 함수에 가중치의 제곱 노름(L2 노름)을 추가하는 방식으로 구현됨. 예를 들어, 원래 손실 함수가 $$L$$이라면, 가중치 감소를 적용한 손실 함수는 다음과 같음

$$L' = L + \frac{1}{2} \lambda \sum W^2$$


2. **가중치 감소 효과**: 이 추가 항목은 가중치가 커질수록 손실 함수의 값이 커짐. 따라서, 모델은 손실 함수를 최소화하기 위해 가중치의 값을 작게 유지하려고 학습하게 됨.

3. **과적합 방지**: 가중치가 너무 크면 모델이 훈련 데이터에 과적합될 가능성이 높아짐. 가중치 감소를 통해 가중치의 값을 작게 유지하면, 모델이 더 단순해지고, 이는 과적합을 방지하는 데 도움이 됨.

### Dropout

신경망 모델이 복잡해지면 가중치 감소만으로 오버피팅에 대응하기 어려움. 

드롭아웃은 학습 과정에서 신경망의 일부 뉴런을 무작위로 비활성화(즉, 0으로 설정)하는 기법이다. 이렇게 하면 특정 뉴런이나 특정 경로에 과도하게 의존하는 것을 방지할 수 있다.

1. **훈련 시**: 각 학습 단계에서 각 뉴런을 일정 확률(p)로 비활성화한다. 예를 들어, p=0.5라면, 각 뉴런이 50% 확률로 비활성화된다.
2. **예측 시**: 예측 단계에서는 모든 뉴런을 사용하지만, 각 뉴런의 출력을 훈련 시 비활성화된 확률(p)로 조정한다. 예를 들어, p=0.5라면, 각 뉴런의 출력을 절반으로 줄인다.

- **훈련 시**: 네트워크의 일부 뉴런을 무작위로 비활성화하여, 네트워크가 특정 뉴런에 과도하게 의존하지 않도록 한다.
- **예측 시**: 모든 뉴런을 사용하지만, 각 뉴런의 출력을 훈련 시 비활성화된 확률로 조정하여, 훈련과 예측 간의 일관성을 유지한다.

- **과적합 방지**: 드롭아웃은 네트워크가 특정 뉴런이나 경로에 과도하게 의존하지 않도록 하여, 과적합을 방지한다.
- **앙상블 효과**: 드롭아웃은 여러 다른 신경망의 앙상블을 학습하는 것과 유사한 효과를 가진다. 이는 모델의 일반화 성능을 향상시킨다.

### 결론
드롭아웃은 과적합을 방지하기 위한 또 다른 강력한 기법이다. 가중치 감소와 함께 사용되기도 하며, 신경망의 일반화 성능을 향상시키는 데 큰 도움이 된다.

```python
class Dropout:
    def __init__(self, dropout_ratio=0.5):
        self.dropout_ratio = dropout_ratio
        self.mask = None

    def forward(self, x, train_flg=True):
        if train_flg:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_ratio)

    def backward(self, dout):
        return dout * self.mask
```
<div align="center">
    <img src="/images/dropout.png" alt="dropout" width="100%">
</div>

왼쪽은 드롭아웃 없이, 오른쪽은 드롭아웃을 적용한 경우이다. 


- `Ensemble Learning` :  
앙상블 학습(Ensemble Learning)은 여러 개의 모델을 결합하여 하나의 강력한 모델을 만드는 기법이다. 이를 통해 단일 모델보다 더 나은 성능과 일반화 능력을 얻을 수 있다. 앙상블 학습은 주로 분류(classification)와 회귀(regression) 문제에서 사용된다.

### 주요 앙상블 기법

1. **배깅(Bagging)**
   - **개념**: 배깅은 Bootstrap Aggregating의 줄임말로, 여러 개의 모델을 독립적으로 학습시키고 그 예측 결과를 평균 내거나 다수결 투표를 통해 최종 예측을 만드는 방법이다.
   - **예시**: 랜덤 포레스트(Random Forest)는 배깅의 대표적인 예로, 여러 개의 결정 트리(decision tree)를 학습시키고 그 결과를 결합하여 최종 예측을 만든다.

2. **부스팅(Boosting)**
   - **개념**: 부스팅은 약한 학습기(weak learner)를 순차적으로 학습시키고, 이전 모델이 틀린 예측에 더 큰 가중치를 부여하여 다음 모델이 이를 보완하도록 하는 방법이다.
   - **예시**: AdaBoost, Gradient Boosting, XGBoost 등이 부스팅의 대표적인 예이다.

3. **스태킹(Stacking)**
   - **개념**: 스태킹은 여러 개의 다른 모델을 학습시키고, 이들의 예측 결과를 새로운 모델의 입력으로 사용하여 최종 예측을 만드는 방법이다. 메타 모델(meta-model)이 이 예측 결과를 조합하여 최종 예측을 만든다.
   - **예시**: 다양한 분류기나 회귀기를 조합하여 최종 예측을 만드는 데 사용된다.

### 장점
- **성능 향상**: 여러 모델의 예측을 결합함으로써 단일 모델보다 더 나은 성능을 얻을 수 있다.
- **일반화 능력**: 다양한 모델을 결합하여 과적합(overfitting)을 방지하고, 더 좋은 일반화 성능을 얻을 수 있다.

### 단점
- **복잡성 증가**: 여러 모델을 학습시키고 결합하는 과정이 복잡하고 계산 비용이 많이 든다.
- **해석 어려움**: 앙상블 모델은 단일 모델보다 해석하기 어려울 수 있다.

### 결론
앙상블 학습은 여러 모델을 결합하여 성능과 일반화 능력을 향상시키는 강력한 기법이다. 배깅, 부스팅, 스태킹 등의 다양한 방법이 있으며, 각각의 방법은 특정 상황에서 유리하게 작용할 수 있다.

## Finding the Optimal Hyperparameter Values

신경망에는 다양한 하이퍼파라미터가 존재한다. 이러한 하이퍼파라미터를 최적화하는 것은 매우 중요하다.
예를 들어, 신경망의 은닉층 수, 은닉층의 뉴런 수, 학습률, 배치 크기, 가중치 초기화 방법, 최적화 방법, 드롭아웃 비율 등이 있다. 하이퍼 파라미터의 값을 최대한 효율적으로 찾는 것이 중요하다.



### Implementing Hyperparameter Optimization

하이퍼 파라미터를 최적화할 때의 핵심은 하이퍼파라미터의 '최적값'이 존재하는 범위를 조금씩 줄여가면 됨. 
우선 대략적인 범위를 설정하고 그 범위에서 무작위로 하이퍼파라미터 값을 골라낸 후,  그 값으로 정확도를 평가하면됨. 

하이퍼파리미터의 범위는 대략적으로 지정하는 것이 효과적임. 10의 거듭제곱 범위로 범위를 지정하는 로그 스케일이 일반적이다. 

- 0단계: 하이퍼파라미터 값의 범위 설정
- 1단계: 설정된 범위에서 하이퍼파라미터 값 무작위로 추출
- 2단계: 1단계에서 샘플링한 하이퍼파라미터 값을 사용하여 학습하고 검증 데이터로 정확도 평가
- 3단계: 1단계와 2단계를 특정 횟수(100회 등) 반복하고, 그 정확도의 결과를 보고 하이퍼파라미터의 범위를 좁힌다.


# Convolutional Neural Networks (CNN)

합성곱 신경망에 대해 알아보자. CNN은 이미지 인식과 음성 인식 등 다양한 분야에서 사용되는 신경망이다. CNN은 이미지의 공간적 구조를 활용하여 이미지 처리를 수행한다.

### Overall Structure

합성곱 계층과 풀링 계층이 추가된다.

`지금까지 배운 신경망`
- 완전연결 계층: 인접하는 계층의 모든 뉴런과 결합되어 있음

<div align="center">
    <img src="/images/aff.png" alt="affine2" width="100%">
</div>

`CNN`
- 합성곱 계층: 입력 데이터 전체에 가중치를 적용
- 풀링 계층: 가로/세로 방향의 공간을 줄이는 연산

<div align="center">
    <img src="/images/cnn.png" alt="cnn" width="100%">
</div>


### Convolutional Layers

CNN에서는 padding, stride, filter, channel 등의 개념이 사용된다. 각 계층 사이에는 3차원 데이터같이 입체적인 데이터가 흐른다는 점에서 완전연결 신경망과는 다르다. 

- `Problems with Fully Connected Layers`

완전연결 계층의 문제점은 데이터의 형상이 무시된다는 것이다. 예를 들어, 이미지는 3차원 데이터인데 완전연결 계층에 입력할 때 1차원 데이터로 평탄화해야 한다. 이렇게 되면 데이터의 형상이 무시된다. 
<br>
<br>
예를 들어 이미지는 가로, 세로, 채널로 구성된 3차원 데이터이다. 이를 완전연결 계층에 입력하려면 1차원 데이터로 평탄화해야 한다. (1, 28, 28) => (1, 784) 공간적으로 가까운 픽셀이나 색상이 비슷한 픽셀은 비슷한 값이 되어야 한다. 그러나 완전연결 계층은 이러한 공간적 정보를 무시한다.
<br>
<br>
**합성곱 계층은 형상을 유지한다.**
<br>
<br>
CNN에서는 합성곱 계층의 입출력 데이터를 특징 맵(feature map)이라고 한다. 합성곱 계층의 입력 데이터를 입력 특징 맵, 출력 데이터를 출력 특징 맵이라고 한다.


- `Convolution Operation`

합성곱 연산은 이미지 처리에서 말하는 필터연산에 해당한다. 필터는 커널이라고도 하며, 이 필터를 이미지의 왼쪽 위부터 오른쪽 아래까지 차례대로 이동하면서 적용한다. 이때 필터의 윈도우가 이미지의 모서리 부분에 도달하면 윈도우를 한 칸 아래로 내리고 다시 왼쪽 끝으로 이동한다. 이러한 연산을 수행하면 이미지의 특징을 추출할 수 있다.

<div align="center">
    <img src="/images/conv.png" alt="conv" width="70%">
</div>

- `Padding`

패딩이란 합성곱 연산을 수행하기 전에 입력데이터 주변을 특정값으로 채우는 것을 말한다. 패딩은 주로 출력 크기를 조정할 목적으로 사용된다. 

<div align="center">
    <img src="/images/pad.png" alt="padding" width="70%">
</div>

- `Stride`

필터를 적용하는 위치의 간격을 스트라이드(stride)라고 한다. 스트라이드는 필터를 적용하는 간격을 말한다. 스트라이드가 1이면 필터를 한 칸씩 이동하고, 2이면 두 칸씩 이동한다.

<div align="center">
    <img src="/images/str.png" alt="stride" width="70%">
</div>

입력 크기를 (H, W), 필터 크기를 (FH, FW), 출력 크기를 (OH, OW), 패딩을 P, 스트라이드를 S라고 하면 출력 크기는 다음과 같이 계산된다.

$$
OH = \frac{H + 2P - FH}{S} + 1
$$

$$
OW = \frac{W + 2P - FW}{S} + 1
$$

- `Convolution with 3D Data`

3차원의 합성곱 연산에서 주의할 점은 입력 데이터의 채널 수와 필터의 채널 수가 같아야 한다는 것이다.

<div align="center">
    <img src="/images/thr.png" alt="conv3d" width="70%">
</div>


필터를 Fn개 적용하면 출력 맵도 Fn개 생성된다. 이 Fn개의 출력 맵을 모으면 (FN, OH, OW) 크기의 데이터가 된다.

<div align="center">
    <img src="/images/fno.png" alt="map" width="100%">
</div>



### Pooling Layers
pooling은 세로 가로 방향의 공간을 줄이는 연산이다. 
예를 들어 2x2 최대 풀링은 2x2 영역에서 가장 큰 원소 하나를 꺼내는 연산이다. 스트라이드 2로 처리하면 출력 크기는 입력의 절반으로 줄어든다.

그리고 풀리의 윈도우크기와 스트라이드의 값은 같은 값으로 처리하는게 일반적이다.  3x3이면 스트라이트는 3으로 4x4이면 스트라이트는 4로 처리한다.

- Characteristics of Pooling Layers 

- 학습해야 할 매개변수가 없다.
최댓값이나 평균을 취하는 연산이므로 학습해야 할 매개변수가 없다.
- 채널 수가 변하지 않는다.
입력 데이터의 채널 수 그대로 출력 데이터의 채널 수가 유지된다.
- 입력의 변화에 영향을 적게 받는다.
입력 데이터가 조금 변해도 풀링의 결과는 잘 변하지 않는다.



### Implementing Convolution/Pooling Layers

지금까지 합성곱 계층과 풀링계층에 대하 알아보았음. 파이썬으로 구현해보자. 

- Expanding Data with im2col  

im2col은 입력 데이터를 필터링하기 좋게 전개하는 함수이다. 
아래 그림과 같이 필터를 적용하는 영역(3차원 블록)을 한 줄로 전개한다.


im2col의 약자는 image to column이다. 입력이미지에서 커널이 움직이면서 보는 부분을 슬라이딩 윈도우하면서 잘라냄

<div align="center">
    <img src="/images/imgcol.png" alt="im2col2" width="70%">
</div>

배치크기가 1, 채널 3, 가로 7, 세로 7의 데이터일경우 im2col(직접구현하기 보다 torch패키지의 unfold모듈을 사용함)을 해보면 2차원 배열이 만들어 진다. (1, 75, 9)

75 = 커널사이즈 * 커널사이즈 * 채널
9 = 3 * 3(슬라이딩 윈도우)

```python
import torch


x1 = torch.randn(1, 3, 7, 7)
patches = torch.nn.functional.unfold(x1, kernel_size = 5, stride = 1, padding = 0)
print(patches.shape) # torch.Size([1, 75, 9])
```

- Implementing Convolution Layers  

```python

import torch

class Convolution:
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad

    def forward(self, x):
        N, C, H, W = x.shape
        FN, C, FH, FW = self.W.shape
        OH = (H + 2 * self.pad - FH + self.stride - 1) // self.stride
        OW = (W + 2 * self.pad - FW + self.stride - 1) // self.stride

        # Use unfold to extract patches
        patches = torch.nn.functional.unfold(x, kernel_size=(FH, FW), stride=self.stride, padding=self.pad)
        col_W = self.W.view(FN, -1).T

        # Perform matrix multiplication and add bias
        out = torch.matmul(patches.transpose(1, 2), col_W) + self.b
        out = out.transpose(1, 2).reshape(N, FN, OH, OW)

        return out
```

- Implementing Pooling Layers  

Pooling layers make using im2col easier like convolution layers. but the differecnce is that the pooling layer uses the maximum or average value of the area.

and Let me explain the pooling layer. The pooling layer is a layer that reduces the size of the input data. 

<div align="center">
    <img src="/images/pooling.png" alt="pool" width="85%">
</div>


Above, the input data is pooled in two dimensions, selecting the maximum value from each row, resulting in an output with dimensions '1 x column'. Finally, the output is reshaped to the original shape. 

Below is the code for the Pooling Layer. 

```python
import torch

class Pooling:
    def __init__(self, pool_h, pool_w, stride=1, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad

    def forward(self, x):
        N, C, H, W = x.shape
        OH = (H + 2 * self.pad - self.pool_h) // self.stride + 1
        OW = (W + 2 * self.pad - self.pool_w) // self.stride + 1

        # Use unfold to extract patches
        patches = torch.nn.functional.unfold(x, kernel_size=(self.pool_h, self.pool_w), stride=self.stride, padding=self.pad)
        
        # Find the maximum value in each patch
        out = patches.view(N, C, self.pool_h * self.pool_w, OH * OW).max(dim=2)[0]
        
        return out.view(N, C, OH, OW)
```

### Befor Implementing CNN

View the 3blue1brown youtube channel informing us about the Convlution. 

- [3blue1brown - Convolution](https://www.youtube.com/watch?v=KuXjwB4LzSA)

This video explains the Convolution operation in a very easy way. 

### Implementing a CNN

Let's implement the CNN with pytorch or tensorflow. 

```python
import torch
import wandb
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# Initialize wandb
wandb.init(name='workshop_1', project="mnist_project-convd")

# MNIST dataset loading (PyTorch)
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# Define CNN model
model = nn.Sequential(
    nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(64 * 14 * 14, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

# Device setup and training preparation
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 32
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Log training loss to wandb
    wandb.log({"epoch": epoch + 1, "loss": running_loss / len(train_loader)})
    print(f"epoch {epoch + 1}, loss: {running_loss / len(train_loader)}")

# Testing loop
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

# Log accuracy to wandb
accuracy = 100 * correct / total
wandb.log({"accuracy": accuracy})
print(f"accuracy: {accuracy:.2f}%")
```

<div align="left">
    <img src="/images/convd.png" alt="cnn" width="40%">
</div>

You can see the results of the CNN modles along with the training and testing process for each epoch. 
The result of accuracy is 99.11%.

#### Explanation:
1. **`nn.Sequential`**: The CNN is defined using `nn.Sequential`, which stacks layers in order without creating a custom class.
2. **Training and Testing**: The training and testing loops remain the same as before.


### Visualizing CNNs

What exactly is a Convlutional layer in a CNN looking at in the imput image?

- **Convolutional Layer**: The convolutional layer applies a filter to the input image, extracting features such as edges, textures, and patterns. The filter slides over the image, performing element-wise multiplication and summing the results to create a feature map.

<div align="center">
    <img src="/images/clear.png" alt="conv_layer" width="70%">
</div>

Above is the image of the fileter layer and after applying sliding window, the image is converted to clear or blurred image. so this technique is used to extract the features of the input image. 


- **Pooling Layer**: The pooling layer reduces the spatial dimensions of the feature map, retaining important features while reducing computational complexity. It typically uses max pooling or average pooling to downsample the feature map.

<div align="center">
    <img src="/images/eight.png" alt="conv_layer" width="70%">
</div>

As deep as layers go, the more complex features are extracted. The first layer might detect edges, the second layer might detect shapes, and deeper layers might detect more complex patterns or objects.

so you can understand that the image becomes clearer and clearer as the layers go deeper. 


# Graph Neural Networks (GNN)

A deep learning has layers that are stacked on top of each other. The more layers, the more complex the features that can be learned.

### What is GNN?

GNN is a type of neural network that operates on graph-structured data. Graphs are a powerful way to represent relationships between entities, and GNNs leverage this structure to learn meaningful representations.

### Preliminary step

- **Graph Representation**: A graph is represented as a set of nodes (vertices) and edges (connections between nodes). Each node can have features, and edges can also have weights or features.

- **Graph Convolution**: GNNs use graph convolutional layers to aggregate information from neighboring nodes. This allows the model to learn node representations based on their local structure and features.

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



