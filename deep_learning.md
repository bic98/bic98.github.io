---
# Page settings
layout: default
keywords:
comments: true

# Hero section
title: Deep Learning Basics
description: A collection of deep learning basics.

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
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

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

```python
import numpy as np

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
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
        
        # If t is one-hot encoded
        if t.size == y.size:
            return -np.sum(t * np.log(y + 1e-7)) / y.shape[0]
        # If t is label encoded
        else:
            return -np.sum(np.log(y[np.arange(y.shape[0]), t] + 1e-7)) / y.shape[0]

    def loss(self, x, t):
        y = self.predict(x)
        
        return _cross_entropy_error(y, t)

    def _numerical_gradient(self, f, x):
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

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)
        
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        
        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        
        return grads
```
- Implementing a Two-Layer Neural Network Class

- Implementing Mini-Batch Learning
- Evaluating with Test Data

---
