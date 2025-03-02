---
# Page settings
layout: default
keywords:
comments: false

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
page_nav:
    prev:
        content: Previous page
        url: '/'
    next:
        content: Next page
        url: '/reinforce/'

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
### Notation Explanation
### Implementing Signal Transmission in Each Layer
### Implementation Summary

## Designing the Output Layer
### Implementing Identity and Softmax Functions
### Considerations When Implementing the Softmax Function
### Characteristics of the Softmax Function
### Determining the Number of Neurons in the Output Layer

## Handwritten Digit Recognition
### MNIST Dataset
### Inference Processing in Neural Networks
### Batch Processing

---
