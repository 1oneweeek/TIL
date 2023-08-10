# Neural Ordinary Differential Equations

## 1. Differential Equations: 미분방정식
종속변수(y)를 독립변수(x)에 대해 미분한 도함수를 포함하는 방정식
상미분 방정식(Ordinary Differential Equation): 단변수, 변수가 하나 
편미분 방정식(Partial Differential Equation): 다변수. 변수가 둘 이상

기존 방정식은 해가 스칼라, 상수 값이지만 미분 방정식의 해는 함수, Original function(적분)

## 2. Euler method
ODE의 solver, 즉 미분 방정식을 푸는 방법 = original function을 찾는, 근사하는 방법(적분의 과정).
non-linear한 original function을 step마다 적분하여 approximation of function을 진행함. 근사성능은 step을 작게 가져갈수록 높일 수 있음.
NODE는 Euler method를 적분의 무수한 더하기로 보는 모델임.

### Residual connection
Residual conncection: Euler method in discrete transformation, 즉 적분한 것들 별개의 합
Neural ODE: Euler method in continuous transformation, 적분의 무수한 더하기, 연속적이고 연결적인 적분의 합
즉 Residual connection은 본질적으로 Neural ODE와 같음.
supervised-learning에서 차이점을 살펴보기 위한 실험을 진행함.

### Custom forward, backward
1. Custom Forward: Euler method
forward: a(1)을 얻기 위해 a(0)에서 시작함
2. Custom Backward: Adjoint Sensitivity method
- 본질적으로 Euler method(오일러)와 같은 과정임
- 각 state 별 Gradient를 Adjoint state, a(t)를 정의
backward: a(0)을 얻기 위해 a(1)에서 시작함
forward와 과정은 같고 방향만 반대임.

## 3. Benefits of Neural ODE

### 1. Model efficiency(Low memory cost)
- Default backpropagation(기존 역전파방법)
: Forward pass는 무수한 더하기 연산, 그 더하기 연산의 역전파는 간단하나 모든 gradient를 기억해야 하기 때문에 memory 효율이 없음.
- Adjoint Sensitivity Method(Customized)
: 초기값 a(1)이 주어진 새로운 ODE solve 과정으로, 기억해야할 것은 초기값 a(1)뿐이고 Adjoint state에서 ODE는 이미 정해져있기에 memory 효율이 높음.

### 2. Adaptive computation

### 3. Replacing Residual network
-ResNet과의 비교
1. ODE-Net으로 ResNet의 Test Error performance 구현
2. ODE-Net의 parameter 수가 ResNet의 1/3 수준
3. Memory cost는 ODE-Net이 O(1)로, 상수의 복잡도를 구현
4. Time(소요시간)은 ResNet이 # Layers에 비례하지만, ODE-Net은 # function evaluations에 비례

### 4. Applications: In generative model
지도학습 외 generative model에서도 활용되는 Neural ODE

1. Normalizing flow: VAE와 같은 생성 모델, 연산량이 많음
Discrete transformation의 Determinant of Jacobian이 연산량 과다의 원인 -> 이를 continuous transformation으로 바꿔주어 Trace of Jacobian(대각합, linear)으로 연산하면 연산량이 많이 줄어들어 모델의 부담이 덜함.
-> CNF(Continuos Normalizing Flow)가 NF 대비 더 낮은 Loss 구현

2. 불규칙한 시계열 데이터 모델(irregular time points)
irregular한 input(ex: medical recrods, network traffic, etc..)이 주어져도 ODE를 활용한 생성모델은 정확한 계산이 가능함.

즉 Neural ODE는 continuous 한 과정이기 때문에 불규칙한 input이더라도 잘 받아낼 수 있고, 불규칙한 input들을 통해서도 original function을 근사해나갈 수 있기에 잠재함수를 잘 찾아낼 수 있다는 것이 핵심.

[enter link description here](https://www.youtube.com/watch?v=UegW1cIRee4)
> Written with [StackEdit](https://stackedit.io/).
