# LSTM(Long Short Term Memory)

## 1. 구조
- RNN보다 더 복잡한 계산 구조를 가짐
- hidden state와 cell state가 있고, Forget gate, Input gate, Output gate를 통해 계산이 이루어짐

### 1. Cell state
- 이전 시점의 cell state를 다음 시점으로 넘겨주고 gate들과 함께 작용하여 정보를 선택적으로 활용할 수 있도록 함.
- cell state의 업데이트는 각 gate의 결과를 더함으로써 진행하는데 이는 시퀀스가 길더라도 gradient, 즉 오차를 상대적으로 잘 전파할 수 있음.

### 2. Forget Gate Layer
- 과거의 정보를 버릴지 말지 결정하는 게이트
- Sigmoid 함수를 이용해 0~1 사이의 값을 출력하며 0에 가까울수록 많은 정보를 잊은 것이고, 1에 가까울수록 많은 정보를 기억하는 것

### 3. Input Gate Layer
- 새로운 정보, 즉 현재의 정보가 cell state에 저장이 될지 말지 결정하는 게이트

### 4. Output Gate Layer
- 상태를 바탕으로 Sigmoid 층에 input 데이터를 넣어 상태의 어느 부분을 출력으로 내보낼지 결정하는 게이트

### 5. Update Cell State
- 과거에서 유지할 정보 + 현재에서 유지할 정보를 통해 현재 시점의 cell state를 update함


# GRU(Gated Recurrent Unit)

-기존 LSTM의 구조를 조금 더 간단하게 개선한 모델
-Reset gate, update gate 2개의 gate만을 사용
-cell state와 hidden state가 합쳐져 하나의 hidden state로 표현
-마지막 출력값에 활성화함수를 적용하지 않음
-학습할 파라미터가 더 적은 장점

## 1. 구조

### 1. Reset Gate
- 이전 시점의 hidden state와 현 시점의 x를 sigmoid 함수를 적용하여 구하는 방식으로 결과값은 0~1 사이를 가지고, 이전 hidden state의 값을 얼마나 활용할 것인지에 대한 정보

### 2. Update Gate
- LSTM의 input, forget gate와 비슷한 역할
- 과거와 현재의 정보를 각각 얼마나 반영할지에 대한 비율을 구하는 것이 핵심
- z: 현재 정보를 얼마나 사용할 것인가(=input)
- 1-z: 과거 정보를 얼마나 사용할 것인가(=forget)
- 
> Written with [StackEdit](https://stackedit.io/).
