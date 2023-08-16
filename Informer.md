# Informer

# 서론

## 1. 시계열 예측과 관련된 연구의 대표적인 문제

### 1) LSTF의 주요 challenge
- long sequence (input/target)에 대한 예측 성능을 향상 시키는 것으로, 
이를 위해 필요한 것에는 두가지가 있다.
1. long-range alignment ability
2. efficient operations on long sequence input/output

### 2) LSTF 관점에서 Transformer의 장점과 한계점
-Transformer 구조는 1번 조건에 대해서는 어느정도 타 네트워크 대비 우수하지만,
Transformer 모델은 Recurrent나 Convolution 구조를 사용하지 않고 Attention mechanism 만을 사용하는 구조이므로 RNN 모델 대비 long-range dependency를 잘 포착할 수 있음.
- 하지만 self-attention mechanism은 2번 조건을 만족하지 못함.
	- L-length를 가진 input/output에 대해서 L-quadratic(L의제곱)한 연산과 메모리 사용이 불가피하기 때문임.
	- L-quadartic 연산은 짧은 길이의 데이터에 대한 연산에서는 어느정도 용이할 수 있지만, long sequence 데이터에 대해서는 bottleneck으로 작용하기 때문임.

### 3) LSTF를 수행하기 위한 Transformer 활용 방안
1. long-range dependency의 장점을 유지
2. self-attention mechanism의 연산, 메모리, 속도 측면의 효율성 개선
3. 예측 능력 향상

## 2. Self-attention Complexity

### 1) Complexity
Self-Attention의 Complexity(O(T2*D))는 문장길이(T) 길어지면 Bottleneck이 된다.
이 때 연산복잡도는 quadartic한 속도로 증가하게 됨.

### 2) Structural Prior
Input에 대한 어떤 Structual Bias가 없음.
Input이 단순한 sequence 형태로 들어오기만 하면 self-attention 계산이 가능함.
이는 모든 input sequence token들에 대한 모든 self-attention 값을 계산해야 하기 때문에 비효율적이라는 단점이 있음.

## 3. Dynamic decoding

- Informer 논문에서 사용되는 용어로, RNN과 같은 autogressive한 step-by-step의 decoding 방식을 의미함.
- Step-by-step의 decoding은 느린 inference 속도를 가져 cumulative한 error를 불러올 수 있음.
- Informer는 이러한 문제를 피하고자 one forward step을 적용함.

## 4. Transformer의 한계점 요약

### 1) Self attention 측면
1. Self-attention의 quadratic한 연산량(o(L2))
2. 긴 input 길이에 대한 stacking layer 구조는 memory bottleneck을 발생

### 2) Decoder 측면
3. Step-by-step decoding 방식의 느린 inference 속도

- 현재 한계점 1번을 다루는 모델은 많지만(Sparse Transformer)
한계점 2번과 3번은 아직 해결되지 못함.

## 5. Informer가 제안한 해결책

1. ProbSparse self-attention mechanism
- 시간 복잡도와 메모리 사용량을 O(LlogL)로 줄임.
2. Self-attention distilling operation
- 주요 feature를 추출하기 위한 stacking layer의 총 공간 복잡도를 감소시켜 long sequence input을 받아들이는 데 용이하게 함.
3. Generative style decoder: one forward step
- 오직 하나의 forward step만으로 long sequence output을 얻을 수 있도록 하는 generative style decoder를 제안함. 이를 통해 step-by-step inference 과정에서 발생할 수 있는 cumulative error(이전 시점의 잘못된 output이 미래 시점에도 계속해서 누적되어 영향을 미치게 되는 현상)를 방지하고자 함.

## 6. Informer가 정의한 LSTF

- Input : 𝑋𝑡 = 𝒙1 𝑡 , … , 𝒙𝐿𝑥 𝑡 𝒙𝑖 𝑡 ∈ 𝑅 𝑑𝑥 
- Output : 𝑌𝑡 = {𝒚1 𝑡 , … , 𝒚𝐿𝑦 𝑡 ∣ 𝒚𝑖 𝑡 ∈ 𝑅 𝑑𝑦} 
- LSTF 문제는 위 Output length의 길이 𝐿𝑦가 선행 연구(~48)들 보다 길 때로 정의하고 있음. 
- 또한, feature dimension의 크기가 단변량에 제한되지 않음 (𝑑𝑦 ≥ 1)

## 7. Informer의 Input 구조

1) Encoder Input
- Uniform input representation
- Global position context 정보와 Local temporal context 정보를 잘 반영할 수 있도록 함.
- Scalar는 input을 d model 차원으로 projection 시킨 값
- Local Time Stamp: 일반적인 Transformer의 Positional Encoding 방식으로 fixed(고정 위치값)을 사용
- Global Time Stamp: 사용자에 따라서 주단위, 월단위, 공휴일단위에 대한 정보를 가지고 있는 것으로
직접 time feature 정보를 구축하여 사용함. 즉 학습 가능한 embedding을 사용

2) Decoder Input
- Start token + 예측하고자 하는 부분(padding)
- Start token이라고 칭하는 일정 길이의 encoder input 데이터를 넣어주고, 예측해야 하는 부분의 값은 0으로 padding하여 구성함. 
- Start token: encoder 뒷부분의 일정 길이만큼을 떼서 decoder input으로 사용하는 부분


# 본론

## 1. ProbSparse Attention

 - 앞서 제안된 다양한 선행연구의 Sparse Attention은 heuristic 방법론을 따르고 있으며,
 multi-head self attention에서 각 head 모두 동일한 선택 전략을 적용하고 있다는 점에서 이론적으로 한계가 있음을 지적
 - 이 논문에서 말하는 heuristic한 방법론이란, query나 key를 선택할 때 잠재된 어떠한 실질적 의미/역할을 고려하는 것이 아니라 단순히 random하게 혹은 일정 window 구간으로 한정하는 방식 등을 사용한 것을 의미

1) Idea 제안 배경
- Sparsity한 self-attention은 꼬리가 긴 분포를 가짐
- 이는 소수의 dot-product pairs만이 주요 attention에 기여하고 다른 dot-product pairs는 trivals attention, 즉 영향력이 낮은 attention을 생성한다는 것을 의미

2) 핵심 idea: 어떻게 유의미한 dot-product pairs를 구분해 낼 것인가?
- Dominant dot-product pairs, 유의미한 query-key pairs의 분포는 uniform 분포로부터 상이함.
- 만약 pairs의 분포가 uniform(lazy) 분포에 근사한다면, self-attention score로 value V를 가중합 할 때 trival한 영향을 주게 되고, 상대적으로 불필요한 query로 작용하게 됨.
- 따라서 p분포와 q분포의 유사도는 중요한 query를 구분해내는 지표로 사용될 수 있음.

3) ProbSparse self-attention
- Sparse 앞에 Prob가 붙은 이유: 확률 분포를 바탕으로 attention을 계산하기 때문에
- Sparsity 측정 지표를 바탕으로 유의미한 query를 top-u개만 선택하여 attention을 계산하는 방법
-  = c(하이퍼파라미터갯수)*lnLQ

4) ProbSparse self-attention 구현
- Step 1: query 중요도 계산(M바)
- Step 2: query 중요도를 기준으로 sorting 하여 top-u개의 query를 선택
- Step 3: top-u개의 query를 바탕으로 Q바 메트릭스를 구축하고, 모든 key set에 대해 attention을 구함.
- 즉 최종적으로  Q바 메트릭스와 기존에 sampling 되지 않은 key metrix의 내적이 진행됨.

*key sampling에 문제가 없을까?
- Proposition 1 : 무수히 많은 이론, 정리, 통계적 기법 등을 통해 증명해 낸 결과, 수식 M바가 M을 대표할 수 있음.
- 즉 sampling된 key를 바탕으로 구해진 query 중요도 순서가 모든 key와의 sparsity measurement 계산을 통해 구해진 query 중요도 순서와 일치한다는 것을 보장하는 역할
- 다시 말해, 잘못 sampling 된 key를 바탕으로 sparsity measurement를 계산하면, query 중요도 순서가 달라질 가능성이 있는데, 본 논문에서는 Proposition 1 에 대한 경험적, 이론적 증명을 통해 제안된 approximation 수식이 유효함을 보이고 있음.

5) ProbSparse self-attention 아이디어 흐름 요약
- 1. 모든 query-key 내적은 비효율적이므로 일부 pair만 골라보자 -> Sparse Attention
- 2. 확률 분포를 바탕으로 query 중요도(sparsity 정도)를 측정하여 실직적으로 유의미한 것을 골라내자 -> M 수식 등장
- 3. 근본적으로 M 계산을 효율적으로 해야함 -> Key sampling과 max operator를 사용한 근사식 M바를 쓰자 -> 근거: Lemma 1과 Proposition 1

6) ProbSparse self-attention 정리
- 타 Sparse Attention 방법론 대비, 보다 효율적이며 실질적으로 유의미한 query로만 attention을 계산하는 방법론을 제안함.
- Attention을 확률 분포의 관점에서 접근함 (ProbSparse에서 Prob의 의미로 해석될 수 있음)
- Query 중요도인 M바 계산을 위해 필요한 연산과, 이를 통해 뽑힌 top-u개의 query를 기반으로 self-attention을 수행하는 연산이 모두 𝑂(𝐿𝑙𝑜𝑔𝐿)의 복잡도를 갖게 됨.  * 복잡: 𝑂(𝐿𝑙𝑜𝑔𝐿) ≪ 𝑂(𝐿2)

## 2. Encoder
long sequence input에 대한 메모리 효율을 높이는 방향으로 설계됨.

1) Encoder embedding 구성
- Scalar embedding과 Stamp embedding을 더하여 최종 embedding 구성

2) Self-attention distilling
- encoder에 들어온 input을 바탕으로 ProbSparse self-attention을 수행하고, 여기서부터 나온 Attention output을 Convolution과 max-pooling을 통해 distilling을 수행
- Attention output으로부터 주요한 정보만을 추출하여 다음 layer에 전달할 정보를 구성하기 위함.

(1) Multi-head-self-attention
- 각 head 별 query, key, value는 64 dimension(𝑑 ′ (= 𝑑/8))을 갖도록 projection 됨. 
- 그 후 query [96, 64] 와 key [96, 64]에 대해 ProbSparse Attention 수행함.
- ProbSparse Attention의 결과로 생긴 Attention output [Top-𝑢, 64(𝑑 ′ )]을 바탕으로 Top-𝑢 개의 index를 저장해 두었다가 [𝐿𝑄(96), 64(𝑑 ′ )] 차원으로 맞춤.
- 차원을 맞춰주는 이유: 차원을 동일하게 유지하여 head 별로 다양하게 뽑힌 query attention 정보들은 concatenate하기 위해서 multi-head-self-attention을 활용함.
-Concatenate 된 8개의 output은 transformer 모델처럼 다시 d모델의 차원으로 projection 되어서 96*512라는 최종 output이 생성됨. 

(2) 1D-Convolution과 Max-pooling
- (1)의 결과로 생성된 output을 1D-Convolution과 Max-pooling하는 distilling이 수행됨.
- [96, 512] 차원의 Multi-head self-attention output을 1D-Convolution과 1D-Maxpooling을 거쳐 절반의 길이인 [48, 512] 차원을 갖도록 함.
- 이 과정을 매 layer마다 반복하여 다음 layer(j+1)에서의 input은 이전 layer(j)의 input의 절반의 길이를 가질 수 있도록 동일하게 ProbSparse self-attention과 distilling 과정을 반복함.

(3) Self-attention distilling의 효과
- 총 메모리 사용량을 (2-e)LlogL로 감소시키는 효과 -> J개의 stacking layer에 대한 memory bottleneck 문제 해결
- self-attention computation complexity에 해당하는 메모리 LlogL이 2-e배만큼 필요하다는 것. (등비수열)

3) Stacking layer replicas (= Multi-Encoder)
- Robustness의 향상을 위해서 원래 input의 절반의 길이만큼만 input으로 받는 복제된 encoder를 추가로 구성
- 각 encoder 별로 생성되는 output feature map의 dimension을 맞추기 위해 self-attention distilling layer 수를 하나씩 점진적으로 감소
- 여러 개로 stack된 encoder를 거친 output들은 concatenate되어 최종적으로 encoder의 hidden represention을 구성
- 최근 시점을 기준으로 input sequence를 절반씩 잘라서 사용


## 3. Decoder
- Decoding 과정이 하나의 forward step으로 이루어질 수 있도록 마지막에 FC layer를 추가함.
- Transformer와 달리 Decoder에서 masked self-attention은 encoder처럼 ProbSparse self-attention을 사용하고, Encoder-decoder attention은 Vanila Transformer와 동일한 attention을 사용함.
- Attention을 통해 출력된 output은 FC layer을 거쳐 최종 output을 예측 (단변량/다변량)
- Loss는 true 값과 prediction 값의 차이인 MSE Loss를 사용



[enter link description here](https://youtu.be/Lb4E-RAaHTs)
> Written with [StackEdit](https://stackedit.io/).
