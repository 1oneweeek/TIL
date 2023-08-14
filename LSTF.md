# Long Sequence time-series forecasting

## Long Sequence time-series data
Time-series forecasting은 Sensor network monitoring(Papadimitriou and Yu 2006), energy and smart grid management, economiocs and finance(Zhu and Shasha 2002), disease propagation analysis(Matsubara et al. 2014)과 같은 많은 영역에서 중요한 요소이다. 이러한 시나리오에서 우리는 보다 더 long sequence한 예측을 위해 과거 행동에 대한 상당한 양의 시계열 데이터를 활용할 수 있다. 이를 LSTF(Long Sequence time-series forecasting)이라 한다.  
오늘은 LSTF를 위해 사용되는 여러 딥러닝 모델들을 소개하도록 하겠다.

## 1. RNN-based Models
- 시계열 데이터를 사용한 딥러닝 모델 중 가장 고전적인 접근 방식.
- 현재 시점 데이터는 이전 시점 데이터의 영향을 받는 시계열 데이터의 특성상 RNN 동작과 유사함.
- Vanilla RNN의 경우, 깊은 신경망을 학습시키는 과정에서 기울기가 소실되는 문제인 vanishing gradient 문제가 있음.
- 만약 시점 간의 간격(gap)이 커진다면 현재 시점으로부터 멀리 떨어진 과거 시점 정보의 영향력이 약해지는 장기 의존성 문제 (Long-Term Dependency Problem)가 있음.

## 2. CNN-based Models
- 컴퓨터비전 분야에서 사용된 CNN 기반 모델이 신호 및 시계열 데이터 모델링에도 응용됨 (e.g. ConvTimeNet, WaveNet) 
- 시간 순서에 따라 filter를 적용함으로써 과거와 현재 정보의 관계를 파악함.
- CNN을 시계열 데이터에 적용할 때의 문제점 
: CNN에서는 과거와 현재 시점 정보가 time-invariant하다고 가정하고 시간에 따라 동일한 filter를 적용함.
좋은 성능을 내기 위해 여러 실험을 거쳐 Receptive field 크기(filter 크기)를 직접 튜닝해주어야 함.
filter 단위로 local한 dependency를 파악하기 때문에 장기 의존성 문제가 존재함.

## 3. Attention-based Models, Transformer
- Attention mechanism에 기반한 모델로, 대표적으로 자연어처리 분야의 SOTA 모델의 근간이 됨.
- Transformer는 서로 다른 시점의 정보들 간의 관계를 바탕으로 attention score를 부여하여 활용하고자 함. 
- 현재, Attention 기반의 모델, 특히 Transformer 기반의 모델들이 RNN 기반의 모델 성능을 뛰어넘고 있음.
- Transformer 기반 모델은 self-attention을 바탕으로 각 시점 정보 간의 관계를 모델링 하는데, CNN 또는 RNN 대비 long range dependency를 학습하기 용이하다는 장점이 있음.

## 4. Sparse Transformer, LogSparse Transformer, Longformer
- Sparse Attention: Attention에 Sparsity Bias(Structual Bias)를 추가하여 Complexity 감소

1. Sparse Matrix
- 행렬 대부분의 값이 0이고, quary와 key간의 attention scare이 높게 잡히는 곳은 소수이며 대부분의 행렬 값은 0에 수렴함을 지칭하는 표현. ( <->  Dense Matrix)
- Self-attention의 Sparsity: Self-attention 학습 후에는 Sparse Attention에 의해 우리가 주목해야 할 대상만 유효하게 사용이 되고 0에 수렴하는 attention 값은 여전히 계산이 되지만 어떠한 representation을 추출하는데 아무런 영향을 주지 않음. 즉 이는 memory 측면에서 비효율적인 낭비임.
- 이를 해결하기 위해 애초에 Structual Bias(하나의 기준, 가정)를 부여하여 Query-Key pair 개수를 제한함.

2. Sparse Attention을 적용한 논문들의 공통 주장
	- Computational Cost: 문장이 길어질수록, Attention에 필요한 비용이 큼.
	: Input Length에 Quadratic하게 증가하는 비용 대신에, Linear하게 증가하도록 만드는 것이 목적 (n2 -> n)
	- Long Range Dependency: 긴 문장에는 Cost 문제가 있어 Transformer를 적용하기 힘듦.
	:  Global Attention을 통해 Long Range Dependency를 해소

3. Sparse Attention의 효과
- 비용과 Dependency를 감소시키면 512 Token보다 더 긴 문장을 input으로 활용할 수 있음.
- Input 문장을 길게 만들면, Downstream Task에 사용되는 단서가 많아지고, 시계열 예측의 관점에서는 더 긴 sequence를 받아서 더 긴 sequence를 예측할 수 있게 됨.

4. Sparse Attention 예시
- LogSparse Self-attention 방법론: 한 시점을 예측하기 위해 특정한 규칙, 순서를 적용하여 token을 뽑아 attention score를 계산하는 것
즉 일정 기준에 따라서 일부의 query-key pair를 가지고 attention score를 계산하여 memory와 연산 효율을 높이고자함.

## 5. Reformer
딥러닝 모델에서 memory를 증가시키는 요인을 제거하는 transformer 기반의 모델로
Transformer 모델에 효율적인 Memory 사용 방법을 적용한 모델.

1. Memory 사용량에 영향을 미치는 요인
- 딥러닝 모델의 memory는 역전파 하기 전까지 중간결과물(bn)을 저장해야함.
- Batch Update를 사용 시 Batch 크기가 메모리 사용량에 영향을 미침.
- 모델깊이, 모델넓이, 문장길이가 메모리 사용량에 영향을 미침.
- > Memory에 저장되는 중간결과물을 줄임.
ex: 필요없는 연결 제거 / 역전파전까지 중간결과물을 저장해야하는 메모리 저장 구조를 변경
/ 독립적, 단계적으로 메모리를 사용하여 전체 메모리 사용량을 줄임.

2. Attention Layer 개선: LSH Attention
- Multi-head Attention을 LSH Attention으로 변경
- 모든 입력에 대해 Attention을 계산하는데 많은 메모리 사용이 발생하는데, 사실 Attention은 몇 개의 토큰에만 집중하고 있음. 이를 이용하여 비슷한 토큰끼리만 Attention을 선택적으로 적용하는 Local Sensitive Hashing(LSH)을 도입함.
- LSH: Local Sensitive Hashing, 임의의 데이터를 길이가 고정된 값으로 변환하는 것을 의미, 즉 가까운 Point는 가까운 Hash 값을 가지도록 Hashing하는 방법
- LSH Attention 적용
	 - 1. 각 토큰의 (Key, Query), Value 생성: Key와 Query는 동일한 Liner Layer에서 생성
	 -> LSH를 적용하기 위하여 Key, Query를 같은 공간에 Projection
	 - 2. Locality-Sensitive Hashing 적용: 각 token에 hashing index가 부여되어 이미지에서 색깔로 구분됨
	 - 3. Index가 같은 것끼리 Sorting
	 - 4. Chunking Sequence: 동일한 index를 가진 token이 많아도 특정 길이 이상의 attention을 하면 memory가 증가하므로 문장 길이에 Robust, 고정된 크기로 sequence를 분절
	 - 5. Attention 적용: 2가지 조건을 충족하는 key에 Attention 적용

3. Residual Block: Residual Block을 Reversible Network로 변경
- Residual Block 구조의 특징: attention과 feed forward layer가 residual connection 구조로 되어 있고 반복됨.
역전파 하기 위해서 중간결과물을 저장해야 하므로 residual block의 개수가 증가할수록 memory가 증가함.
-> Residual Block의 구조를 변경함으로써 역전파하기 위해 중간결과물을 저장해야하는 구조를 변경함
-> 즉 출력층 변수로 입력층 변수를 계산할 수 있는 Residual Connection 구조를 적용하여 더 이상 residual block의 개수에 따라 memory가 증가하는 것을 방지함.
- Reversible Network: Residual Connection 구조를 적용한 방법
: 출력값으로 입력값을 계산할 수 있는 구조로 중간 결과값을 계산하지 않아도 되므로 memory 사용량을 줄일 수 있음.

4. Feedforward Layer 개선: Feedforward를 일정 단위로 잘라서(Chunking) 계산
- 계산하는 단위를 나누어 메모리를 사용할 수 있는 Chunking 적용
- attention과 달리 각 token에 대하여 sequence의 위치와 상관없이 독립적으로 계산이 가능함.
: attention은 문장 위치를 고려하여 계산해야 하므로 한꺼번에 memory에 올려 계산해야함.
feedforward layer는 attention을 계산한 뒤에는 각 token에 독립적이므로 각자 계산이 가능함.

* Reformer의 성능은 Transformer와 비슷하거나 살짝 낮은 정도
-> 일반적인 길이에서는 pre-trained 모델을 사용하는 것을 추천
* Reversible Network는 속도 측면에서 단점이 크게 부각
-> 메모리 문제가 있다면 apex를 활용하여 Transformer 모델을 사용하는 것을 추천
* 긴 길이의 Task에 Attention을 적용할 때는 LSH-Attention을 고려해보는 것을 권유
-> 상품추천, 로그분석 등

## 6. Informer
트랜스포머는 quadratic한 time complexity를 가지고, high memory를 사용하고, encoder-decoder architecture에서 오는 근본적인 한계가 존재함. 이를 보완해서 나온 트랜스포머 기반 모델이 인포머임.
인포머는 분리하여 다루도록 하겠음.

> Written with [StackEdit](https://stackedit.io/).
