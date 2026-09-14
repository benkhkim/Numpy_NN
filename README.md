# XOR 신경망: NumPy vs TensorFlow(Keras) 구현 비교

간단한 2층 신경망으로 XOR 문제를 풀어보면서, NumPy로 직접 구현한 코드와 TensorFlow(Keras)로 구현한 코드가 내부적으로 어떻게 대응되는지 정리한 문서입니다.

---

## 1. 문제 정의: XOR

XOR(배타적 논리합)은 다음과 같은 입출력 관계를 가집니다.

| x1 | x2 | y (정답) |
|:--:|:--:|:--------:|
| 0  | 0  | 0 |
| 0  | 1  | 1 |
| 1  | 0  | 1 |
| 1  | 1  | 0 |

이 4개 점을 2차원 평면에 찍으면, 정답이 0인 점(대각선 방향)과 1인 점(반대 대각선 방향)이 서로 교차하는 형태로 배치됩니다.

```
      x2
      |
  (0,1)●        ●(1,1)
   y=1 |         y=0
       |
       |
  (0,0)○--------○(1,0)
   y=0 |         y=1
       +------------- x1
```

**핵심 사실**: 직선 하나로는 ○ 두 점과 ● 두 점을 절대 분리할 수 없습니다. 이를 **선형분리 불가능(linearly inseparable)** 문제라고 부르며, 은닉층(hidden layer)이 있는 다층 신경망이 필요한 이유가 여기서 나옵니다.

---

## 2. 신경망 구조

| 항목 | 값 |
|---|---|
| 입력층 뉴런 수 | 2 |
| 은닉층 뉴런 수 | 4 |
| 출력층 뉴런 수 | 1 |
| 활성화 함수 | Sigmoid (모든 층) |
| 손실 함수 | MSE (Mean Squared Error) |
| 최적화 알고리즘 | SGD (Stochastic Gradient Descent) |
| 학습률 (learning rate) | 0.5 |
| 학습 반복 횟수 (epochs) | 10,000 |

---

## 3. NumPy 구현 (전체 코드)

라이브러리 없이 forward pass, loss 계산, backward pass(역전파), 파라미터 업데이트를 모두 직접 구현합니다.

```python
import numpy as np


```

### 3.1 코드 주석 상세 설명

| 코드 | 수식/의미 | 비고 |
|---|---|---|
| `z1 = X @ W1 + b1` | 입력의 선형결합 (pre-activation) | `@`는 행렬곱 연산자 |
| `a1 = sigmoid(z1)` | 은닉층 출력 (post-activation) | 비선형성 부여 |
| `z2 = a1 @ W2 + b2` | 은닉층 출력의 선형결합 | 출력층 pre-activation |
| `a2 = sigmoid(z2)` | 최종 예측값 | 0~1 사이 확률 형태 |
| `loss = np.mean((y-a2)**2)` | MSE 손실함수 | 예측과 정답의 오차 |
| `d_a2 = (a2 - y)` | ∂Loss/∂a2 | 손실을 출력에 대해 미분 |
| `d_z2 = d_a2 * sigmoid_derivative(a2)` | ∂Loss/∂z2 (체인룰) | 활성화 함수 미분 반영 |
| `d_W2 = a1.T @ d_z2` | ∂Loss/∂W2 | forward의 `a1 @ W2`가 전치되어 뒤집힘 |
| `d_a1 = d_z2 @ W2.T` | 오차를 은닉층으로 역전파 | "역전파"라는 이름의 유래 |
| `d_W1 = X.T @ d_z1` | ∂Loss/∂W1 | 입력층 방향 gradient |
| `W -= lr * d_W` | 경사하강법 업데이트 | gradient 반대 방향으로 이동 |

### 3.2 실행 결과

| epoch | loss |
|:-----:|:------:|
| 0     | 0.2557 |
| 2000  | 0.0052 |
| 4000  | 0.0010 |
| 6000  | 0.0005 |
| 8000  | 0.0004 |

**최종 예측값**

| 입력 | 예측값 | 정답 | 일치 여부 |
|:----:|:------:|:----:|:---------:|
| [0,0] | 0.019 | 0 | ✓ |
| [0,1] | 0.984 | 1 | ✓ |
| [1,0] | 0.984 | 1 | ✓ |
| [1,1] | 0.015 | 0 | ✓ |

---

## 4. TensorFlow(Keras) 구현 (전체 코드)

같은 구조를 고수준 API로 구현하면 코드가 크게 줄어듭니다.

```python
import tensorflow as tf
import numpy as np


```

### 4.1 실행 결과

| 입력 | 예측값 | 정답 | 일치 여부 |
|:----:|:------:|:----:|:---------:|
| [0,0] | 0.0156 | 0 | ✓ |
| [0,1] | 0.9764 | 1 | ✓ |
| [1,0] | 0.9753 | 1 | ✓ |
| [1,1] | 0.0303 | 0 | ✓ |

---

## 5. NumPy ↔ Keras 대응표

라이브러리가 무엇을 "자동화"했는지 한눈에 보는 표입니다.

| NumPy에서 직접 구현한 것 | Keras에서 대체하는 것 | 설명 |
|---|---|---|
| `W1 = np.random.randn(2,4)*0.5` 등 | `Dense(4, input_shape=(2,))` | 가중치 자동 생성 (초기화 방식은 다름) |
| `sigmoid(x)` 함수 직접 작성 | `activation='sigmoid'` | 동일한 활성화 함수를 문자열로 지정 |
| `z1=X@W1+b1; a1=sigmoid(z1)` | Dense 레이어 1개 | forward pass 한 층 |
| `loss = np.mean((y-a2)**2)` | `loss='mse'` | 동일한 손실함수 |
| `d_W1, d_W2, ...` 체인룰 직접 계산 | 자동미분 (`GradientTape` 내부 동작) | 역전파 자동화 |
| `W -= lr * d_W` | `SGD(learning_rate=0.5)` | 동일한 업데이트 규칙 |
| `for epoch in range(10000): ...` | `model.fit(..., epochs=10000)` | 학습 반복 자동화 |

### 5.1 `model.fit()` 내부에서 실제로 벌어지는 일 (의사코드)

```python
with tf.GradientTape() as tape:      # 연산을 "테이프"에 기록 시작
    predictions = model(X)            # forward pass
    loss = mse(y, predictions)        # loss 계산

gradients = tape.gradient(loss, model.trainable_variables)
# 테이프에 기록된 연산을 역방향으로 훑으며 체인룰 자동 적용
# (NumPy의 d_a2, d_z2, d_W2, d_a1, d_z1, d_W1 전체에 해당)

optimizer.apply_gradients(zip(gradients, model.trainable_variables))
# W -= lr * d_W 를 모든 레이어에 대해 자동 수행
```

---

## 6. 결과가 완전히 일치하지 않는 이유

동일한 알고리즘이라도 NumPy와 Keras의 결과값이 미세하게 다른 이유는 다음과 같습니다.

| 원인 | NumPy | Keras (기본값) |
|---|---|---|
| 가중치 초기화 방식 | `np.random.randn * 0.5` (직접 지정) | Glorot(Xavier) 초기화 |
| 데이터 정밀도 | float64 | float32 |
| 부동소수점 연산 순서 | 직접 작성한 순서대로 | 내부 최적화된 순서 (BLAS 등) |
| 배치 처리 방식 | 항상 full-batch | 기본 batch_size=32 (여기선 사실상 전체) |

### 6.1 결과를 최대한 일치시키는 방법

두 구현을 정확히 같은 초기 가중치에서 출발시키면 결과를 훨씬 가깝게 맞출 수 있습니다.

```python
# 1) NumPy에서 초기 가중치를 명시적으로 생성해 보관
np.random.seed(42)
W1_init = np.random.randn(2, 4) * 0.5
b1_init = np.zeros((1, 4))
W2_init = np.random.randn(4, 1) * 0.5
b2_init = np.zeros((1, 1))

# 2) 같은 초기값을 Keras 레이어에 그대로 주입
model.build(input_shape=(None, 2))
model.layers[0].set_weights([W1_init.astype(np.float32), b1_init.flatten().astype(np.float32)])
model.layers[1].set_weights([W2_init.astype(np.float32), b2_init.flatten().astype(np.float32)])

# 3) 학습 조건도 동일하게 맞춤 (full-batch, shuffle 없음)
model.fit(X, y, epochs=10000, batch_size=4, shuffle=False, verbose=0)
```

| 맞춰야 할 조건 | 방법 |
|---|---|
| 초기 가중치 | `set_weights()`로 NumPy에서 만든 값을 그대로 주입 |
| 배치 크기 | `batch_size=4` (전체 데이터 크기와 동일하게) |
| 데이터 순서 | `shuffle=False` |
| 정밀도 | 양쪽 모두 동일한 dtype(float32 또는 float64)으로 통일 |

이렇게 맞춰도 부동소수점 연산 순서 차이로 인해 **bit-level로 100% 동일한 값**은 보장되지 않을 수 있습니다. 완전한 결정론적 결과를 원한다면 `tf.config.experimental.enable_op_determinism()`과 같은 설정이 추가로 필요합니다.

---

## 7. 핵심 정리

- 왜 은닉층이 필요한가?: XOR은 선형분리 불가능한 문제이므로, 단일 퍼셉트론(은닉층 없음)으로는 원리적으로 풀 수 없습니다.
- 역전파의 본질: forward pass에서 `X → z1 → a1 → z2 → a2` 순으로 흐른 계산이, backward pass에서는 정확히 반대 순서로 오차 신호가 전파됩니다.
- NumPy로 직접 구현해보는 이유: `loss.backward()` 한 줄 뒤에 숨겨진 체인룰 계산을 직접 경험해보면, 이후 vanishing gradient, NaN loss 같은 문제를 마주쳤을 때 원리에 기반해 원인을 추론할 수 있게 됩니다.
- Keras/TensorFlow의 가치: 동일한 수학적 원리를 훨씬 적은 코드로, 더 최적화된 연산(GPU 가속 포함)으로 실행할 수 있습니다.
