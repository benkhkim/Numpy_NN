import tensorflow as tf
import numpy as np

# ============================================================
# 1. 데이터 준비
# ============================================================
# numpy 버전과 동일한 XOR 데이터
# dtype=float32인 이유: TensorFlow는 기본적으로 float32를 사용
# (GPU 연산 효율성과 메모리 절약을 위해. numpy는 기본이 float64라서 미세한 정밀도 차이 발생)
X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float32)
y = np.array([[0],[1],[1],[0]], dtype=np.float32)

# ============================================================
# 2. 모델 구조 정의 (Forward pass의 "설계도"만 선언)
# ============================================================
# Sequential: 레이어를 순서대로 쌓는 가장 단순한 모델 구조
# 주의: 여기서는 아직 가중치가 랜덤 초기화만 될 뿐, 실제 forward 계산은 일어나지 않음
model = tf.keras.Sequential([

    # Dense(4, ...): 완전연결층(fully-connected layer), 뉴런 4개
    # 내부적으로 하는 일은 numpy에서 우리가 직접 짠 것과 동일:
    #   출력 = activation(입력 @ W + b)
    # 여기서 W의 shape은 (2,4), b의 shape은 (4,) -- numpy의 W1, b1과 대응
    # input_shape=(2,)는 "입력이 2차원 벡터"라는 것을 명시 (X의 열 개수)
    tf.keras.layers.Dense(4, activation='sigmoid', input_shape=(2,)),

    # Dense(1, ...): 두 번째 완전연결층, 뉴런 1개 (출력층)
    # 이 레이어의 W는 shape (4,1), b는 shape (1,) -- numpy의 W2, b2와 대응
    # 첫 번째 레이어의 출력(4차원)을 입력으로 자동으로 받음 (별도 명시 불필요)
    tf.keras.layers.Dense(1, activation='sigmoid')
])
# activation='sigmoid': 우리가 numpy에서 직접 만든 sigmoid() 함수와 정확히 동일한 함수
# Keras는 'relu', 'tanh', 'softmax' 등 문자열 하나로 다양한 활성화 함수를 바로 사용 가능

# ============================================================
# 3. 학습 방식 설정 (compile 단계)
# ============================================================
model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.5),
    # SGD(Stochastic Gradient Descent): numpy에서 우리가 직접 짠
    #   W -= lr * d_W
    # 이 한 줄을 그대로 자동화한 것. learning_rate=0.5는 numpy의 lr=0.5와 동일 개념
    # (참고: SGD 외에 Adam, RMSprop 등으로 optimizer만 바꾸면 더 정교한 업데이트 방식 사용 가능)

    loss='mse'
    # Mean Squared Error: numpy에서 직접 계산한
    #   loss = np.mean((y - a2) ** 2)
    # 와 동일한 손실함수. 문자열 'mse'만 지정하면 Keras가 알아서 사용
)
# compile()은 "어떤 loss로, 어떤 optimizer로 학습할지"를 모델에 등록하는 단계
# 이 시점까지도 아직 실제 학습(가중치 업데이트)은 시작되지 않음

# ============================================================
# 4. 학습 실행 (fit 단계) — numpy의 for-loop 전체를 대체
# ============================================================
model.fit(X, y, epochs=10000, verbose=0)
# 이 한 줄이 numpy 코드의 다음 부분 전체를 대신 실행함:
#
#   for epoch in range(10000):
#       forward pass (z1, a1, z2, a2 계산)
#       loss 계산
#       backward pass (d_W1, d_b1, d_W2, d_b2 계산 — 체인룰 자동 적용)
#       W -= lr * d_W  (파라미터 업데이트)
#
# 내부적으로 TensorFlow는 "GradientTape"라는 자동미분 엔진을 사용:
#   1) forward pass를 실행하면서 모든 연산을 "계산 그래프"로 기록
#   2) loss가 계산된 후, 이 그래프를 역방향으로 순회하며
#      체인룰을 자동으로 적용해 각 가중치의 gradient를 계산
#   3) optimizer가 그 gradient로 가중치를 업데이트
#
# epochs=10000: numpy의 for epoch in range(10000)과 동일
# verbose=0: 학습 중 진행 로그(1 epoch당 한 줄씩)를 출력하지 않음
#            (verbose=1로 하면 매 epoch마다 loss가 출력되어 매우 많은 로그가 찍힘)

# ============================================================
# 5. 예측 (학습된 모델로 forward pass만 실행)
# ============================================================
print(model.predict(X))
# 학습이 끝난 model에 X를 다시 넣어 최종 예측값(a2에 해당)을 출력
# 내부적으로는 학습 때와 똑같은 forward pass 계산이지만,
# 이번엔 gradient 계산이나 가중치 업데이트 없이 "추론(inference)"만 수행
# 출력 shape: (4, 1) — 4개 샘플 각각에 대한 0~1 사이의 예측 확률값
