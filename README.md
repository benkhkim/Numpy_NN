# Numpy_NN
Numpy NN vs Keras NN



numpy 코드	Keras 코드	의미
W1 = np.random.randn(2,4)*0.5	Dense(4, input_shape=(2,))	가중치 자동 생성 (초기화 방식은 다름)
sigmoid(z1)	activation='sigmoid'	동일한 활성화 함수
z1 = X@W1+b1; a1=sigmoid(z1)	레이어 1개	forward pass 한 층
loss = np.mean((y-a2)**2)	loss='mse'	동일한 손실함수
d_W1, d_W2, ... (체인룰 직접 계산)	.fit() 내부의 자동미분(GradientTape)	역전파 자동화
W -= lr*d_W	SGD(learning_rate=0.5)	동일한 업데이트 규칙
for epoch in range(10000): ...	epochs=10000	학습 반복 자동화
