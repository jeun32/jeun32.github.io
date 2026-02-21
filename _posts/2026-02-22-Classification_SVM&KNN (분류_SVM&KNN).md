---
title : Classification_SVM&KNN (분류_SVM&KNN)
date : 2026-02-22 01:05:00 +0900
categories : [Graduate School, (FA25) Introduction to Analytics Modeling]
tags : [ISYE6501, R Classification, R SVM, R KNN]
math : true
---

> [📎 전체 코드 Github Link](https://github.com/jeun32/analytics-modeling/blob/main/notebooks/01-classification(SVM%26KNN).R)

## Question 2.2.1 – 2.2.2 : Support Vector Machine (SVM)

### 1. 데이터 불러오기 및 전처리

먼저 신용카드 데이터셋을 불러오고 구조를 확인한다.

```r
data <- read.table("credit_card_data-headers.txt", header = T)
str(data)

data[,11] = as.factor(data[,11])
str(data)
```

SVM은 `kernlab` 패키지에서 **분류 문제의 경우 종속변수가 factor 타입이어야 하므로**,  
R1 변수를 factor로 변환하였다.
<div style="display: flex; justify-content: center; gap: 2em; margin: 2em 0;">
  
  <div style="text-align: center;">
    <img 
      src="/assets/img/posts/2026-02-22-Classification_SVM&KNN (분류_SVM&KNN)/SVM_1.jpg"
      alt="credit_card data 형태"
      style="width:100%; max-width:400px; display:block;"
    >
    <div style="font-size: 0.9em; color: gray; margin-top: 0.4em;">
      <em>credit_card data 형태</em>
    </div>
  </div>
</div>
---

### 2. 모형 설정

설명변수 행렬(X)과 종속변수(y)를 정의한다.

```r
library(kernlab)

X <- as.matrix(data[,1:10])
y <- data$R1
```

이후 서로 다른 커널을 사용한 세 가지 SVM 모형을 추정하였다.

- Linear Kernel (`vanilladot`)
- RBF Kernel (`rbfdot`)
- Polynomial Kernel (`polydot`)

비교를 위해 비용 파라미터 C는 1로 고정하였다.

```r
model  <- ksvm(X, y, kernel='vanilladot', C=1, scaled=TRUE)
model2 <- ksvm(X, y, kernel='rbfdot',     C=1, scaled=TRUE)
model3 <- ksvm(X, y, kernel='polydot',    C=1, scaled=TRUE)
```

C 값은 **마진 최대화와 분류 오차 최소화 사이의 균형을 조정하는 파라미터**이다.

- C가 클수록 → 오분류 덜 허용 (복잡한 classification 경계)
- C가 작을수록 → 규제가 강해짐 (더 넓은 margin)

---

### 3. Intercept (Bias Term)

```r
a0 <- model@b
-a0
```

`ksvm`에서 추출한 `model@b`는 내부적으로 저장된 bias 값이다.  
우리가 일반적으로 사용하는 결정경계 식

$$
w^T x + b = 0
$$

과의 표현을 맞추면, 실제 절편 b는 `-a0`에 해당한다.

<div style="display: flex; justify-content: center; gap: 2em; margin: 2em 0;">
  
  <div style="text-align: center;">
    <img 
      src="/assets/img/posts/2026-02-22-Classification_SVM&KNN (분류_SVM&KNN)/SVM_2.jpg"
      alt="Linear Kernel (vanilladot)의 추정 절편"
      style="width:100%; max-width:400px; display:block;"
    >
    <div style="font-size: 0.9em; color: gray; margin-top: 0.4em;">
      <em> ex) Linear Kernel (vanilladot)의 추정 절편</em>
    </div>
  </div>
</div>
---

### 4. 예측 및 정확도 비교

각 커널에 대해 예측값을 계산하고 정확도를 비교하였다.

```r
pred  <- predict(model,  X)
pred2 <- predict(model2, X)
pred3 <- predict(model3, X)

sum(pred  == y) / nrow(data)
sum(pred2 == y) / nrow(data)
sum(pred3 == y) / nrow(data)
```

정확도는 다음과 같이 계산된다.

정확히 분류된 관측치 수 / 전체 관측치 수

이를 통해 서로 다른 커널의 분류 성능을 비교할 수 있다.
<div style="display: flex; justify-content: center; gap: 2em; margin: 2em 0;">
  
  <div style="text-align: center;">
    <img 
      src="/assets/img/posts/2026-02-22-Classification_SVM&KNN (분류_SVM&KNN)/SVM_3.jpg"
      alt="각 모델에서의 accuracy"
      style="width:100%; max-width:400px; display:block;"
    >
    <div style="font-size: 0.9em; color: gray; margin-top: 0.4em;">
      <em>각 모델에서의 accuracy</em>
    </div>
  </div>
</div>
---

### 5. 계수 해석 (Linear SVM 결과 기반 설명)

선형 SVM에서 계산된 계수 벡터는 다음과 같다.
<div style="display: flex; justify-content: center; gap: 2em; margin: 2em 0;">
  
  <div style="text-align: center;">
    <img 
      src="/assets/img/posts/2026-02-22-Classification_SVM&KNN (분류_SVM&KNN)/SVM_4.jpg"
      alt="계수 벡터"
      style="width:100%; max-width:600px; display:block;"
    >
    <div style="font-size: 0.9em; color: gray; margin-top: 0.4em;">
      <em>선형 SVM의 계수 벡터</em>
    </div>
  </div>
</div>

```
           A1            A2            A3            A8            A9           A10           A11 
-0.0011026642 -0.0008980539 -0.0016074557  0.0029041700  1.0047363456 -0.0029852110 -0.0002035179 
          A12           A14           A15 
-0.0005504803 -0.0012519187  0.1064404601
```

계수의 해석은 선형 분류모형과 유사한데,

- 계수의 절댓값이 클수록 결정경계에 더 큰 영향을 준다.
- 계수가 양수이면 클래스 1로 분류되는 방향으로 작용하고
- 계수가 음수이면 클래스 0 방향으로 작용한다.

이번 결과에서 가장 눈에 띄는 변수는 다음과 같다.

- **A9 (≈ 1.00)** → 가장 큰 양의 계수  
- **A15 (≈ 0.106)** → 두 번째로 큰 양의 계수  

즉, A9가 결정경계 형성에 가장 큰 영향을 미치고 있다.
(클래스 분리에 핵심적인 역할을 하고 있음)

---

## Question 2.2.3 : K-Nearest Neighbors (KNN)

### 1. 종속변수 변환

KNN에서는 종속변수를 numeric(0/1)로 변환하여 사용하였다.

```r
library(kknn)

data$R1 <- as.numeric(as.character(data$R1))
```

---

### 2. Leave-One-Out Cross Validation (LOOCV)

k 값을 1부터 25까지(홀수) 변화시키면서 LOOCV 방식으로 정확도를 계산하였다.

```r
n <- nrow(data)
k.values <- seq(1, 25, by = 2)
acc <- numeric(length(k.values))

for (j in seq_along(k.values)) {
  preds <- numeric(n)
  
  for (i in 1:n) {
    fit <- kknn(
      R1 ~ ., 
      train = data[-i, ], 
      test  = data[i, , drop = FALSE], 
      k = k.values[j],
      scale = TRUE
    )
    
    p <- fitted(fit)
    preds[i] <- ifelse(p > 0.5, 1, 0)
  }
  
  acc[j] <- mean(preds == data$R1)
}
```

---

### 3. LOOCV 결과 및 최적의 k 값

```r
results <- data.frame(k = k.values, accuracy = acc)
print(results)
```
<div style="display: flex; justify-content: center; gap: 2em; margin: 2em 0;">
  
  <div style="text-align: center;">
    <img 
      src="/assets/img/posts/2026-02-22-Classification_SVM&KNN (분류_SVM&KNN)/KNN_1.jpg"
      alt="LOOCV 수행결과"
      style="width:100%; max-width:400px; display:block;"
    >
    <div style="font-size: 0.9em; color: gray; margin-top: 0.4em;">
      <em>LOOCV 수행결과</em>
    </div>
  </div>
</div>
```
    k  accuracy
1   1 0.8149847
2   3 0.8149847
3   5 0.8516820
4   7 0.8470948
5   9 0.8470948
6  11 0.8516820
7  13 0.8516820
8  15 0.8532110
9  17 0.8516820
10 19 0.8501529
11 21 0.8486239
12 23 0.8440367
13 25 0.8455657
```
가장 높은 정확도는 다음과 같다.

- k = 15일 때 accuracy ≈ **0.8532 (85.32%)**

k가 너무 작을 경우 정확도가 약 81% 수준으로 낮게 나타났는데,
이는 매우 소수의 이웃만을 기준으로 분류하면서 데이터의 작은 변동(noise)까지 반영했기 때문으로 볼 수 있다.(overfitting)

반면 k가 너무 커지면(예: k ≥ 21) 정확도가 다시 감소하는 경향을 보였다.  
이는 지나치게 많은 이웃을 평균내면서 결정경계가 과도하게 단순해졌기 때문이다.(underfitting)

- 작은 k → 복잡한 결정경계 → 분산 증가 (variance ↑)
- 큰 k → 단순한 결정경계 → 편향 증가 (bias ↑)

이 data에서는 k=15 부근에서 
bias와 variance 사이의 균형이 가장 잘 맞은것으로 보인다.

---

### 4. SVM 결과와 비교

KNN의 최고 정확도는 약 85.3%로 나타났다.  

반면 SVM은 Linear 기준(vanilladot) 약 86.4%,  
RBF 커널은 약 87.0%로 더 높은 정확도를 보였다.

즉, 이번 데이터에서는  
주변 이웃을 기반으로 판단하는 KNN보다  
결정경계를 학습하는 SVM이 더 효과적이었다.

쉽게 말하면,

- KNN은 “이 점 주변에 누가 많은가?”를 보고 판단하는 방법이고
- SVM은 데이터를 가장 잘 가르는 경계를 찾는 방법이다.

이번 결과에서는 SVM의 성능이 더 높았기 때문에,  
이 데이터는 비교적 명확한 결정경계를 형성할 수 있는 구조를 가진 것으로 보인다.

---