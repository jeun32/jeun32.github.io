---
title : KNN Classification & K-means Clustering
date : 2026-03-23 22:02:00 +0900
categories : [Graduate School, (FA25) Introduction to Analytics Modeling]
tags : [ISYE6501, R Classification, R SVM, R KNN]
math : true
---

> [📎 전체 코드 Github Link](https://github.com/jeun32/analytics-modeling/blob/main/notebooks/02-classification(KNN%26clustering-Kmeans).R)

이번 글에서는 supervised learning 기반의 KNN classification과 unsupervised learning 기반의 K-means clustering을 적용하고, 두 방법의 목적과 성능을 비교한다.

## Question 3.1 : KNN Classification (CV vs Holdout)

### 1. 데이터 불러오기 및 전처리

```r
data <- read.table("credit_card_data-headers.txt", header = TRUE)
data$R1 <- factor(as.numeric(as.character(data$R1)), levels = c(0, 1))
```

- 종속변수 R1을 **0/1 factor 형태**로 변환
- KNN 분류에서 클래스 비교를 위해 factor 형태 유지

---

### 2. 10-Fold Cross Validation (Stratified)

일반적인 K-fold CV가 아닌,  
**class 비율을 유지하는 stratified 방식**으로 fold를 구성하였다.

```r
fold_id <- make_stratified_folds(data$R1, K = 10)
```

이후 k 값을 1~29 (홀수)로 변화시키며 성능을 비교하였다.

<div style="display: flex; justify-content: center; gap: 2em; margin: 2em 0;">
  
  <div style="text-align: center;">
    <img 
      src="/assets/img/posts/2026-03-23-KNN Classification & K-means Clustering (분류_KNN & 군집_K-means)/KNN&K-means_1.jpg"
      alt="근접한 k개수에 따른 accuracy"
      style="width:100%; max-width:600px; display:block;"
    >
    <div style="font-size: 0.9em; color: gray; margin-top: 0.4em;">
      <em>근접한 k개수에 따른 accuracy</em>
    </div>
  </div>
</div>
---

📌 **결과**

- Best k = **23**
- CV Accuracy ≈ **84.71%**

---

### 3. Train / Validation / Test Split

데이터를 다음과 같이 분할하였다.

- Train : 60%
- Validation : 20%
- Test : 20%

```r
# split
```

Validation set에서 최적 k를 선택:

```
Best k = 21  
Validation Accuracy ≈ 83.21%
```

<div style="display: flex; justify-content: center; gap: 2em; margin: 2em 0;">
  
  <div style="text-align: center;">
    <img 
      src="/assets/img/posts/2026-03-23-KNN Classification & K-means Clustering (분류_KNN & 군집_K-means)/KNN&K-means_2.jpg"
      alt="최적 k개수 도출"
      style="width:100%; max-width:600px; display:block;"
    >
    <div style="font-size: 0.9em; color: gray; margin-top: 0.4em;">
      <em>최적 k개수 도출</em>
    </div>
  </div>
</div>

---

### 4. Final Model & Test 성능

Train + Validation 데이터를 합쳐 최종 모델을 학습하고,  
Test set에서 성능을 평가하였다.

```
Test Accuracy ≈ 84.73%
```

#### Confusion Matrix

```
        Pred
True     0   1
   0    64  14
   1     6  47
```
<div style="display: flex; justify-content: center; gap: 2em; margin: 2em 0;">
  
  <div style="text-align: center;">
    <img 
      src="/assets/img/posts/2026-03-23-KNN Classification & K-means Clustering (분류_KNN & 군집_K-means)/KNN&K-means_3.jpg"
      alt="모델 성능 & Confusion Matrix"
      style="width:100%; max-width:600px; display:block;"
    >
    <div style="font-size: 0.9em; color: gray; margin-top: 0.4em;">
      <em>모델 성능 & Confusion Matrix</em>
    </div>
  </div>
</div>

---

### 5. 결과 해석

#### (1) CV vs Holdout 비교

| 방법 | Best k | Accuracy |
|------|--------|----------|
| 10-fold CV | 23 | 84.71% |
| Holdout (Validation) | 21 | 83.21% |
| Test (Final) | 21 | 84.73% |

👉 두 방법 모두 유사한 결과를 보이며,  
**k ≈ 20대 초반이 최적 영역**으로 확인됨

👉 또한 CV와 Holdout 결과가 유사하게 나타난 점에서,  
모델의 generalization 성능이 안정적인 것으로 해석할 수 있다.

---

#### (2) Bias-Variance 관점

- k가 작을 때 → noise에 민감 → **overfitting**
- k가 클 때 → 경계 단순화 → **underfitting**

👉 **k ≈ 21~23에서 최적 균형**

---

#### (3) Confusion Matrix 해석

- False Positive (14) > False Negative (6)

👉 모델은 약간 **공격적인 성향 (positive 예측 많음)**

---

## Question 4.2 : K-means Clustering (Iris Data)

### 1. 데이터 및 전처리

```r
X <- scale(iris[,1:4])
y <- iris$Species
```

- 거리 기반 알고리즘 → scaling 필수

---

### 2. K-means 수행

```r
k <- 3
km <- kmeans(X, centers = k, nstart = 50)
```

- k = 3 (실제 클래스 수)
- nstart = 50 → 안정성 확보

---

### 3. Clustering 결과

```
           setosa versicolor virginica
cluster1      0        11        36
cluster2     50         0         0
cluster3      0        39        14
```

---

### 4. Clustering 성능

```r
Simple accuracy = 0.8333
```

※ K-means는 비지도학습이므로, 해당 정확도는 실제 라벨과 비교한 **사후적 평가 지표로 해석해야 한다.**

👉 약 **83.33%**

<div style="display: flex; justify-content: center; gap: 2em; margin: 2em 0;">
  
  <div style="text-align: center;">
    <img 
      src="/assets/img/posts/2026-03-23-KNN Classification & K-means Clustering (분류_KNN & 군집_K-means)/KNN&K-means_4.jpg"
      alt="Clustering 결과 및 성능"
      style="width:100%; max-width:600px; display:block;"
    >
    <div style="font-size: 0.9em; color: gray; margin-top: 0.4em;">
      <em>Clustering 결과 및 성능</em>
    </div>
  </div>
</div>


---

### 5. 결과 해석

#### (1) Setosa 완벽 분리

- cluster2 → setosa 50개

👉 매우 잘 구분되는 클래스

---

#### (2) Versicolor vs Virginica 혼합

👉 두 클래스는 feature 공간에서 겹침

👉 이는 비지도학습의 특성상 클래스 경계가 명확하지 않은 경우 완벽한 분리가 어렵다는 점을 보여준다.

---

#### (3) 중요한 변수

- petal length / width → 핵심 변수
- sepal → 영향 상대적으로 작음

---

## 🔥 Final Insight

- **KNN**
  → 주변 이웃 기반  
  → 예측 성능 우수

- **K-means**
  → 거리 기반 군집화  
  → 구조 파악에 유리

👉 결론:

> **지도학습(KNN)은 예측 정확도에 강점을 가지며,  
비지도학습(K-means)은 데이터의 내재된 구조를 파악하는 데 강점을 가진다.**