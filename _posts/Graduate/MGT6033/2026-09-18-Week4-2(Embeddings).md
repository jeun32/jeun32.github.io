---
title : "(Week4-2) NLP : 워드 임베딩부터 Word2Vec, BERT까지"
date : 2026-09-18 15:00:11 +0900
categories : [Graduate School, (MGT6033) Analysis of Unstructured Data_'26 Fall]
tags : [MGT6033, NLP, Word Embeddings, Word2Vec, gensim, BERT, Transformer, LSTM, Dimension Reduction]
math : true
---

지난 편(4-1)에서는 **정규 표현식**을 처음부터 실습까지 완결해서 다뤘습니다.
이번 편에서는 Week 4의 또 다른 축인 **워드 임베딩(Word Embeddings)** 으로 넘어갑니다.
단어의 "의미"를 좌표(벡터)로 바꾸는 직관에서 시작해, 이를 실제로 학습하는 대표 알고리즘 **Word2Vec**,
임베딩을 분석에 활용하는 법(사실은 **차원 축소**), **gensim 실습**,
그리고 문맥까지 이해하는 최신 모델 **BERT**까지 살펴봅니다.

> 📎 이 글은 **Week 4 - 2편(워드 임베딩 & 문맥 임베딩)** 입니다. 

## <span style="color:#FEB99C;">5. 워드 임베딩: 단어를 좌표로 바꾸기</span>

우리는 머릿속에서 단어를 "동물", "감정" 같은 범주로 정리합니다. **워드 임베딩(Word Embedding)** 은 이걸 흉내 냅니다.

> **워드 임베딩 = 의미가 비슷한 단어가 비슷한 표현을 갖도록 학습된, k차원 공간의 텍스트 표현**

예를 들어 1차원($k=1$)이라면 `elated`(1.0), `happy`(0.9), `sad`(-0.9), `distraught`(-1.0)처럼 배치할 수 있습니다.
실제로는 $k$를 **100, 200, 500** 처럼 크게 잡아, 각 단어를 그만큼의 숫자(벡터)로 표현합니다.
(그래서 **word vector = word embedding**, 같은 말입니다.)

$k=2$로 `king, president, horse, chicken, army, farm, castle, house`를 그려보면,
king·president는 위쪽(리더십), horse·chicken은 오른쪽 아래(생물), farm·castle·house는 왼쪽(장소)에 모입니다.
**임베딩은 이런 배치를 텍스트 사용 패턴으로부터 자동으로 학습**합니다.

## <span style="color:#FEB99C;">6. Word2Vec: 단순하지만 강력한 임베딩</span>

**Word2Vec** 은 2013년 구글이 발표한, **단일 은닉층 신경망(single-layer neural network)** 기반의 임베딩입니다.
단순함에 속으면 안 됩니다 — 굉장히 강력합니다. 핵심은 **비슷하게 쓰이는 단어의 코사인 유사도를 최대화**하는 최적화 문제입니다.

단일 층이라 **가법적(additive)** 성질이 있어, 그 유명한 관계식이 성립합니다.

$$ \text{king} - \text{man} + \text{woman} \approx \text{queen} $$

훈련 방식은 두 가지입니다.

| 방식 | 예측 방향 | 특징 |
|---|---|---|
| **CBOW** (Continuous Bag of Words) | 주변 단어 → **중심 단어** 예측 | **빠름**, 데이터 많이 필요, **빈번한 단어**에 강함 |
| **Skip-gram** | 중심 단어 → **주변 단어** 예측 | **작은 데이터**·**희귀 단어**에 강함 |

> 💡 창(window) `N`으로 앞뒤 몇 단어를 볼지 정합니다. 어느 쪽이든 결과 임베딩은 대체로 비슷합니다.
> 파이썬에서는 **gensim**("topic modeling for humans") 패키지를 씁니다.

**커스텀 모델을 왜 쓸까?** 특정 도메인의 언어 사용을 학습하기 위해서입니다.
회계·금융 텍스트에서 `research`와 `development`는 매우 비슷하게 쓰이지만,
일반 텍스트에서 `development`는 부동산·교육과 엮입니다.
강의의 유추(analogy) 비교에서 `creditor:lend :: debtor:borrow` 같은 금융 뉘앙스는
**fintext(금융 특화) 모델만** 제대로 잡아냈습니다. 단, 커스텀 모델은 **충분히 큰 코퍼스와 훈련 시간**이 필요합니다.

## <span style="color:#FEB99C;">7. 임베딩 활용: 사실은 차원 축소다</span>

의외의 관점: 워드 임베딩은 **차원 축소(dimension reduction)** 기법입니다.
DTM의 컬럼이 1,500~5,000개라도, 임베딩을 곱하면 **k(예: 300)차원**으로 줄일 수 있습니다.
(요인분석·PCA가 설문 50문항을 몇 개 잠재요인으로 요약하는 것과 같은 발상입니다.)

행렬 곱으로 간단히 표현됩니다.

$$ \underbrace{DTM}_{d \times n} \times \underbrace{E}_{n \times k} = \underbrace{문서임베딩}_{d \times k} $$

즉 **각 단어의 등장 횟수 × 그 단어의 임베딩**을 문서 단위로 합산하는 것입니다.

**어디에 쓰나?**
- **동의어 처리**: 같은 의견을 다른 단어로 쓴 리뷰가 서로 가깝게 표현됨
- **감성 측정 보강**: 사전에 없는 단어도 유사어를 통해 커버
- **문서 분류·토픽 식별** 성능 향상 (확장판으로 Doc2Vec, Top2Vec)

**한계**는 명확합니다: **문맥(context)을 놓칩니다.** 부정어(`not`), 수식어(`very`), 비꼼(sarcasm),
그리고 결정적으로 **다의어**(state capital / capital punishment / capital expenditure)를 구분하지 못합니다.

## <span style="color:#FEB99C;">8. 실습 C: gensim으로 Word2Vec 다루기</span>

> 💡 **gensim이란?**
> Word2Vec, FastText 등 단어나 문서를 벡터(숫자)로 변환하고, 단어 간의 유사도를 빠르게 계산·분석해 주는 대표적인 파이썬 자연어 처리(NLP) 라이브러리입니다.

### 사전학습 모델 사용
gensim의 Google News 사전학습 모델(300차원)을 불러오면 `KeyedVectors` 객체가 됩니다.
- `most_similar("president")` → chairman, vice_president, chief_executive (점수는 **코사인 유사도**)
- 이 모델은 **대소문자 구분** + **bigram** 허용 → `President`(고유명사 뒤)와 `president`(직위)의 뉘앙스 차이, 심지어 오타(`Pesident`)까지 포착
- `similarity("President","CEO")` ≈ 0.659, `get_vector`로 300개 숫자 벡터 직접 확인

### DTM을 임베딩 공간으로 투영
1. 실적 발표문(earnings announcement) 2,000건을 `CountVectorizer`로 DTM 생성 (cased 유지, 상위 500단어, 밀집행렬로 변환)
2. 각 단어의 300차원 벡터를 수집 — 모델 어휘에 **없는 단어는 0벡터**(`np.zeros(300)`)로 처리(합산에 영향 없음)
3. `np.stack`으로 **500×300 임베딩 행렬** 완성
4. `DTM(2000×500) @ 임베딩(500×300)` = **2000×300** 문서 임베딩 (`np.matmul`)
5. 문서 길이 효과 제거를 위해 **`normalize`(단위 길이 정규화)** 적용 → 코사인 유사도와 사실상 동일
6. "oil" 벡터와 가장 유사한 실적 발표문을 `cosine_similarity` + `argsort`로 탐색 → 실제로 **석유·가스 관련 기업**들이 상위에 등장(의미기반 검색 작동하였음)✅

### 커스텀 Word2Vec 훈련
gensim은 입력 구조가 독특한데, **리스트의 리스트**(바깥=문장, 안쪽=토큰) 형태여야 합니다.
- `nltk`의 `sent_tokenize` + `word_tokenize`로 전처리 함수 작성: 너무 짧거나(≤3) 긴(≥50) 문장 제거, 불용어·비알파벳·2글자 이하 제거, **약어(GAAP, EPS, EBITDA)는 대문자 유지**, 나머지는 소문자화
- 훈련 단위는 문서가 아닌 **문장**이므로, 3중 리스트를 **list comprehension으로 평탄화(flatten)**
- `Word2Vec(all_sentences, ...)` 로 인스턴스화하며 훈련 (기본은 **CBOW**, `sg=1`이면 skip-gram). 커스텀 모델은 `model.wv.most_similar(...)`처럼 **`.wv`** 를 거쳐 접근

> 🎯 결과 비교: 고작 2,000건 코퍼스인데도, `capital`·`goodwill` 같은 단어에서
> 커스텀 모델이 Google 모델보다 **금융 문맥을 훨씬 잘** 잡았습니다. 
> (예: goodwill → impairments, charges, write-offs, assets)
> 이 학습된 모델은 `model.save()`로 저장해 재훈련 없이 다시 사용할 수 있습니다.

## <span style="color:#FEB99C;">9. 문맥 기반 임베딩: BERT까지</span>

워드 임베딩의 한계(문맥 부재)를 극복하는 게 **문맥 기반 임베딩(Contextualized Embeddings)** 입니다.
"그거 참 잘됐네"가 진심인지 비꼼인지는 **주변 단어**와 **순서**로 결정되죠.
문맥을 넣는 두 가지 큰 접근이 있습니다.

| 접근 | 원리 | 대표 모델 |
|---|---|---|
| **순환(Recurrence)** | 이전 출력을 다음 입력으로 넣어 순서·근접성 포착 | **LSTM** |
| **어텐션(Attention)** | 시퀀스를 **한꺼번에** 처리하며 주변에 "주목" | **Transformer** |

### LSTM — "앞 내용을 기억하며 읽기"

**LSTM**은 문장을 왼쪽에서 오른쪽으로 **한 단어씩** 읽습니다.
각 단어를 이해할 때 ①지금 보는 단어 ②바로 앞에서 이해한 내용 ③지금까지 대화의 전반적 분위기(기억)를 함께 씁니다.
사람이 글을 읽을 때 앞 문장을 기억하며 다음 문장을 이해하는 것과 똑같죠.

> 다만 **순서대로 한 단어씩** 처리해야 해서 **속도가 느립니다.** 오래 최강자였지만 이 점이 발목을 잡았어요.

### Transformer & BERT — "한눈에 보고 중요한 데 집중하기"

2017년 *"Attention is All You Need"* 라는 논문이 판을 바꿨습니다.
핵심 아이디어는 문장을 **통째로 한 번에 넣고**, 각 단어가 **어떤 단어에 주목(attention)해야 하는지**를 스스로 배우는 것입니다.
순서대로 읽지 않으니 **훨씬 빠르고**, 여러 개를 동시에 처리(병렬화)할 수 있어 GPU와 궁합이 좋습니다.

이 구조로 만든 대표 모델이 구글의 **BERT**입니다.

- 구글 검색의 핵심 엔진. 문장을 넣으면 **문맥이 반영된 임베딩**을 만들어 냄
- **양방향(bidirectional)**: 단어의 **왼쪽·오른쪽을 동시에** 봄 → 그래서 문맥 파악이 뛰어남
- 규모가 엄청 커서(3.45억개의 parameter) 성능이 매우 좋음

> 💡 "3.45억 개를 내가 어떻게 훈련해?" → **전이학습(Transfer Learning)** 덕분에 괜찮습니다.
> 거대 코퍼스로 학습된 파라미터에서 출발해, 우리가 **수백~수천 개만 라벨링**해서 **미세조정(fine-tuning)** 하면 됩니다.
> 즉 밑바닥부터 만들 필요 없이 "완성품을 내 용도에 맞게 손보는" 방식이라, BERT 및 파생모델들이 NLP의 게임 체인저가 됐습니다.

## <span style="color:#FEB99C;">10. 최종 정리</span>

> Week 4의 메시지는 **"유연함"** 입니다.
> RegEx로 **원하는 패턴을 정확히 잡아내고**,
> 워드 임베딩으로 **단어의 의미를 벡터에 담고**,
> 문맥 모델로 **말의 뉘앙스까지 이해**하는 것 — 이것이 NLP에 유연성을 더하는 도구들입니다.

이번 주에 배운 것을 정리하면:

- **정규 표현식**은 "언어 속의 언어". 문자/클래스/연산자 + 그룹으로 검색·추출·치환. **greedy vs. lazy**, `\b`, look-around, 백레퍼런스가 핵심 무기
- 실무 철학: **완벽보다 80~95%**, 단순하게 짜고 필요할 때 조이기, [regex101](https://regex101.com)에서 테스트
- **워드 임베딩**은 단어를 k차원 벡터로 → 사실은 **차원 축소**. 유사어·감성·분류에 유용하나 **문맥 부재**가 한계
- **Word2Vec**: 단일층 신경망, **CBOW vs. Skip-gram**, gensim으로 사용/훈련. **도메인 커스텀 모델**이 뉘앙스를 잘 잡음
- **문맥 임베딩**: 순환(**LSTM**) 또는 어텐션(**Transformer/BERT**)으로 문맥 포착, **전이학습 + 미세조정**이 실전의 열쇠

---
