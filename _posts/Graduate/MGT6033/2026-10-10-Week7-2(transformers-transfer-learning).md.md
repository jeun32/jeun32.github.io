---
title : "(Week7-2) Transformer와 전이학습 : 감성 모델을 재무 문장 분류기로 바꾸기 (Hugging Face & Fine-tuning)"
date : 2026-10-10 16:56:10 +0900
categories : [Graduate School, "(MGT6033) Analysis of Unstructured Data_'26 Fall"]
tags : [MGT6033, NLP, Transformer, Transfer Learning, Fine-tuning, BERT, DistilBERT, Hugging Face, Attention, Financial Text]
math : true
---

1편에서는 신경망을 직접 만들고,
뉴스의 단어 빈도로 가짜 뉴스를 분류했습니다.

그런데 문맥을 이해하는 대형 신경망을 **매번 처음부터 학습**해야 한다면 어떨까요?

- 수천만~수억 개의 파라미터
- 방대한 텍스트
- 긴 훈련 시간
- 큰 계산 비용

개인 연구자나 작은 조직이 감당하기 쉽지 않습니다.
여기서 등장하는 것이 **전이학습(Transfer Learning)** 입니다.

> **“이미 언어를 배운 모델을 가져와, 우리 문제에 맞게 조금 더 가르치자.”**

이번 2편도 두 부분으로 나뉩니다.

- **1부 (개념)**: 전이학습의 원리, Transformer 구조(Attention·Encoder/Decoder), BERT·GPT 계열, 활용 분야
- **2부 (실습)**: Hugging Face로 감성 모델을 재무 문장 분류기로 바꾸는 파인튜닝

> 📝 **읽기 안내**
> 성능 수치는 강의에서 관찰한 결과입니다.
> 코드는 강의 흐름을 재구성한 학습용 예시이며,
> 분류 head 초기화·최적 체크포인트 복원·토크나이저 저장 등 보완 사항을 포함합니다.
> 라이브러리 버전과 실행 환경에 따라 API와 결과가 달라질 수 있습니다.

---

# 📘 1부. 개념

---

## <span style="color:#FEB99C;">1. 전이학습: 처음부터 배우지 않아도 된다</span>

전이학습은 **기존 과제에서 학습한 지식을 새로운 과제에 재사용하는 것**입니다.

영화 리뷰 감성 모델을 생각해 봅시다. 이 모델은 이미:

- `excellent` 같은 긍정 표현
- `terrible` 같은 부정 표현
- 부정어가 문장 의미를 바꾸는 방식
- 단어 사이의 문맥 관계

를 어느 정도 학습했습니다.

이 모델을 병원 이용 후기 분석에 가져오면,
영어의 구조를 처음부터 다시 배울 필요는 없습니다.
대신 진단·치료·대기 시간·의료진 설명 같은 표현을 새 환경에 맞게 익히면 됩니다.

> 💡 **전이하는 것은 최종 레이블만이 아니라,
> 입력을 이해하는 내부 표현과 연결 구조입니다.**

### 🔎 전이학습을 정확히 요약하기

NLP 전이학습을 설명하는 서술에는 참·거짓이 섞여 있어 구분이 필요합니다.

| 서술 | 맞을까? | 이유 |
|---|---|---|
| 단어 문맥이 분야 간에 크게 바뀌지 않을 때 가장 효과적이다 | ✅ | 쓰는 언어가 비슷한 분야끼리는 기존 표현이 잘 전이됩니다. |
| 완전히 고유한 어휘만 쓰는 분야는 거의 없다 | ✅ | 대부분 분야는 공통 언어를 많이 공유해, 기존 지식을 재활용할 수 있습니다. |
| 전이학습은 새 과제마다 수백만 개의 레이블이 필요하다 | ❌ | 정반대입니다. **적은 데이터로도 적응**할 수 있는 것이 전이학습의 핵심 장점입니다. |
| 전이학습 모델은 새 과제에서도 언어·문맥·문법을 포착할 수 있다 | ✅ | 사전학습으로 익힌 언어 능력을 새 과제에 활용합니다. |

> ⭐ "수백만 개의 레이블이 매번 필요하다"는 서술은 **전이학습을 쓰는 이유 자체와 모순**됩니다.
> 전이학습은 "이미 똑똑한 모델을 데려와 적은 데이터로 조금만 더 가르치는 것"이기 때문입니다.
> 영어를 잘하는 사람에게 "의학 영어"만 추가로 가르치는 비유를 떠올리면 됩니다.

### 1-1. 데이터와 과제가 바뀌는 세 가지 경우

| 경우 | 예시 |
|---|---|
| **같은 데이터, 다른 과제** | 영화 리뷰의 감성 → 리뷰의 전문성 |
| **다른 데이터, 유사한 과제** | Wikipedia 주제 분류 → 뉴스 주제 분류 |
| **다른 데이터, 다른 과제** | 기존 문서 분류 → 새로운 말뭉치의 감성 분석 |

이번 실습(2부)은 기존의 **긍정/부정 감성 분류 모델**을 가져와,
**재무 성과 관련 문장인지 아닌지**를 판단하도록 바꿉니다.

### 1-2. 사전학습·파인튜닝·추론의 차이

| 단계 | 하는 일 | 가중치 업데이트 |
|---|---|---|
| **사전학습(Pretraining)** | 대규모 데이터에서 일반적인 표현 학습 | O |
| **파인튜닝(Fine-tuning)** | 특정 과제·도메인에 맞게 추가 학습 | O |
| **추론(Inference)** | 학습된 모델로 새 입력 예측 | 보통 X |

> 🙋 **“모델을 다운로드해서 쓰면 그게 파인튜닝인가요?”**
>
> 아닙니다. 다운로드한 모델로 바로 예측하면 **추론**입니다.
> 새 데이터와 목적함수로 가중치를 업데이트해야 **파인튜닝**입니다.

### 1-3. 적은 데이터로도 가능한 이유

모델을 처음부터 학습하면 **언어 표현과 분류 규칙을 모두** 배워야 합니다.
반면 전이학습은 이미 좋은 출발점이 있으므로,
상대적으로 적은 레이블로도 유용한 모델을 만들 수 있습니다.

강의에서는 **500~1,000개의 문서로도 유용한 적응이 가능할 수 있다**고 설명합니다.
실제 데모는 **2,000개의 레이블된 문장**을 사용합니다.

> ⚠️ 이것은 “1,000개면 언제나 충분하다”는 보장이 아닙니다.
> 과제 난이도·레이블 품질·도메인 차이·클래스 수에 따라 필요한 데이터는 달라집니다.

### 🔎 전이학습이 항상 이득일까?

그렇지도 않습니다. 기존 모델의 학습 환경과 새 환경이 크게 다르면:

- 기존 표현이 새 과제에 맞지 않음
- 기존 편향이 이전됨
- 새로운 데이터에 과적합
- 원래 유용했던 표현이 손상됨

같은 문제가 생길 수 있습니다.
관련성이 낮은 지식 때문에 성능이 나빠지는 경우를 **부정적 전이(negative transfer)** 라고 부릅니다.

> 핵심은 **“큰 모델을 가져왔는가?”가 아니라
> “새 과제에 적절한 출발점을 가져왔는가?”** 입니다.

## <span style="color:#FEB99C;">2. Transformer: 단어 사이의 관계를 Attention으로 계산한다</span>

Transformer의 핵심 질문은 간단합니다.

> **“이 단어를 이해하려면, 문장 안의 어떤 다른 단어를 봐야 할까?”**

예를 들어 `I am wearing a jacket because I am cold.` 에서
`jacket`의 의미를 해석할 때, 바로 옆 단어뿐 아니라 뒤쪽의 `cold`도 중요한 정보입니다.
Transformer는 **Attention**으로 이런 관계를 계산합니다.

> 🙋 **강의가 Transformer를 배우며 가장 먼저 세운 목표는?**
>
> 강의의 1차 학습 목표는 **"언어 Transformer가 이전에 쓰던 신경망(RNN 등)과 어떻게 다른지를 구별하는 것"** 이었습니다.
> "처음부터 Transformer를 설계하기"나 "파이썬으로 구현하기"가 아니라,
> **기존 방식과의 차이를 이해하는 것**이 출발점입니다.
> 왜 Transformer가 등장했는지(순차 처리의 한계, 긴 의존성 문제 등)를 아는 것이 핵심입니다.

### 2-1. 네 가지 핵심 요소

| 요소 | 역할 |
|---|---|
| **Self-attention** | 같은 시퀀스의 토큰 사이 관계 계산 |
| **위치 정보(Positional encoding)** | 토큰 순서와 위치 반영 |
| **문맥 표현(Contextual representation)** | 주변 토큰에 따라 표현을 달리 생성 |
| **Encoder / Decoder 구조** | 이해·표현과 생성 과제에 맞는 계산 수행 |

> 🔎 **Transformer를 정의하는 핵심 특징과, 그렇지 않은 것**
>
> Transformer를 "정의하는" 특징으로는 **Self-attention, Positional encoding, Contextual representation** 이 꼽힙니다.
>
> 반면 함정이 되는 보기들:
> - **토큰의 순차 처리(sequential processing)** → ❌ 이것은 오히려 **RNN의 방식**입니다.
>   Transformer는 토큰을 순서대로 하나씩 읽지 않고 **한꺼번에(병렬로)** 처리합니다.
> - **Encoder-decoder 구조** → △ 틀린 설명은 아니지만, 모든 Transformer가 두 부분을 다 쓰는 것은 아닙니다.
>   BERT는 encoder만, GPT는 decoder만 사용합니다. 그래서 "핵심 정의 특징"에서는 빠지기도 합니다.
>
> 정리하면, **"순차 처리/RNN처럼"** 이 보이면 Transformer 특징이 아니라고 보면 됩니다.

### 2-2. Self-attention의 직관

토큰 표현에서 세 가지 벡터를 만듭니다.

| 벡터 | 직관 |
|---|---|
| **Query(Q)** | 나는 어떤 정보를 찾고 있는가? |
| **Key(K)** | 나는 어떤 정보를 제공하는가? |
| **Value(V)** | 실제로 전달할 정보는 무엇인가? |

대표적인 수식은 다음과 같습니다.

$$
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt{d_k}}
\right)V
$$

직관적으로 읽으면:

1. Query와 Key로 토큰 사이 관련성 점수 계산
2. Softmax로 가중치 생성
3. 그 가중치로 Value를 섞음

> 💡 **Self-attention은 토큰들이 서로를 참고해 자신의 표현을 업데이트하는 과정**입니다.

여러 attention head를 사용하면 서로 다른 관계를 병렬적으로 표현할 수 있습니다.
다만 각 head가 언제나 사람이 해석할 수 있는 특정 문법 역할을 맡는 것은 아닙니다.

### 2-3. 위치 정보가 필요한 이유

토큰의 순서는 의미를 바꿉니다.

```text
The dog bit the man.
The man bit the dog.
```

단어 목록이 같아도 문장은 다릅니다.
Transformer는 RNN처럼 한 단어씩 순서대로 읽는 구조에만 의존하지 않으므로,
**별도의 위치 정보**를 제공해야 합니다.

> 💡 바로 이 때문에 위치 정보가 **핵심 요소**입니다.
> 토큰을 한꺼번에 보는 구조라 "순서를 모르는" 약점이 생기고,
> 그 약점을 메우려고 각 토큰에 "너는 몇 번째" 정보를 따로 넣어 주는 것입니다.
> (원래 Transformer의 positional encoding, BERT의 학습된 위치 임베딩 등 방식은 모델마다 다릅니다.)

### 2-4. 문맥 임베딩: 같은 단어도 다른 벡터

`bank`는 문맥에 따라 달라집니다.

- `deposit money in a bank`: 은행
- `sit on the river bank`: 강둑

정적 단어 임베딩은 하나의 단어에 하나의 벡터를 부여하지만,
문맥 임베딩은 **주변 문장을 반영한 표현**을 만듭니다.

> ⭐ Transformer가 주는 큰 변화는
> **“단어를 숫자로 바꾼다”를 넘어 “문맥 속 단어를 숫자로 바꾼다”** 는 점입니다.

### 2-5. Encoder와 Decoder

| 구조 | 주된 역할 | 대표적인 계열 |
|---|---|---|
| **Encoder-only** | 입력의 문맥 표현 생성 | BERT 계열 |
| **Decoder-only** | 이전 토큰을 바탕으로 다음 토큰 생성 | GPT 계열 |
| **Encoder-decoder** | 입력을 읽고 다른 시퀀스를 생성 | 번역·요약용 seq2seq 계열 |

원래 Transformer는 encoder-decoder 구조였지만,
**모든 Transformer가 두 부분을 모두 사용하는 것은 아닙니다.**

#### Encoder의 두 핵심 구성요소

> 🔎 **Transformer encoder를 이루는 ‘두 가지 핵심 요소’**
>
> encoder 한 층은 크게 두 부분으로 구성됩니다.
> 1. **Self-attention mechanism** — 토큰들끼리 서로의 관계를 파악
> 2. **Feed-forward 신경망** — 그렇게 모은 정보를 한 번 더 가공
>
> 혼동하기 쉬운 오답들:
> - "Masked self-attention + encoder-decoder attention" → 이것은 **decoder** 쪽 구성요소입니다.
> - "Positional encoding + contextual embedding" → 입력을 준비하는 단계이지 encoder 내부 블록 구조가 아닙니다.
> - "순차 처리 + 재귀 피드백" → RNN의 특징이지 Transformer가 아닙니다.

BERT 같은 양방향 encoder는 입력 안에서 현재 토큰의 앞뒤 문맥을 함께 활용할 수 있습니다.

#### Decoder의 특징과 Masked Self-attention

> 🔎 **Transformer decoder에 대한 정확한 서술**
>
> | 서술 | 맞을까? | 이유 |
> |---|---|---|
> | 문맥 임베딩을 다시 자연어로 풀어낸다 | ✅ | 압축된 의미 표현을 사람이 읽을 출력으로 변환합니다. |
> | 미래 단어를 못 보도록 **masked self-attention**을 쓴다 | ✅ | 아직 생성하지 않은 뒤쪽 토큰을 가립니다(커닝 방지). |
> | encoder-decoder attention 메커니즘을 포함한다 | ✅ | 입력(encoder 표현)과 출력을 연결하는 다리 역할을 합니다. |
> | RNN처럼 토큰을 순차 처리한다 | ❌ | Transformer는 RNN식 순차 처리에 의존하지 않습니다. |

생성 중에는 아직 만들지 않은 미래 토큰을 볼 수 없습니다.
따라서 현재 위치보다 뒤쪽을 가리는 **causal mask**를 사용합니다.

> 💡 **encoder의 attention vs decoder의 attention**
> - encoder: 일반 self-attention (문장 전체를 양방향으로 봄)
> - decoder: **masked** self-attention (현재 위치 기준 뒤쪽은 가림)
> 이 "가림(mask)"이 decoder의 가장 특징적인 부분입니다.

#### Encoder-decoder attention

번역 같은 과제에서는 decoder가 다음 출력 토큰을 만들면서,
encoder가 표현한 원문도 참고합니다.

> 🔎 **번역 예시에서 encoder-decoder attention이 하는 일**
>
> 번역에서 encoder-decoder attention의 역할은
> **decoder가 원본 데이터(원문)의 구조를 이해하고 참고하도록 돕는 것**입니다.
> 지금 어떤 출력 단어를 만들 때 원문의 어느 부분을 봐야 할지 연결해 줍니다.
>
> 자주 나오는 오답들:
> - "출력 문장의 단어 수를 입력과 똑같이 맞춘다" → ❌ 번역은 언어마다 단어 수가 다릅니다.
> - "positional encoding을 불필요하게 만든다" → ❌ 위치 정보는 여전히 필요합니다.
> - "각 단어를 독립적으로 처리한다" → ❌ attention은 단어들을 **연결**해서 봅니다.

### 🙋 Encoder와 Decoder는 어떻게 함께 작동할까?

> | 서술 | 맞을까? | 이유 |
> |---|---|---|
> | encoder가 입력 의미를 담은 문맥 임베딩을 만든다 | ✅ | encoder = "읽고 이해하는 쪽" |
> | decoder가 encoder의 표현을 바탕으로 출력을 생성한다 | ✅ | decoder = "그 이해를 받아 답을 쓰는 쪽" |
> | 둘 다 쌓거나 연결해 성능을 높일 수 있다 | ✅ | 층을 여러 개 쌓으면 더 복잡한 표현을 학습합니다. |
> | **항상 같은 수의 층**이어야 한다 | ❌ | encoder와 decoder의 층 수가 같아야 한다는 규칙은 없습니다. "항상"이 함정입니다. |

> ⭐ **역할 분담만 확실히 잡으면 됩니다. encoder = 이해, decoder = 생성.**
> 그리고 신경망 서술에서 "항상(always)", "똑같이(equal)" 같은 한정 표현은 대체로 오답입니다.

### 🙋 “Transformer는 병렬적이니까 문장도 한 번에 생성하나요?”

학습 중 알려진 시퀀스의 여러 위치를 병렬 계산할 수 있지만,
일반적인 자기회귀 생성은 **앞에서 생성한 토큰을 바탕으로 다음 토큰을 순차 생성**합니다.

> **병렬 학습 가능성**과 **생성 과정의 순차성**은 구분해야 합니다.

## <span style="color:#FEB99C;">3. BERT·DistilBERT·GPT: 같은 Transformer라도 목적이 다르다</span>

| 모델 | 핵심 성격 | 대표 활용 |
|---|---|---|
| **BERT** | 양방향 encoder | 분류·토큰 분류·문맥 표현 |
| **DistilBERT** | 지식 증류로 경량화한 BERT 계열 | 상대적으로 가벼운 분류·추론 |
| **RoBERTa** | BERT의 학습 방식을 개선한 계열 | 분류·표현 학습 |
| **GPT** | 자기회귀 decoder 계열 | 텍스트 생성·대화 |
| **Seq2seq 모델** | 입력과 출력 시퀀스를 연결 | 번역·요약 |

강의에서는 BERT-base가 약 **1억 1,000만 개**,
DistilBERT 계열이 약 **6,600만~6,700만 개** 수준의 파라미터를 가진다고 설명합니다.
정확한 수는 체크포인트와 출력 head에 따라 달라집니다.

> 💡 **한 줄 비유: BERT는 "이해 전문가", GPT는 "생성 전문가".**
> BERT는 Transformer의 **encoder만** 떼어내 **양방향**으로 문맥을 읽고,
> GPT는 **decoder만** 써서 앞에서 뒤로 **생성**합니다.

### 3-1. BERT는 무엇으로 사전학습할까?

대표적인 과제는 **Masked Language Modeling(MLM)** 입니다.

```text
The company reported strong [MASK] growth.
```

가려진 위치에 어떤 토큰이 들어갈지 주변 문맥으로 예측합니다.
이때 정답은 텍스트 자체에서 만들어지므로,
사람이 모든 문장에 별도 레이블을 붙일 필요가 없습니다.

> 💡 강의에서 비지도 학습이라고 넓게 설명하는 부분은,
> 엄밀하게는 **자기지도학습(Self-supervised learning)** 으로 이해하면 좋습니다.

기존의 단어 관계와 문맥 지식을 학습한 뒤,
분류용 head를 붙여 새 과제에 맞게 추가 학습할 수 있습니다.

> 🔎 **BERT의 흔한 활용 분야**
>
> BERT는 **"의미를 이해하는 능력"** 이 강해서 다양한 과제에 쓰입니다.
>
> | 활용 | 포함? | 설명 |
> |---|---|---|
> | 텍스트 분류 | ✅ | 스팸/정상, 긍정/부정 등. BERT의 대표 용도 |
> | Masked language modeling | ✅ | BERT가 학습하는 방식 그 자체 |
> | 의미 유사도(semantic similarity) | ✅ | 두 문서가 얼마나 비슷한 의미인지 측정 |
> | 텍스트 생성 | ✅ | 제한적으로 가능 (단 생성은 GPT가 더 강함) |
> | 이미지 인식(image recognition) | ❌ | BERT는 **텍스트 전용** 모델입니다. |
>
> ⚠️ **"이미지"가 들어간 보기는 BERT의 활용이 아닙니다.** BERT는 언어 모델이기 때문입니다.

### 3-2. Classification head란?

개념적으로 다음 구조입니다.

```text
텍스트
  ↓
Tokenizer
  ↓
Transformer backbone
  ↓
문맥을 반영한 표현
  ↓
Classification head
  ↓
클래스별 점수
```

- **Backbone**: 언어의 표현을 만드는 본체
- **Head**: 그 표현을 특정 과제의 출력으로 변환하는 부분

> 2부 실습에서는 **언어 표현은 재사용하고,
> 출력의 의미는 ‘긍정/부정’에서 ‘재무/비재무’로 바꾸는 것**입니다.

### 🔎 BERT로 감정 분류기를 설계할 때의 단계

| 단계 | 포함? | 설명 |
|---|---|---|
| 모델에 classification head를 추가 | ✅ | BERT 위에 "긍정/부정 판단기"를 얹습니다. |
| 목표 과제의 레이블 데이터로 fine-tuning | ✅ | 내 과제용 데이터로 가중치를 추가 학습합니다. |
| **BERT를 내 데이터로 처음부터(from scratch) 학습** | ❌ | 처음부터 학습하면 비용이 막대하고 전이학습의 장점이 사라집니다. |
| BERT의 출력 임베딩을 분류에 사용 | ✅ | BERT가 뽑은 의미 표현을 분류 재료로 씁니다. |

> ⭐ 핵심은 **"처음부터 만들지 않고, 이미 똑똑한 BERT를 가져와 조금만 더 가르친다"** 는 것입니다.
> 그래서 "from scratch로 학습한다"는 보기는 전이학습의 취지와 정반대라 오답입니다.

### 🔎 Masked language modeling의 실용적 활용

MLM은 "빈칸 채우기" 능력인데, 이 능력으로 여러 가지를 할 수 있습니다.

| 활용 | 포함? | 설명 |
|---|---|---|
| 문서의 엔트로피·"새 정보량" 측정 | ✅ | 빈칸을 쉽게 맞추면 뻔한 정보, 못 맞추면 새롭고 놀라운 정보로 해석할 수 있습니다. |
| 글·문법 품질 평가 | ✅ | 모델이 자연스럽다고 느끼면 좋은 글, 어색하면 품질 문제일 수 있습니다. |
| 이상한 문서 탐지 | ✅ | 예측과 많이 어긋나는 문서를 비정상으로 볼 수 있습니다. |
| **워드 클라우드 생성** | ❌ | 이것은 단순히 단어 빈도를 세어 그림을 그리는 작업으로, MLM과 무관합니다. |

> ⚠️ 다만 MLM 출력으로 문서 전체의 '엔트로피'를 얻는 방법은 자동으로 정해지지 않습니다.
> 어떤 토큰을 가리고 어떻게 집계할지 별도의 측정 설계가 필요합니다.

### 🔎 BERT가 문서 간 의미 유사도를 측정하는 방법

> | 방법 | 맞을까? | 이유 |
> |---|---|---|
> | encoder가 만든 **문맥적 임베딩을 비교**해서 | ✅ | 문서를 "의미 좌표(벡터)"로 바꿔 가까운 정도를 봅니다. |
> | 공통 단어 개수를 세어서 | ❌ | 단어가 달라도 의미가 같을 수 있습니다("기쁘다"="행복하다"). |
> | 문서 길이를 맞춰서 | ❌ | 길이는 의미와 무관합니다. |
> | 첫 문장만 사용해서 | ❌ | 전체 문맥을 봐야 합니다. |

> 💡 핵심은 **단어 겹침이 아니라 의미(임베딩)를 비교한다**는 점입니다.

### 🔎 BERT의 고급 응용·확장 사례

| 사례 | 포함? | 설명 |
|---|---|---|
| BERTopic (토픽 모델링) | ✅ | BERT 임베딩으로 문서들의 주제를 자동 분류하는 확장 도구 |
| 자동 요약(Automatic summarization) | ✅ | 긴 글의 핵심을 뽑아내는 과제. 의미 이해 능력을 활용 |
| 표절 탐지(Plagiarism detection) | ✅ | 두 글의 의미가 지나치게 비슷하면 표절을 의심 (유사도 응용) |
| **이미지 분할(Image segmentation)** | ❌ | 또다시 "이미지" 함정. BERT는 텍스트 전용입니다. |

> ⭐ 요약·표절 탐지·토픽 모델링은 모두 **"BERT가 의미를 잘 이해한다"** 는 공통 뿌리에서 나온 응용입니다.
> 반면 이미지 관련 과제는 BERT의 영역이 아닙니다.

### 🔎 MLM(BERT) 계열의 잘 알려진 한계

> | 서술 | 맞을까? | 이유 |
> |---|---|---|
> | **학습 데이터에 있는 편향을 그대로 이어받을 수 있다** | ✅ | 인터넷 텍스트의 편견을 함께 학습할 수 있습니다. |
> | 항상 편향이 없다 | ❌ | "항상 편향 없음"은 거짓입니다. |
> | 모든 과제에 거대한 레이블 데이터가 필요하다 | ❌ | 오히려 fine-tuning으로 적은 데이터면 됩니다. |
> | fine-tuning이 불가능하다 | ❌ | fine-tuning 가능성이 BERT의 최대 장점입니다. |

> ⚠️ 가장 잘 알려진 한계는 **"학습 데이터의 편향 전이"** 입니다(쓰레기가 들어가면 쓰레기가 나옵니다).

## <span style="color:#FEB99C;">4. Transformer의 네 가지 활용 분야</span>

### 4-1. 텍스트·토큰 분류

- 금융 문서 감성 분석 · 뉴스 주제 분류 · 스팸 탐지 · 공시의 특정 주제 식별 · 개체명 인식(NER)

단, NER은 문장 전체에 하나의 레이블을 붙이는 것이 아니라
**각 토큰에 사람·조직·장소 등의 레이블을 붙이는 token classification**입니다.

### 4-2. Masked Language Modeling

특정 문맥에서 어떤 단어가 자연스러운지 평가할 수 있습니다.
응용: 비정상적인 표현 탐지, 문장의 언어적 자연스러움 분석, 예상하기 어려운 정보 측정, 문서 품질·이상치 분석.

> ⚠️ 다만 MLM의 출력에서 문서 전체의 ‘엔트로피’를 얻는 방법은 자동으로 정해지지 않습니다.
> 어떤 토큰을 가리고 어떻게 집계할지 별도의 측정 설계가 필요합니다.

### 4-3. 의미 유사도

문서가 **같은 단어를 쓰는가**보다 **비슷한 의미를 전달하는가**를 비교합니다.
중복 문서 탐지, 의미 기반 검색, 문서 군집화, 표절 탐지의 보조 수단, 토픽 분석 등에 쓰이며 BERTopic과도 연결됩니다.

> 📝 BERTopic의 중심은 **임베딩 기반 토픽 모델링**입니다.
> 이를 곧바로 완성된 생성형 요약 모델과 같은 것으로 이해하지는 않는 편이 좋습니다.

### 4-4. 텍스트 생성

대화·질의응답·번역·요약·초안 작성.
다만 **자연스러운 문장 생성 능력**과 **사실 검증 능력**은 다릅니다.

> ⭐ **그럴듯함은 정확성의 보증이 아닙니다.**
> 모델 자체의 생성은 학습한 패턴과 입력에 기반합니다.
> 검색·데이터베이스 같은 도구를 연결하면 외부 정보를 참고할 수 있지만,
> 그래도 출처와 답변의 정확성은 검증해야 합니다.

---

# 🧪 2부. 실습 — 감성 모델을 재무 문장 분류기로 바꾸기

> 1부에서 다진 전이학습·Transformer·BERT 개념을 실제 파인튜닝에 적용합니다.
> 긍정/부정 감성 분류로 학습된 DistilBERT를 가져와,
> **"이 문장이 기업의 재무 성과를 말하는가?"** 를 판단하는 분류기로 바꿉니다.

---

## <span style="color:#FEB99C;">5. 실습 문제: 이 문장은 재무 성과를 말하는가?</span>

Demo B에서는 **Seeking Alpha 기사에서 추출한 문장**을 사용했습니다.
연구자들이 직접 레이블을 붙인 데이터이며,
과제는 문장이 **기업의 기초적 재무 성과(fundamental performance)** 와 관련되는지 판단하는 것입니다.

### 5-1. 데이터 구성

| 구분 | 크기 |
|---|---:|
| 전체 문장 | 5,000 |
| 레이블 있음 | 2,000 |
| 레이블 없음 | 3,000 |
| 레이블된 데이터 중 재무 문장 | 약 31.9% |

레이블은 **0: Non-financial / 1: Financial** 입니다.
여기서 `financial`은 **투자나 주식 이야기가 조금이라도 나온다**는 뜻이 아닙니다.

- “계속 보유하라” → 투자 의견이지만 실적 설명은 아닐 수 있음
- “매출이 증가했다” → 재무 성과
- “영업이익률이 개선됐다” → 재무 성과

> ⭐ **좋은 분류기의 출발점은 좋은 모델이 아니라 명확한 레이블 정의입니다.**
> ‘재무’라는 단어를 넓게 이해하면 사람이 붙인 정답과 모델의 목표가 어긋납니다.

### 5-2. 불균형 기준선부터 확인하기

재무 문장이 약 31.9%라면,
무조건 비재무라고 답하는 모델도 대략 **68.1% 정확도**를 얻습니다.
따라서 뒤에서 약 80% 정확도를 보더라도,
그 숫자만 보지 말고 **재무 클래스의 precision·recall·F1**을 확인해야 합니다.
(1편 §3-3의 "정확도 하나만 보지 말라"는 교훈과 이어집니다.)

### 5-3. Hugging Face 도구들의 역할

| 도구 | 역할 |
|---|---|
| **`transformers`** | 모델·토크나이저·Trainer·pipeline |
| **`datasets`** | 데이터셋 변환·저장·배치 처리 |
| **PyTorch** | 텐서 연산·자동 미분·학습 |
| **scikit-learn** | 분할·평가 지표 |

### 5-4. DataFrame → DatasetDict

아래 코드는 원본 DataFrame이 `df`이고, 원래 열 이름이 `sent`, `label`인 경우를 가정합니다.

```python
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict

df = df.rename(
    columns={"sent": "sentence", "label": "labels"}
)

labeled = df.loc[
    df["labels"].notna(),
    ["sentence", "labels"]
].copy()

labeled["labels"] = labeled["labels"].astype("int64")

train_df, val_df = train_test_split(
    labeled,
    test_size=0.2,
    random_state=42,
    stratify=labeled["labels"]
)

ds = DatasetDict({
    "train": Dataset.from_pandas(
        train_df, preserve_index=False
    ),
    "validation": Dataset.from_pandas(
        val_df, preserve_index=False
    )
})
```

2,000개를 80:20으로 나누면 훈련 **1,600개**, 검증 **400개**입니다.
`labels`는 분류용 정수로 지정합니다.

> ⚠️ **같은 기사에서 나온 문장끼리는 서로 비슷할 수 있습니다.**
> 실제 연구에서는 원문 기사 ID로 그룹 분할하거나, 시간·기업 단위의 분할을 고려해야 합니다.
> 위 무작위 분할 코드는 실습 구조를 보여 주는 예시이지, 모든 연구 상황에서 최선의 평가 설계는 아닙니다.

## <span style="color:#FEB99C;">6. Tokenizer: 기존 모델과 같은 언어로 입력하기</span>

이제는 1편의 `CountVectorizer`를 그대로 사용할 수 없습니다. 이유는 두 가지입니다.

1. **사전학습 모델이 기대하는 어휘와 토큰 ID가 이미 정해져 있음**
2. **문맥을 계산하려면 순서를 유지한 토큰 시퀀스가 필요함**

### 6-1. 모델과 Tokenizer는 한 쌍이다

어떤 모델이 토큰 ID 100번을 특정 단어 조각으로 학습했는데,
새 tokenizer가 같은 100번을 전혀 다른 단어에 할당하면 모델은 입력을 엉뚱하게 해석합니다.

> 💡 **토큰 ID는 숫자의 크기보다 ‘어느 어휘 항목을 가리키는지’가 중요합니다.**

```python
from transformers import AutoTokenizer

checkpoint = (
    "distilbert-base-uncased-finetuned-sst-2-english"
)

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
```

위 체크포인트 ID는 녹취에 완전하게 제시되지 않은 문자열을 재현 코드에서 명시한 예시입니다.
`AutoTokenizer`는 체크포인트 설정에 맞는 tokenizer를 선택합니다.

### 6-2. 토큰 수와 단어 수는 다르다

#### ① 특수 토큰
BERT 계열 tokenizer는 보통 다음을 추가합니다.

- `[CLS]`: 시퀀스 표현·분류 등에 사용
- `[SEP]`: 시퀀스의 끝이나 경계 표시
- `[PAD]`: 길이를 맞추기 위한 패딩

#### ② Subword 토큰화
단어가 여러 조각으로 분리될 수 있습니다. (예: `disagree` → `dis` + `agree`)
실제 분리 결과는 tokenizer의 어휘에 따라 다릅니다.

> ⭐ **최대 512 tokens는 최대 512 words와 같은 뜻이 아닙니다.**

### 6-3. Padding·Truncation·Attention mask

1편 §5(개념)에서 본 padding/truncation이 Transformer 입력에서 실제로 이렇게 쓰입니다.

| 개념 | 역할 |
|---|---|
| **Padding** | 짧은 입력의 길이를 맞춤 |
| **Truncation** | 너무 긴 입력을 잘라냄 |
| **Attention mask** | 실제 토큰과 패딩을 구분 |

```python
MAX_LENGTH = tokenizer.model_max_length

def tokenize_batch(batch):
    return tokenizer(
        batch["sentence"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH
    )

tokenized = ds.map(
    tokenize_batch,
    batched=True,
    remove_columns=["sentence"]
)
```

이 체크포인트 계열은 일반적으로 최대 512 토큰을 사용하지만,
**모든 Transformer의 길이 제한이 같은 것은 아닙니다.**

출력에는 보통 다음 정보가 들어갑니다.

```text
input_ids       : 토큰 ID 시퀀스
attention_mask  : 실제 토큰 1, 패딩 위치 0
labels          : 정답 클래스
```

> 🙋 **모든 입력을 최대 길이까지 채워야 하나요?** 아닙니다.
> 강의는 이해하기 쉬운 **고정 최대 길이 padding**을 쓰지만,
> 실무에서는 각 배치에서 가장 긴 입력에만 맞추는 **동적 padding**도 많이 사용합니다.

> ⚠️ **Truncation은 단순 편의 기능이 아니라 정보 손실입니다.**
> 긴 문서의 뒤쪽에 중요한 실적 정보가 있다면 앞부분만 남기는 방식으로는 놓칠 수 있습니다.
> 문장·문단 분할이나 겹치는 구간을 사용하는 전략이 필요할 수 있습니다.

## <span style="color:#FEB99C;">7. Fine-tuning: 모델·학습 설정·평가·조기 종료를 연결한다</span>

파인튜닝의 전체 흐름은 다음과 같습니다.

```text
사전학습 모델 + 토큰화된 데이터 + TrainingArguments + 평가 함수 + 조기 종료
   ↓
Trainer.train()
```

### 7-1. 출력의 의미를 바꿀 때 주의할 것

기존 모델의 출력은 `NEGATIVE / POSITIVE`,
새 과제의 출력은 `NON_FINANCIAL / FINANCIAL` 입니다.
둘 다 클래스가 2개라고 해서 **같은 과제**는 아닙니다.

> ⚠️ **`num_labels=2`라고 쓰면 기존 분류층이 자동으로 새로 만들어질까요?**
>
> 기존 모델도 출력이 2개라면 크기가 같아서 기존 head 가중치가 그대로 로드될 수 있습니다.
> **`ignore_mismatched_sizes=True`도 크기가 실제로 다를 때 관련되는 옵션**이지,
> 과제의 의미가 달라졌다는 이유만으로 자동 초기화해 주는 기능은 아닙니다.

아래는 이 DistilBERT 구조에 한정해 분류 head를 명시적으로 초기화하는 예시입니다.

```python
from transformers import (
    AutoModelForSequenceClassification,
    set_seed
)

set_seed(42)

id2label = {0: "NON_FINANCIAL", 1: "FINANCIAL"}
label2id = {label: idx for idx, label in id2label.items()}

model = AutoModelForSequenceClassification.from_pretrained(
    checkpoint,
    num_labels=2,
    id2label=id2label,
    label2id=label2id
)

# DistilBERT 분류 모델의 head를 새 과제용으로 초기화
# 다른 모델 계열에서는 해당 모듈 이름과 구조가 다를 수 있음
model.pre_classifier.reset_parameters()
model.classifier.reset_parameters()
```

다른 방법으로는 **분류 과제에 아직 맞추지 않은 base checkpoint**에서 시작해 새 head를 학습할 수도 있습니다.
또한 다음 두 전략도 구분해야 합니다.

| 전략 | 업데이트 범위 |
|---|---|
| **고정된 표현 + head 학습** | Backbone은 동결, head만 학습 |
| **전체 파인튜닝** | Backbone과 head를 함께 업데이트 |

위 코드는 별도로 동결하지 않았으므로 **전체 파인튜닝**입니다.

### 7-2. 주요 학습 설정

| 항목 | 설정·설명 |
|---|---|
| 최대 epochs | 5 |
| 훈련 batch size | 16 |
| 평가 batch size | 64 |
| 평가 간격 | 25 training steps |
| 조기 종료 patience | 3번의 평가 |
| 학습률 | 사전학습 모델의 설정을 참고한 작은 값 |
| 재현성 | 여러 라이브러리의 seed 설정 |

1,600개 훈련 문장, batch size 16이면 $1\text{ epoch}=100\text{ steps}$,
5 epochs면 최대 $100\times5=500\text{ steps}$ 입니다.

> 🔎 **Warm-up은 “평가를 늦게 시작하는 것”이 아니다**
>
> **Learning-rate warm-up**은 학습 초반에 학습률을 작은 값에서 목표값까지 점진적으로 올리는 전략입니다.
> - `warmup_steps`: 학습률 스케줄
> - `eval_steps`: 평가 간격
> 둘은 다른 설정입니다.

### 7-3. Logits를 레이블로 바꾸기

모델의 원시 출력은 보통 **logits**입니다. 아직 0~1 확률로 바꾸기 전의 클래스별 점수입니다.

```text
NON_FINANCIAL: -0.8
FINANCIAL:     1.4
```

둘 중 큰 값을 가진 클래스가 예측 결과입니다.

```python
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    return {
        "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(
            labels,
            predictions,
            average="macro",
            zero_division=0
        )
    }
```

> 💡 **클래스만 고를 때는 Softmax를 먼저 계산하지 않아도 됩니다.**
> Softmax는 점수의 순서를 유지하므로 logits의 `argmax`와 결과가 같습니다.
> 확률값이 필요할 때 Softmax를 적용합니다.

### 7-4. Trainer 설정 예시

```python
from transformers import (
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)

args = TrainingArguments(
    output_dir="./financial_checkpoints",

    learning_rate=2e-5,  # 재구성 예시값
    warmup_steps=5,
    num_train_epochs=5,

    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,

    eval_strategy="steps",
    eval_steps=25,

    save_strategy="steps",
    save_steps=25,
    save_total_limit=2,

    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,

    seed=42,
    data_seed=42,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    compute_metrics=compute_metrics,
    callbacks=[
        EarlyStoppingCallback(early_stopping_patience=3)
    ]
)

trainer.train()
```

> 📝 `transformers` 버전에 따라 `eval_strategy` 대신 `evaluation_strategy`를 쓰는 경우가 있습니다.
> 설치한 버전의 API를 확인해야 합니다.

### 7-5. 왜 학습·평가 batch size가 다를까?

학습은 순전파·역전파·optimizer 상태에 메모리를 씁니다.
평가는 역전파가 필요 없으므로 같은 장치에서도 더 큰 배치를 쓸 여지가 있습니다.

> **“GPU를 쓰면 무조건 정확히 10배 빠르다”처럼 고정된 배수로 이해하지는 않는 편이 좋습니다.**
> 실제 속도는 모델 크기·시퀀스 길이·GPU 메모리·데이터 로딩 등에 따라 달라집니다.

## <span style="color:#FEB99C;">8. 학습 결과: 약 80% 정확도를 어떻게 읽을까?</span>

강의에서는 최대 500 steps를 계획했지만, 조기 종료로 약 **300 steps**에서 학습이 끝났습니다.

| 평가 시점 | 검증 정확도 |
|---|---:|
| 초기 평가 | 약 64% |
| 이후 | 약 72% |
| 200 steps | 약 81% |
| 225 steps | 약 81.25% |
| 이후 평가 | 약 80.5% → 79.75% → 80% |

최고 성능 이후 세 번의 평가에서 개선되지 않아 종료한 것입니다.

### 8-1. `patience=3`은 3 epochs가 아니다

25 steps마다 평가하면:

```text
최고 기록
  ↓ 25 steps 후: 개선 없음 1회
  ↓ 25 steps 후: 개선 없음 2회
  ↓ 25 steps 후: 개선 없음 3회 → 종료
```

즉, patience는 **callback이 평가 결과를 확인하는 횟수**를 기준으로 해석해야 합니다.

### 8-2. 최고 성능과 마지막 성능을 구분하자

- 최고 검증 정확도: 약 **81.25%**
- 마지막 시점의 정확도: 약 **80%**

최고 체크포인트를 복원하는 설정(`load_best_model_at_end`)을 쓰면,
마지막 시점의 모델이 아니라 최적 시점의 모델을 활용할 수 있습니다.

> 결과를 기록할 때는 **“어떤 체크포인트를 평가한 수치인가?”** 를 명확히 해야 합니다.

### 8-3. 클래스별 성능

강의의 400개 검증 문장 평가에서는:

- 전체 정확도: 약 **80%**
- 비재무 클래스가 상대적으로 더 잘 분류됨
- 재무 클래스 재현율: 약 **75%**, F1: 약 **0.7**

> 재무 클래스 재현율 약 75%는 **실제 재무 문장 네 개 중 약 세 개를 찾아냈다**는 뜻입니다.
> 정확도 80%는 무조건 비재무로 답하는 약 68% 기준선보다 낫지만,
> 재무 문장을 놓치는 비용이 크다면 충분하지 않을 수 있습니다.

### ⚠️ 이 400개는 독립적인 최종 테스트셋일까?

이 데이터는 학습 중 조기 종료와 성능 확인에 사용되었습니다.
따라서 엄밀히는 **모델 선택용 검증셋**입니다. 코드나 자료에서 `test`라고 이름 붙였더라도 실제 역할이 중요합니다.

> ⭐ **검증셋에서 선택한 모델을 같은 검증셋으로 다시 평가하면,
> 그 결과를 완전히 독립적인 최종 테스트 성능이라고 부르기 어렵습니다.**
> 실제 연구·배포에서는 별도 테스트셋을 확보하는 편이 좋습니다.

### 8-4. 성능 개선 방향

| 방향 | 기대 효과 |
|---|---|
| 파싱 오류·깨진 문장 정리 | 입력 노이즈 감소 |
| 레이블 기준 점검 | 모호한 정답 감소 |
| 금융 도메인 모델 비교 | 과제에 맞는 표현 활용 |
| 클래스 가중치·재표집 | 재무 클래스의 손실 비중 조절 |
| 적절한 데이터 증강 | 학습 사례 보완 |
| 임계값 조정 | Precision·Recall 균형 조절 |
| 그룹·시간 기반 평가 | 실제 적용 상황과 가까운 검증 |

> **좋은 사전학습 모델도 나쁜 입력과 불명확한 정답을 완전히 해결해 주지는 못합니다.**

## <span style="color:#FEB99C;">9. 저장·평가·새 데이터 적용: Pipeline으로 연결하기</span>

### 9-1. 레이블 매핑과 Tokenizer까지 함께 저장

```python
model = trainer.model

model.config.id2label = id2label
model.config.label2id = label2id

save_dir = "./final_financial_classifier"

trainer.save_model(save_dir)
tokenizer.save_pretrained(save_dir)
```

> ⚠️ **레이블 이름을 바꾸는 것과 분류기를 학습하는 것은 다릅니다.**
> `POSITIVE`를 `FINANCIAL`로 이름만 바꾼다고 감성 모델이 재무 분류 모델로 바뀌지는 않습니다.
> 새 과제로 학습한 가중치와 정확한 레이블 매핑이 함께 있어야 합니다.

### 9-2. `pipeline`: 토큰화부터 후처리까지

```python
import torch
from transformers import pipeline

financial_classifier = pipeline(
    task="text-classification",
    model=save_dir,
    tokenizer=save_dir,
    device=0 if torch.cuda.is_available() else -1
)
```

`pipeline`은 보통 `원문 → 토큰화 → 텐서 변환 → 모델 추론 → 레이블·점수 반환` 을 묶어 줍니다.

### 9-3. 검증 데이터 평가

```python
import pandas as pd
from sklearn.metrics import classification_report

outputs = financial_classifier(
    val_df["sentence"].tolist(),
    batch_size=64,
    truncation=True,
    max_length=MAX_LENGTH
)

pred_df = pd.DataFrame(outputs)
pred_ids = pred_df["label"].map(label2id).to_numpy()

print(
    classification_report(
        val_df["labels"].to_numpy(),
        pred_ids,
        labels=[0, 1],
        target_names=["non_financial", "financial"],
        zero_division=0
    )
)
```

> 정답과 예측을 비교할 때는 **행 순서를 유지**해야 합니다.
> index가 달라도 배열로 변환한 뒤 동일 순서로 비교하면 의도하지 않은 index 정렬을 피할 수 있습니다.

### 9-4. 레이블 없는 데이터에 적용

```python
unlabeled = df.loc[df["labels"].isna(), ["sentence"]].copy()

sample = unlabeled.sample(n=5, random_state=42).reset_index(drop=True)

outputs = financial_classifier(
    sample["sentence"].tolist(),
    truncation=True,
    max_length=MAX_LENGTH
)

review = pd.concat([sample, pd.DataFrame(outputs)], axis=1)
print(review)
```

강의에서 확인한 다섯 문장:

- 비즈니스 비유·웹사이트 개편 → 비재무
- 기록적인 매출·매출 성장률 → 재무
- 영업이익률·영업현금흐름 마진 → 재무

전반적으로 납득할 만한 결과였지만, 다음을 구분해야 합니다.

> **“다섯 문장을 읽어 보니 그럴듯했다”와
> “새 데이터에서 높은 정확도가 검증됐다”는 같은 말이 아닙니다.**
> 레이블 없는 데이터에는 비교할 정답이 없으므로, 이는 정성적 점검이지 공식 성능 평가가 아닙니다.

### 🔎 `score=0.95`면 정말 95% 확률로 맞을까?

기본적인 단일 레이블 분류 pipeline의 score는 보통 선택된 클래스에 대한 모델의 확률형 점수입니다.
하지만 다음을 보장하지는 않습니다.

- 점수 0.95인 예측의 실제 정답률이 정확히 95%
- 새로운 도메인에서도 확신이 신뢰할 만함
- 높은 점수의 오분류가 없음

또한 `NON_FINANCIAL`이 선택됐다면, 그 score는 **비재무 클래스의 점수**이지 재무 확률이 아닙니다.

> 💡 확률 자체로 의사결정하려면 클래스별 확률을 확인하고
> calibration·임계값·오분류 비용까지 검토해야 합니다.

## <span style="color:#FEB99C;">10. 모델이 배운 편향도 함께 이전된다</span>

전이학습은 지식만 가져오는 것이 아닙니다. **기존 데이터의 편향도 가져올 수 있습니다.**
(1편 §6 윤리적 고려사항이 Transformer 전이학습에서 더 크게 작동합니다.)

강의는 직업과 성별의 연관 같은 예를 듭니다.
대규모 말뭉치에 특정 연결이 반복되면 모델도 그 패턴을 학습할 수 있습니다.

### 10-1. 무엇을 점검할까?

| 점검 대상 | 질문 |
|---|---|
| 데이터 대표성 | 특정 집단·주제·문체에 치우쳤나? |
| 레이블 | 사람의 일관되지 않은 판단이 들어갔나? |
| 입력 특성 | 이름·성별·출처가 불필요한 지름길인가? |
| 하위 집단 성능 | 전체 평균 뒤에 큰 성능 차이가 숨었나? |
| 적용 환경 | 학습 데이터와 실제 입력이 얼마나 다른가? |

가능한 대응은: 민감한 표현을 바꾼 비교 문장으로 테스트, 하위 집단별 오류 분석,
레이블 가이드 정비, 적용 도메인의 대표적인 평가셋 구축, 중요한 판단에 사람의 검토 추가 등입니다.

> ⚠️ 다만 민감한 단어를 지우기만 해도 편향이 완전히 제거되지는 않습니다.
> 다른 표현이 대리변수 역할을 할 수 있기 때문입니다.

### 10-2. 운영에서는 정확도 외에도 필요한 것

- 모델·tokenizer·라이브러리 버전 기록
- 데이터와 모델 접근 권한 관리
- 민감한 문장을 외부 서비스로 보내는지 확인
- 새 데이터의 분포 변화 모니터링
- 낮은 확신·고비용 사례에 대한 검토 절차

> **모델을 잘 학습시키는 것과 모델을 책임 있게 운영하는 것은 연결되어 있지만 다른 과제**입니다.

## <span style="color:#FEB99C;">11. Week 7 최종 정리</span>

### 1편: 신경망을 직접 만들기

- **ANN 구조** = 입력층 → 은닉층 → 출력층 / **학습 대상** = 가중치와 bias
- **비선형성의 핵심** = 활성화 함수 / **학습 과정** = 순전파 → 손실 → 역전파 → 업데이트
- **Keras** = 구조·학습 방식·훈련을 분리해 구성 / **조기 종료** = 검증 성능이 개선되지 않으면 중단
- **높은 정확도** = 누수·출처·주제 단서부터 점검 / **외부 적용** = 다른 분포에서도 통하는지 확인

### 2편: 이미 학습한 언어 지식을 가져오기

**[개념]**
- **전이학습** = 기존의 표현과 가중치를 새 과제에 재사용 (적은 데이터로 적응, 수백만 레이블 불필요)
- **파인튜닝** = 새로운 목표로 가중치를 추가 업데이트 (처음부터 학습이 아님)
- **Transformer 핵심 특징** = Self-attention·위치 정보·문맥 표현 (순차 처리는 RNN 특징)
- **학습 목표** = Transformer가 기존 신경망과 무엇이 다른지 구별
- **BERT / GPT** = 같은 Transformer 계열이지만 구조·목적이 다름 (encoder=이해, decoder=생성)
- **Encoder 구성** = Self-attention + feed-forward 신경망
- **Decoder 특징** = masked self-attention(미래 가림) + encoder-decoder attention
- **encoder-decoder attention** = decoder가 원문 구조를 참고하도록 연결
- **BERT 활용** = 분류·유사도·요약·표절탐지·토픽모델링 (이미지는 불가), **한계** = 편향 전이

**[실습]**
- **Tokenizer와 모델** = 어휘·ID 체계가 맞아야 하는 한 쌍
- **Padding / Truncation** = 길이 맞춤과 길이 제한 처리 / **Classification head** = 과제별 출력
- **Logits → Argmax** = 클래스 예측 / **Trainer** = 학습 설정·데이터·지표·callback 연결
- **Pipeline** = 원문에서 예측까지의 추론 흐름을 묶음
- **검증셋과 테스트셋** = 이름이 아니라 실제 사용 방식으로 구분
- **확률 점수·높은 정확도** = 사실성·신뢰성을 자동으로 보장하지 않음

### ⚖️ Demo A와 Demo B 한눈에

| 항목 | Demo A: 가짜 뉴스 ANN | Demo B: 재무 문장 Transformer |
|---|---|---|
| 입력 | 정규화한 Count DTM | 순서를 유지한 토큰 ID |
| 출발점 | 무작위 초기화 | 사전학습 가중치 |
| 구조 | 작은 Dense 신경망 | DistilBERT + 분류 head |
| 주요 목표 | 가짜 / 진짜 | 재무 / 비재무 |
| 강의의 검증 성능 | 약 98% | 약 80% |
| 중요한 한계 | 출처·주제 변화에 취약할 수 있음 | 불균형·파싱 오류·레이블 모호성 |
| 공통 교훈 | 데이터와 평가 설계가 중요 | 데이터와 평가 설계가 중요 |

> ⚠️ **98%와 80%를 보고 ANN이 Transformer보다 낫다고 결론 내리면 안 됩니다.**
> 데이터·정답 정의·난이도·평가 조건이 모두 다른 문제입니다.

> 🎯 Week 7 한 줄 요약:
> **“좋은 NLP 모델은 큰 신경망을 쓰는 데서 끝나지 않는다. 무엇을 전이했고, 어떤 데이터를 배웠으며, 새로운 환경에서 무엇을 틀리는지까지 확인해야 한다.”**

