# 토큰 (Token)

> LLM의 모든 비용/처리량/성능 지표가 토큰 단위로 표현된다. 토큰이 정확히 무엇이고, 왜 단어가 아닌가.

## 정의

- **토큰** = LLM이 처리하는 최소 단위. 보통 단어보다 작거나 같다.
- **Tokenizer** = 텍스트를 토큰 ID 시퀀스로 변환하는 알고리즘. 모델별로 다름.
- 모델 입장에서 텍스트는 존재하지 않음. 토큰 ID 시퀀스(정수 배열)만 본다.

## 왜 단어가 아닌가

단어 단위로 만들면:
- vocab가 너무 커짐 (영어만 해도 수십만 단어 + 변형)
- 본 적 없는 단어(OOV)에 약함
- 다국어 처리 어려움

글자 단위로 만들면:
- 시퀀스 길이가 너무 길어짐 (FLOPs 폭발)
- 의미 단위가 너무 작아 학습 비효율

**Subword tokenization**이 타협점. 빈도 높은 글자 조합을 단일 토큰으로 묶음.

## 주요 tokenization 알고리즘

| 알고리즘 | 사용처 | 핵심 |
|---|---|---|
| BPE (Byte-Pair Encoding) | GPT-2, GPT-3, GPT-4, LLaMA | 빈도 높은 byte/char pair를 반복 병합해 vocab 생성 |
| WordPiece | BERT | BPE와 유사하나 likelihood 기반 병합 |
| SentencePiece | T5, LLaMA, mBART | 공백도 토큰의 일부로 다룸. language-agnostic |
| tiktoken (OpenAI) | GPT-3.5/4 | OpenAI의 BPE 변종 (cl100k_base 등) |

## 실제 예시 (tiktoken cl100k_base 기준)

```
"Hello world"       -> 2 tokens   ["Hello", " world"]
"ChatGPT"           -> 2 tokens   ["Chat", "GPT"]
"안녕하세요"         -> 7 tokens   (한국어 1글자 = 2-3 토큰)
"GPU 클러스터"       -> 9 tokens
"floating point"    -> 2 tokens
"flop"              -> 1 token
"flops"             -> 1 token
"FLOPS"             -> 2 tokens   (대문자 분해)
```

### 언어별 토큰 효율

| 언어 | 1 token이 평균 몇 글자 | 1 단어가 평균 몇 토큰 |
|---|---|---|
| 영어 | ~4 chars | ~1.3 tokens/word |
| 한국어 | ~1 char (자주 1글자=2-3 tokens) | - |
| 코드 | 가변 (들여쓰기, 변수명) | - |

**한국어는 영어 대비 같은 정보량에 1.5-2배 토큰을 쓴다.** API 비용도 그만큼 더 든다.

## LLM에서 토큰이 등장하는 모든 곳

### 1. 비용 단위

OpenAI/Anthropic API: input tokens / output tokens 별로 과금
- 예: GPT-4 input $30/1M tokens, output $60/1M tokens
- 한국어 입력은 영어보다 1.5-2배 비쌈

### 2. 컨텍스트 윈도우 (context window)

모델이 한 번에 받을 수 있는 최대 토큰 수.
- GPT-4: 128K tokens (~96K 영어 단어, ~50K 한국어 글자)
- Claude 3/4: 200K-1M tokens
- Llama-3 8B: 8K tokens
- Gemini 1.5: 2M tokens

### 3. 학습 데이터 양

데이터셋 크기를 토큰 수로 표현.
- GPT-3: ~300B tokens
- LLaMA-3: ~15T tokens
- Chinchilla scaling law: 모델 1 param 당 ~20 tokens 학습이 효율적

### 4. 추론 throughput 단위

- **TTFT (Time To First Token)**: 사용자가 보낸 prompt 끝나고 첫 출력 토큰까지의 latency. Prefill 단계 시간.
- **TPOT (Time Per Output Token)** = ITL (Inter-Token Latency): 두 출력 토큰 사이 평균 시간. Decode 단계 시간.
- **Tokens/sec**: 추론 시스템이 초당 생성 가능한 토큰 수. 시스템 throughput 표준 지표.
- 사용자 경험: TTFT < 1s, TPOT < 50ms (인간 읽기 속도) 정도가 좋다고 봄

### 5. FLOPs 계산

토큰이 FLOPs와 직결. 책에서 "100T model을 29T tokens 학습 시 1.2e29 FLOPs" 같은 추정이 나오는 근거.

| 작업 | FLOPs per token (근사) |
|---|---|
| Dense forward pass | ~2 * params |
| Dense forward + backward (학습) | ~6 * params |
| MoE forward (active params만) | ~2 * active_params |

예시 계산:
```
Llama-3 405B 학습 1T tokens:
  6 * 405e9 * 1e12 = 2.43e24 FLOPs = 2.43 ZettaFLOPs
H100 SXM 1대 BF16 peak = 989 TFLOPS = 9.89e14 FLOPs/s
이론적 시간 = 2.43e24 / 9.89e14 = 2.46e9 s = 78 GPU-년

실제: 16K H100, 약 60일 = ~2.6M GPU-시간 = 297 GPU-년
MFU = 78 / 297 = ~26% (대략. 공개 자료에선 38-43% MFU 보고)
```

(근사가 거친 이유: backward pass는 forward의 정확히 2배가 아니고, attention은 토큰 수에 quadratic이며, recompute로 fwd가 두 번 실행되기도 함)

### 6. Prefill vs Decode 구분

LLM 추론은 두 단계로 나뉨:
- **Prefill**: 입력 prompt 토큰 N개를 한 번에 처리 (병렬). 1 step.
  - 시간 ~ N (compute-bound, GPU 잘 활용)
- **Decode**: 출력 토큰을 1개씩 autoregressively 생성. M번 반복.
  - 매 step 시간 ~ KV cache 크기 (memory-bound, GPU stall 많음)

이 비대칭이 책 Ch 17, 18의 disaggregated prefill-decode 동기.

## 응용/적용 아이디어

- 학습 워크로드 throughput 보고 시 "초당 iterations" 대신 "tokens/s"로 표현하면 LLM 산업 표준과 align됨
- 비용 환산이 직관적: lost time -> lost tokens -> lost dollars (API 단가 또는 GPU-시간 단가)
- 한국어 데이터 비중이 큰 시스템에서는 영어 대비 1.5-2배 토큰 / 비용 / 메모리 / 시간 가정 필요

## 자주 쓰는 어림셈

| 변환 | 근사 |
|---|---|
| 영어 1 word | ~1.3 tokens |
| 영어 1 token | ~4 chars |
| 한국어 1 글자 | ~2-3 tokens |
| 1 page A4 영어 | ~500 tokens |
| 1 page A4 한국어 | ~700-1000 tokens |
| 책 1권 | ~80K-150K tokens |
