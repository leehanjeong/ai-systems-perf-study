# Glossary

책 전체에서 누적되는 용어 통합 인덱스. 챕터 노트에도 "새 용어" 표가 있지만 그건 그 챕터 self-contained용. 본 파일은 cross-chapter reference.

용어 추가 시: 알파벳순 유지, 등장한 챕터 노트로 링크.

## A

| 용어 | 한국어 | 한 줄 정의 | 등장 |
|---|---|---|---|

## B

| 용어 | 한국어 | 한 줄 정의 | 등장 |
|---|---|---|---|
| BPE (Byte-Pair Encoding) | - | 빈도 높은 byte/char pair를 반복 병합하는 subword tokenization | [concepts/tokens](concepts/tokens.md) |

## C

| 용어 | 한국어 | 한 줄 정의 | 등장 |
|---|---|---|---|
| CP (Context Parallelism) | 컨텍스트 병렬 | 긴 시퀀스를 토큰 차원으로 나눠 GPU 분산 | [ch01](ch01_intro.md) |

## D

| 용어 | 한국어 | 한 줄 정의 | 등장 |
|---|---|---|---|
| DCGM | - | NVIDIA의 GPU monitoring 데이터 모델. PROF_* 시리즈가 honest 활용도 지표 | [concepts/flops_and_utilization](concepts/flops_and_utilization.md) |
| DP (Data Parallelism) | 데이터 병렬 | 같은 모델 복제, 다른 미니배치 | [ch01](ch01_intro.md) |
| DualPipe | - | DeepSeek의 bidirectional pipeline parallelism. fwd/bwd를 통신과 overlap | [ch01](ch01_intro.md) |

## E

| 용어 | 한국어 | 한 줄 정의 | 등장 |
|---|---|---|---|
| EP (Expert Parallelism) | 전문가 병렬 | MoE expert를 GPU 분산 | [ch01](ch01_intro.md) |
| EPLB | - | Expert Parallelism Load Balancer. hot expert 복제로 부하 균형 | [ch01](ch01_intro.md) |
| ETTR (Effective Training-Time Ratio) | - | productive_training_time / total_wall_clock_time. Meta 2025 | [ch01](ch01_intro.md), [concepts/goodput_calculation](concepts/goodput_calculation.md) |

## F

| 용어 | 한국어 | 한 줄 정의 | 등장 |
|---|---|---|---|
| FlashAttention | - | tiling 기반 attention. 메모리 read/write 최소화로 2-4x 속도 | [ch01](ch01_intro.md) |
| FLOP / FLOPS | 부동소수점 연산 (수) / 처리율 | FLOP은 단위 연산, FLOPS는 초당 처리율 | [concepts/flops_and_utilization](concepts/flops_and_utilization.md) |
| FSDP (Fully Sharded Data Parallel) | - | 모델 파라미터/optimizer state도 GPU 분산하는 DP 변종 | [ch01](ch01_intro.md) |

## G

| 용어 | 한국어 | 한 줄 정의 | 등장 |
|---|---|---|---|
| Goodput | 유용한 처리량 | 시스템이 실제로 useful work에 쓴 시간 비율 (Meta ETTR) | [ch01](ch01_intro.md), [concepts/goodput_calculation](concepts/goodput_calculation.md) |
| GPU-Util | - | nvidia-smi 지표. kernel이 떠 있던 시간만 측정. SM stall 못 잡음 | [concepts/flops_and_utilization](concepts/flops_and_utilization.md) |

## H

| 용어 | 한국어 | 한 줄 정의 | 등장 |
|---|---|---|---|
| HBM (High Bandwidth Memory) | - | GPU 칩 옆 stacked DRAM. B200=192GB, B300=288GB | [ch01](ch01_intro.md) |
| HFU (Hardware FLOPs Utilization) | - | 실제 실행 FLOPs / peak. recompute 등 redundant compute 포함 | [concepts/flops_and_utilization](concepts/flops_and_utilization.md), [concepts/goodput_calculation](concepts/goodput_calculation.md) |

## M

| 용어 | 한국어 | 한 줄 정의 | 등장 |
|---|---|---|---|
| Mechanical sympathy | - | hardware를 깊이 이해한 software 설계. Martin Thompson | [ch01](ch01_intro.md) |
| MFU (Model FLOPs Utilization) | - | 모델 정의의 useful FLOPs / peak FLOPS. 일반 학습에 38-50% | [concepts/flops_and_utilization](concepts/flops_and_utilization.md), [concepts/goodput_calculation](concepts/goodput_calculation.md) |
| MIG (Multi-Instance GPU) | - | GPU를 여러 인스턴스로 분할 | [ch01](ch01_intro.md) |
| MLA (Multi-Head Latent Attention) | - | DeepSeek attention. H800에서 FlashAttention보다 빠름 | [ch01](ch01_intro.md) |
| MoE (Mixture of Experts) | 전문가 혼합 모델 | 토큰별로 일부 expert만 활성화 (DeepSeek-V3: ~37B/680B active) | [ch01](ch01_intro.md) |

## N

| 용어 | 한국어 | 한 줄 정의 | 등장 |
|---|---|---|---|
| NCCL | "nickel" | GPU collective 통신 라이브러리 (all-reduce, all-gather 등) | [ch01](ch01_intro.md) |
| NIXL (NVIDIA Inference Xfer Library) | - | 추론용 point-to-point 데이터 이동 (KV cache 등) | [ch01](ch01_intro.md) |
| NVL72 | - | rack 안 72-GPU NVLink 도메인 (36 Grace + 72 Blackwell) | [ch01](ch01_intro.md) |

## P

| 용어 | 한국어 | 한 줄 정의 | 등장 |
|---|---|---|---|
| PP (Pipeline Parallelism) | 파이프라인 병렬 | 모델 레이어를 GPU 분산해 micro-batch로 흐르기 | [ch01](ch01_intro.md) |
| Prefill | - | LLM 추론 첫 단계. 입력 prompt 전체를 한 번에 처리 (compute-bound) | [concepts/tokens](concepts/tokens.md) |

## S

| 용어 | 한국어 | 한 줄 정의 | 등장 |
|---|---|---|---|
| SentencePiece | - | language-agnostic subword tokenizer. T5/LLaMA 등 사용 | [concepts/tokens](concepts/tokens.md) |
| SLO (Service Level Objective) | - | 응답 시간/품질 목표. 어긴 요청은 inference goodput에서 제외 | [concepts/goodput_calculation](concepts/goodput_calculation.md) |
| Speed of light | 이론 최대치 | NVIDIA 용어. 하드웨어가 낼 수 있는 peak | [ch01](ch01_intro.md) |

## T

| 용어 | 한국어 | 한 줄 정의 | 등장 |
|---|---|---|---|
| TE (Transformer Engine) | - | NVIDIA Tensor Core용 transformer 가속. FP4/FP8 지원 | [ch01](ch01_intro.md) |
| Token | 토큰 | LLM 처리 최소 단위. 단어보다 작거나 같음 | [concepts/tokens](concepts/tokens.md) |
| TP (Tensor Parallelism) | 텐서 병렬 | 단일 layer 내부 행렬을 GPU 분할 | [ch01](ch01_intro.md) |
| TPOT (Time Per Output Token) | - | decode 단계 토큰당 평균 생성 시간. = ITL | [concepts/tokens](concepts/tokens.md) |
| TTFT (Time To First Token) | - | prompt 입력 끝나고 첫 출력 토큰까지 latency. prefill 시간 | [concepts/tokens](concepts/tokens.md) |

## 3

| 용어 | 한국어 | 한 줄 정의 | 등장 |
|---|---|---|---|
| 3FS (Fire-Flyer File System) | - | DeepSeek의 분산 파일시스템 | [ch01](ch01_intro.md) |
