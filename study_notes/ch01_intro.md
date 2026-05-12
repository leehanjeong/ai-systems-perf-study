# Ch 1. Introduction and AI System Overview

> AI systems performance engineer 라는 직무가 무엇이고, 왜 ultrascale 시대(100T 파라미터)에 그 가치가 폭증하는가. 핵심 사고방식은 mechanical sympathy(HW-SW codesign)와 goodput(유용한 처리량) 측정.

PyMuPDF p.17-46 | Pages 1-31

## 챕터 흐름

- 1.1 The AI Systems Performance Engineer
  - Benchmarking & profiling
  - Scaling distributed training/inference
  - Managing resources efficiently
  - Cross-team collaboration
  - Transparency & reproducibility
- 1.2 DeepSeek case study (H800 제약 하에서 ~680B MoE 학습 성공)
- 1.3 Toward 100T parameter models
- 1.4 NVIDIA "AI Supercomputer in a Rack" (GB200/GB300 NVL72)
- 1.5 Mechanical sympathy (FlashAttention, MLA)
- 1.6 Measuring "Goodput" Useful Throughput
- 1.7 Book roadmap & methodology
- 1.8 Key takeaways

## 핵심 개념

### Goodput (유용한 처리량) - **이 챕터에서 가장 중요한 개념**

- **정의**: 단위 시간당 실제로 모델 학습/추론에 기여한 작업량 (토큰 처리, 추론 요청 완료). 통신 대기, 데이터 로딩 stall, 실패 후 재시작에 쓰인 시간은 빼고 측정.
- **왜 중요한가**: FLOPS/device utilization은 misleadingly high할 수 있다. 클러스터가 100% utilized로 보여도 실제 useful work는 25-30%일 수 있음 (Meta 2025 paper "Revisiting Reliability"의 관찰).
- **공식**:
  - Goodput = useful tokens or requests per second
  - Efficiency % = goodput / theoretical peak throughput
- **예시**: 8-GPU 노드가 100K 토큰을 10s에 처리하면 goodput=10K tok/s. 이론 peak가 12K tok/s면 efficiency 83.3%.
- **관련 용어**: NVIDIA가 말하는 "speed of light"는 이론 hardware 최대치. Goodput을 speed of light 대비 %로 표현하면 시스템 건강성 단일 지표 확보.

### Mechanical Sympathy (HW-SW codesign)

- **정의**: 소프트웨어를 hardware의 동작 방식에 맞춰 설계하는 사고방식. Martin Thompson이 F1 드라이버 Jackie Stewart에서 따와 처음 명명.
- **왜 중요한가**: 미미해 보이는 GPU 커널/메모리 접근 패턴 변경이 거대한 성능 변화를 만든다. FlashAttention은 attention 연산을 tiling으로 재구성해 메모리 read/write를 줄여 2x-4x 속도 + 메모리 사용량 동시 감소.
- **확장 사례**: DeepSeek MLA(Multi-Head Latent Attention)는 H800의 NVLink bandwidth 한계 환경에서 FlashAttention조차 능가.

### Profile-driven mindset / Anti-anecdotal

- **정의**: "X 했더니 빨라진 것 같아"식 vibe optimization을 거부. 가설 -> reproducible benchmark -> 결과 -> 조정 -> 재측정 -> 모든 단계 공개.
- **왜 중요한가**: 추측 기반 최적화는 cluster scale에서 quadratic하게 비용을 부풀린다.

### Scaling 종류 (구체 종류는 Ch 13-14에서 깊이 다룸)

- DP (Data Parallelism), FSDP (Fully Sharded DP), TP (Tensor Parallel), PP (Pipeline Parallel), CP (Context Parallel), EP (Expert Parallel for MoE)
- 모델이 단일 GPU에 안 들어가면 TP/PP, MoE면 EP, 메모리 절감하려면 FSDP. Hybrid가 일반.

### Cross-Team Collaboration이 직무의 절반

- 성능 개선은 종종 driver 업그레이드, 모델 코드 변경, infra 변경을 동시에 요구. 따라서 researcher/data scientist/DevOps/infra/network/storage 팀과 동시에 일해야 함. AI systems perf engineer는 multidisciplinary translator.

## 새 용어

| 영문 | 한국어 | 설명 |
|---|---|---|
| Goodput / ETTR | 유용한 처리량 | useful work per time. Meta는 Effective Training-Time Ratio라고 부름 |
| Speed of light | 이론적 최대치 | NVIDIA 용어. 하드웨어가 낼 수 있는 peak |
| Mechanical sympathy | (직역 어색) | HW를 깊이 이해한 SW 설계 |
| FlashAttention | 그대로 | tiling 기반 attention, 메모리 read/write 최소화 |
| MLA (Multi-Head Latent Attention) | 그대로 | DeepSeek attention. H800에서 FlashAttention보다 빠름 |
| MoE (Mixture of Experts) | 전문가 혼합 모델 | 토큰별로 일부 expert만 활성화. ~680B 중 ~37B만 active |
| DualPipe | 그대로 | bidirectional pipeline parallelism. fwd/bwd 계산을 통신과 overlap |
| EPLB (Expert Parallelism Load Balancer) | 그대로 | hot expert를 복제해 부하 균형 |
| 3FS (Fire-Flyer File System) | 그대로 | DeepSeek의 분산 파일시스템 |
| NIXL (NVIDIA Inference Xfer Library) | 그대로 | 추론용 point-to-point 데이터 이동 (KV cache 등) |
| NCCL | 그대로 ("nickel") | collective 통신 (all-reduce, all-gather 등) |
| Transformer Engine (TE) | 그대로 | NVIDIA Tensor Core용 transformer 전용 가속 (FP4/FP8 지원) |
| MIG (Multi-Instance GPU) | 그대로 | GPU를 분할해 여러 워크로드에 나눠 줌 |
| HBM (High Bandwidth Memory) | 그대로 | GPU 칩 옆 stacked DRAM. B200=192GB, B300=288GB |
| NVL72 | 그대로 | rack 안 72-GPU NVLink 도메인 (36 Grace + 72 Blackwell) |
| Hopper | NVIDIA GPU 세대 | 2022~. H100 / H200 / H800. FP8 Tensor Core 도입 |
| Blackwell | NVIDIA GPU 세대 | 2024~. B200 / B300 / GB200. FP4 Tensor Core, NVLink5 |
| Grace | NVIDIA ARM CPU | NVL72의 CPU. Blackwell GPU와 superchip 구성 |
| Reduced precision | 줄어든 정밀도 | FP32 대신 BF16/FP8/FP4 등 더 적은 비트. 메모리/속도/대역폭 ↑, 정확도 ↓ |

## 핵심 숫자/벤치마크

| 수치 | 값 | 출처 |
|---|---|---|
| GPT-4 학습 비용 | ~$100M (2023) | 추정치 |
| Gemini Ultra 학습 비용 | ~$191M (late 2023) | 추정치 |
| DeepSeek-R1 학습 비용 | < $6M (논란 있음) | DeepSeek 발표 |
| H100 NVLink BW per GPU | 900 GB/s | NVIDIA |
| H800 NVLink BW per GPU | 400 GB/s | export-compliant 변종 |
| H100 HBM BW | 3.35 TB/s | NVIDIA |
| Blackwell B200 HBM | 192 GB (180 usable) | NVIDIA |
| Blackwell B300 (Ultra) HBM | ~288 GB | NVIDIA |
| NVLink 5 BW per Blackwell GPU | 1.8 TB/s bidirectional (18x100) | NVIDIA |
| NVL72 aggregate BW | ~130 TB/s GPU-to-GPU | NVIDIA |
| NVL72 FP4 peak | ~1.44 exaFLOPS (2:1 sparsity) | NVIDIA |
| NVL72 FP8 peak | ~720 petaFLOPS (2:1 sparsity) | NVIDIA |
| NVL72 HBM total | 13.5 TB (192 x 72) | 계산 |
| NVL72 + Grace memory | ~30 TB | NVIDIA |
| NVL72 rack power | 120-132 kW | 벤더/냉각 의존 |
| MLPerf v5.0 Train | GB200 NVL72 vs Hopper: **2.6x per GPU** | MLPerf |
| MLPerf v5.0 Inference | GB200 NVL72 vs Hopper: **3.4x per GPU** | MLPerf |
| 100T model 메모리 (16-bit) | ~182 TB | 계산 |
| 100T model 로드 GPU 수 | ~1000x B200 또는 ~700x B300 (~125 / 86 노드) | 계산 |
| DeepSeek-V3 active per token | 1 shared + 8 of 256 = 9 experts (~37B params) | DeepSeek-V3 report |
| 100T training compute | ~1.2e29 FLOPs (29T tokens, dense 가정) | 추정 |
| Switch Transformer (Google 2021) | 1.6T MoE, dense 대비 7x 학습 가속 | Google |
| Meta "Revisiting Reliability" | nominal util 100%여도 실제 goodput 25-30% | Meta 2025 |

## 응용/적용 아이디어

이 챕터는 mindset 챕터라 구체 기법보다는 사고 프레임이 핵심.

- **Goodput 사고방식**: GPU 클러스터/학습 작업/추론 시스템 어디든 "raw util %"가 아니라 "useful work %"로 보고 지표를 재정의해보면 새로운 비효율이 보인다.
- **Profile-driven**: 추측 대신 측정. Nsight Systems/Compute, PyTorch profiler, DCGM PROF_* 시리즈가 표준 도구.
- **Communication vs compute bound**: 통신/IO가 GPU를 굶기는지를 먼저 진단. 그 다음에 compute 효율 최적화. 순서가 중요.
- **System-level metric**: 단일 컴포넌트 점수보다 end-to-end 점수가 신뢰할 만하다는 MLPerf 권고. 모니터링/평가 지표 설계 시 적용.
- **Cross-team translator**: AI systems perf engineer는 researcher / infra / network / storage 팀의 번역자. 한 팀에 갇히지 말 것.

## Appendix 체크리스트 매핑

부록 175+ 항목 중 본 챕터의 사고방식과 직접 연결되는 섹션:

- "Performance Tuning and Cost Optimization Mindset" (p.1411): goodput 사고방식
- "Reproducibility and Documentation Best Practices" (p.1415): profile-driven, transparency
- "Performance Profiling, Debugging, and Monitoring" (p.1438): Nsight + PyTorch profiler

본 챕터 자체는 mindset 챕터라서 구체적 체크리스트 매핑은 적음. 하지만 부록 첫 두 섹션은 사실상 1장의 압축.

## Open Questions / 추가 확인

> 책이 명확히 답하지 않았거나, 추정/근사라서 외부 자료로 보충 확인이 필요한 항목. 다음 챕터 또는 후속 학습에서 풀어보기.

- DeepSeek $6M 학습비 주장의 검증 상태 (책도 "doubt as to validity"라고 적음). 정확한 inclusion/exclusion 범위 미상.
- NVL72의 130 TB/s aggregate 수치가 어떤 측정 방식인지 (peer-to-peer all-reduce인지, all-pairs sum인지) 책에서 모호. Ch 2에서 명확화 기대.
- "100T params trained on 29T tokens => 1.2e29 FLOPs" 추정의 token 수 출처. Chinchilla scaling law 확장인지 확인 필요.

## 다음 챕터 예고

Ch 2: Grace CPU + Blackwell GPU "superchip" 구조, Tensor Core, NVLink/NVSwitch, SHARP, 냉각, 전력. 모든 후속 chapter의 hardware baseline.
