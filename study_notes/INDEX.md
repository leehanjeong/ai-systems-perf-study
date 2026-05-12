# AI Systems Performance Engineering - 학습 노트 인덱스

저자: Chris Fregly (O'Reilly, 2025) | PDF: `book.pdf` (211p, PyMuPDF 1709p)

## 진행 상황

| Ch | 제목 | PyMuPDF p. | 상태 | 노트 |
|---:|---|---:|:---:|---|
| 1 | Introduction and AI System Overview | 17-46 | done | [ch01_intro.md](ch01_intro.md) |
| 2 | AI System Hardware Overview | 48-95 | todo | |
| 3 | OS, Docker, Kubernetes Tuning | 97-162 | todo | |
| 4 | Tuning Distributed Networking Communication | 164-252 | todo | |
| 5 | GPU-Based Storage I/O Optimizations | 254-292 | todo | |
| 6 | GPU Architecture, CUDA Programming, Occupancy | 294-370 | todo | |
| 7 | Profiling and Tuning GPU Memory Access Patterns | 372-440 | todo | |
| 8 | Occupancy, Warp Efficiency, ILP | 441-516 | todo | |
| 9 | Kernel Efficiency and Arithmetic Intensity | 518-567 | todo | |
| 10 | Intra-Kernel Pipelining, Warp Specialization | 569-655 | todo | |
| 11 | Inter-Kernel Pipelining, CUDA Streams | 657-730 | todo | |
| 12 | Dynamic Scheduling, CUDA Graphs | 732-797 | todo | |
| 13 | Profiling, Tuning, Scaling PyTorch | 799-908 | todo | |
| 14 | PyTorch Compiler, Triton, XLA | 910-991 | todo | |
| 15 | Multinode Inference, Parallelism, Decoding | 993-1050 | todo | |
| 16 | Profiling, Debugging, Tuning Inference at Scale | 1051-1137 | todo | |
| 17 | Scaling Disaggregated Prefill and Decode | 1139-1196 | todo | |
| 18 | Advanced Prefill-Decode and KV Cache Tuning | 1197-1259 | todo | |
| 19 | Dynamic and Adaptive Inference Engine | 1260-1375 | todo | |
| 20 | AI-Assisted Performance Optimizations | 1377-1409 | todo | |
| A | Performance Checklist (175+ Items) | 1411-1469 | todo | |

## Concepts (`concepts/`)

책 본문에 짧게만 등장해서 별도 정리가 필요한 기초/보강 개념.

| 토픽 | 노트 |
|---|---|
| FLOPS와 device utilization (왜 misleading한가) | [concepts/flops_and_utilization.md](concepts/flops_and_utilization.md) |
| 토큰 (LLM의 처리/비용/throughput 단위) | [concepts/tokens.md](concepts/tokens.md) |
| Goodput 실제 계산법 (ETTR, MFU, inference SLO) | [concepts/goodput_calculation.md](concepts/goodput_calculation.md) |
| 단위와 메모리 계산법 (K/M/G/T, 모델 메모리, GPU 세대) | [concepts/units_and_memory.md](concepts/units_and_memory.md) |

## Cross-references

- [glossary.md](glossary.md) - 챕터 누적 용어 사전 (alpha order)
- [../examples/](../examples/) - 책 코드 실습 폴더 (Ch 6부터)
