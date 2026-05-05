# Examples

책에 등장하는 CUDA / PyTorch / Triton 코드를 직접 실행하고 확장해보는 폴더.

읽기만 했을 때보다 직접 돌렸을 때 retention이 훨씬 높다는 학습과학의 권고를 따른다. Ch 1-5는 mindset / hardware / OS 중심이라 코드가 적고, Ch 6 이후부터 본격적으로 채워진다.

## 구조 (예정)

```
examples/
  ch06_cuda_basics/
    README.md          # 책 어디 코드, 무엇을 검증
    main.cu
    Makefile
    results.md         # 측정 결과, 왜 그런지 분석
  ch07_memory_patterns/
    coalesced_vs_strided.cu
    ...
  ch13_pytorch_profiling/
    profile_eager_vs_compile.py
    ...
```

## 환경

- NVIDIA GPU (책의 코드는 H100/B200 가정. consumer GPU에서도 대부분 동작)
- CUDA Toolkit 12.x+
- PyTorch 2.x
- Python 3.10+

## 작성 규칙

각 디렉토리는 self-contained:
1. `README.md` - 책 페이지 인용, 무엇을 측정/검증하려는지
2. 코드 (가능하면 책 그대로 + 작은 확장)
3. `results.md` - 실제 돌린 결과 + 해석

코드 주석은 영어. README/results는 한국어.
