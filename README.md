# SenescenceEngine

Pure Python cellular senescence analysis pipeline.

## Features
- ssGSEA-style senescence scoring (SenMayo gene set)
- Telomere length estimation (TL-score)
- SASP factor expression analysis
- Cell cycle arrest detection (p21/p16 axis)
- Senescence pseudotime trajectory
- Differential expression (BH FDR)

## Usage
```bash
pip install numpy scipy matplotlib
python senescence_engine.py
```

## Results (800 cells, 2000 genes)
- 42 DE genes (FDR<0.05, |log2FC|>1)
- Top SASP factor: CCL5 (log2FC=3.67)
- p21-arrested: 53 cells (6.6%)
- Senescence-SASP correlation: r=0.386
