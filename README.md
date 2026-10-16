# [NeurIPS 2026] OliO: ODE-based Linear Transition Operator for Self-Supervised Time Series Forecasting
<a href="https://github.com/eai-lab/AliO"><img src="https://img.shields.io/badge/Python-3776AB?style=flat&logo=Python&logoColor=white"/></a>
<a href="https://github.com/eai-lab/AliO"><img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=PyTorch&logoColor=white"/></a>
<a href="https://neurips.cc/virtual/2026/loc/sydney/poster/150464"><img src="https://img.shields.io/badge/NeurIPS25:150464-3776AB?style=flat"/></a>


Poster <a href="https://neurips.cc/media/PosterPDFs/NeurIPS%202026/150464.png?t=1791857063.7298853">link</a>

# Abstract
Self-supervised representation learning has  advanced time series forecasting by capturing robust features from unlabeled data. However, existing contrastive and masking-based methods often struggle to explicitly model the underlying temporal flow, either by treating time lag as noise or by disrupting the inherent temporal dynamics essential for forecasting. To address these limitations, we propose OliO (**O**DE-based **Li**near Transition **O**perator), a plug-and-play self-supervised learning method that learns bidirectional temporal flows as a continuous and structurally consistent dynamical system. OliO introduces a transition operator derived from linear Ordinary Differential Equations (ODEs) to align latent representations across arbitrary time shifts. By imposing a strictly upper triangular constraint on the transition matrix, we mathematically ensure numerical stability and enforce a robust polynomial boundary that effectively captures long-term dependencies while preventing the exponential divergence typically found in unconstrained ODE-based models. This structural constraint induces a temporally coherent representation space that preserves the underlying flow essential for accurate forecasting. Extensive experiments across various backbone architectures and benchmarks demonstrate that OliO achieves significant performance gains, providing Mean Squared Error (MSE) reductions of up to 5.75% over the most competitive grid-searched SOTA baselines. The code is available at *https://github.com/eai-lab/OliO*

### How to use
```bash
pip install -r requirements.txt
```

#### Pretrain
```bash
bash scripts/<pretrain>/<model>/<dataset>.sh
```

#### Finetune
```bash
bash scripts/<finetune>/<model>/<dataset>.sh
```
