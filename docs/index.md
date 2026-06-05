# FAL Framework

FAL is a small Python framework for running Federated Adversarial Learning
experiments with Flower, scikit-learn, PyTorch, and ART.

It provides one public entrypoint:

```python
from fal import FAL
```

Core features:

- built-in datasets, CSV datasets, or in-memory `X`/`y` arrays
- IID and non-IID client splits
- scikit-learn and PyTorch model adapters
- clean, attack-only, adversarial, and mixed trainers
- Flower strategies: FedAvg, FedProx, and FedAdam
- adversarial attacks through ART: FGSM, PGD, and DeepFool

## Install

```bash
pip install fal-framework
```

## Import

```python
from fal import FAL
```
