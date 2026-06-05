# Getting Started

Minimal example with a built-in dataset and a scikit-learn model:

```python
from sklearn.linear_model import LogisticRegression
from fal import FAL

fal = FAL(
    dataset="wine",
    model=LogisticRegression(max_iter=1000, random_state=42),
    trainer_type="clean",
    metrics=["accuracy", "f1"],
    loss="log_loss",
    num_clients=3,
    num_rounds=3,
    iid=True,
    fl_algorithm="fedavg",
    random_state=42,
)

result = fal.run()
print(result.latest_metrics)
```

Using arrays directly:

```python
fal = FAL(
    X=X,
    y=y,
    model=model,
    trainer_type="clean",
    num_clients=3,
    num_rounds=3,
    fl_algorithm="fedavg",
)
```

For PyTorch models, pass the training objects in `model_params`:

```python
model_params={
    "criterion": nn.CrossEntropyLoss(),
    "optimizer": torch.optim.Adam(model.parameters(), lr=0.001),
    "epochs": 1,
    "batch_size": 128,
    "device": "cpu",
}
```
