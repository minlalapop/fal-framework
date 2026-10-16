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

For best-effort reproducibility across Python, NumPy, PyTorch, and federated
clients, provide a base seed and enable deterministic PyTorch operations:

```python
fal = FAL(
    dataset="wine",
    model=LogisticRegression(max_iter=1000, random_state=42),
    random_state=42,
    deterministic=True,
)
```

Client seeds are derived from the base seed, client identifier, federated
round, and execution phase. Exact bitwise reproducibility can still depend on
the hardware and third-party operations. Because a model object is constructed
before it is passed to `FAL`, seed its initialization before constructing it:

```python
from fal import seed_everything

seed_everything(42, deterministic=True)
model = MyPyTorchModel()
```

After a run, timing data are available separately from model metrics and are
also saved to `timings.json`:

```python
result = fal.run()
print(result.timings["total_execution_time"])
print(result.timings["rounds"])
print(result.timings["clients"])
```

If no evaluation loss is configured, `result.losses` is empty and
`result.loss_available` is `False`; no artificial zero is written to the
metrics CSV.

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
