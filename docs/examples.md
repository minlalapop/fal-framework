# Examples

The repository includes runnable examples in `examples/`.

## Wine

```bash
python examples/fal_wine.py
```

## Iris

```bash
python examples/fal_iris.py
```

## MNIST

```bash
python examples/fal_mnist_demo.py
```

Flower simulations require Ray. If Flower reports that Ray is missing, install:

```bash
python -m pip install "flwr[simulation]"
```

Use Python 3.11 or 3.12 if Ray is not available for your current Python version.
