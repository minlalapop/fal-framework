import torch
from torch import nn

from fal import FAL


class SimpleCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        if x.ndim == 2:
            x = x.view(-1, 1, 28, 28)
        return self.network(x)


if __name__ == "__main__":
    model = SimpleCNN()

    fal = FAL(
        dataset="mnist",
        model=model,
        model_params={
            "criterion": nn.CrossEntropyLoss(),
            "optimizer": torch.optim.Adam(model.parameters(), lr=0.001),
            "epochs": 5,
            "batch_size": 64,
            "device": "cpu",
        },
        trainer_type="mixed",
        attack_type="pgd",
        attack_params={
            "eps": 0.2,
            "eps_step": 0.02,
            "max_iter": 5,
            "batch_size": 64,
            "verbose": False,
        },
        metrics=["accuracy", "f1"],
        loss="model_loss",
        num_clients=3,
        num_rounds=5,
        fl_algorithm="fedavg",
        random_state=42,
    )

    result = fal.run()
    result.print_summary()
