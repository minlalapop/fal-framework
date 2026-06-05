import torch
from torch import nn

from fal import FAL


class SimpleMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.network(x)


model = SimpleMLP()

fal = FAL(
    dataset="mnist",
    model=model,
    model_params={
        "criterion": nn.CrossEntropyLoss(),
        "optimizer": torch.optim.Adam(model.parameters(), lr=0.001),
        "epochs": 10,
        "batch_size": 64,
        "device": "cpu",
    },
    trainer_type="mixed",
    attack_type="fgsm",
    attack_params={"eps": 0.3},
    metrics=["accuracy", "f1","precision","recall", "confusion_matrix", "classification_report"],
    loss="model_loss",
    num_clients=3,
    num_rounds=5,
    fl_algorithm="fedprox",
    fl_params={"proximal_mu": 0.1},
    random_state=42,
)
result = fal.run()
