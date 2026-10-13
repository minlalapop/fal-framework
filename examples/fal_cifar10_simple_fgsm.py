import torch
from torch import nn

from fal import FAL


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.activation(x + self.block(x))


class CIFARResNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("mean", torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1))

        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.features = nn.Sequential(
            ResidualBlock(64),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            ResidualBlock(128),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            ResidualBlock(256),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        if x.ndim == 2:
            x = x.view(-1, 3, 32, 32)
        x = (x - self.mean) / self.std
        x = self.stem(x)
        x = self.features(x)
        return self.classifier(x)


if __name__ == "__main__":
    torch.manual_seed(42)
    model = CIFARResNet()

    fal = FAL(
        dataset="cifar10",
        dataset_params={
            "samples_per_class": 200,
            "random_state": 42,
        },
        model=model,
        model_params={
            "criterion": nn.CrossEntropyLoss(),
            "optimizer": torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01),
            "epochs": 2,
            "batch_size": 128,
            "predict_batch_size": 256,
            "device": "cpu",
        },
        trainer_type="mixed",
        attack_type="fgsm",
        attack_params={
            "eps": 0.03,
            "batch_size": 128,
        },
        metrics=["accuracy", "f1"],
        loss="model_loss",
        num_clients=3,
        num_rounds=3,
        fl_algorithm="fedadam",
        fl_params={
            "eta": 0.05,
            "eta_l": 1.0,
        },
        random_state=42,
    )

    result = fal.run()
    result.print_summary()
    print("saved artifacts:", result.artifact_paths)
