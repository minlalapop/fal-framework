import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

from fal import FAL


torch.hub.set_dir(".torch_cache")


class CIFARTransferNet(nn.Module):
    def __init__(self):
        super().__init__()
        weights = MobileNet_V3_Small_Weights.DEFAULT
        self.backbone = mobilenet_v3_small(weights=weights)

        for parameter in self.backbone.features.parameters():
            parameter.requires_grad = False

        in_features = self.backbone.classifier[-1].in_features
        self.backbone.classifier[-1] = nn.Linear(in_features, 10)

        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.features.eval()
        return self

    def forward(self, x):
        if x.ndim == 2:
            x = x.view(-1, 3, 32, 32)
        x = F.interpolate(x, size=(96, 96), mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std
        return self.backbone(x)


if __name__ == "__main__":
    model = CIFARTransferNet()

    fal = FAL(
        dataset="cifar10",
        model=model,
        model_params={
            "criterion": nn.CrossEntropyLoss(),
            "optimizer": torch.optim.AdamW(model.backbone.classifier.parameters(), lr=0.001),
            "epochs": 3,
            "batch_size": 64,
            "predict_batch_size": 128,
            "device": "cpu",
        },
        trainer_type="mixed",
        attack_type="fgsm",
        attack_params={
            "eps": 0.01,
            "batch_size": 64,
        },
        metrics=["accuracy", "f1"],
        loss="model_loss",
        num_clients=3,
        num_rounds=3,
        iid=True,
        fl_algorithm="fedadam",
        random_state=42,
    )

    result = fal.run()
    result.print_summary()
