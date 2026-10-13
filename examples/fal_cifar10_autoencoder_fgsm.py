import torch
from torch import nn

from fal import FAL


class CifarVisionTransformer(nn.Module):
    def __init__(
        self,
        patch_size: int = 4,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 3,
        num_classes: int = 10,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (32 // patch_size) ** 2
        self.patch_dim = 3 * patch_size * patch_size

        self.patch_embedding = nn.Linear(self.patch_dim, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.position_embedding = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=256,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, x):
        if x.ndim == 2:
            x = x.view(-1, 3, 32, 32)

        batch_size = x.shape[0]
        patches = x.unfold(2, self.patch_size, self.patch_size)
        patches = patches.unfold(3, self.patch_size, self.patch_size)
        patches = patches.permute(0, 2, 3, 1, 4, 5)
        patches = patches.contiguous().view(batch_size, self.num_patches, self.patch_dim)

        tokens = self.patch_embedding(patches)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls_tokens, tokens], dim=1)
        tokens = tokens + self.position_embedding

        encoded = self.encoder(tokens)
        return self.classifier(encoded[:, 0])


if __name__ == "__main__":
    torch.manual_seed(42)
    model = CifarVisionTransformer()

    fal = FAL(
        dataset="cifar10",
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
            "eta": 0.01,
            "eta_l": 1.0,
        },
        random_state=42,
    )

    result = fal.run()
    result.print_summary()
    print("saved artifacts:", result.artifact_paths)
