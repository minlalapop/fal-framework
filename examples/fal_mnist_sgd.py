from sklearn.linear_model import LogisticRegression

from fal import FAL


model = LogisticRegression(
    max_iter=5,
    random_state=42,
)

fal = FAL(
    dataset="mnist",
    model=model,
    trainer_type="mixed",
    trainer_params={
        "local_max_iter": 5,
        "suppress_convergence_warnings": True,
    },
    attack_type="fgsm",
    attack_params={"eps": 0.05},
    metrics=["accuracy", "f1"],
    loss="log_loss",
    num_clients=3,
    num_rounds=3,
    iid=True,
    fl_algorithm="fedavg",
    random_state=42,
)

result = fal.run()
result.print_summary()
