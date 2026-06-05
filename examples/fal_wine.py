from sklearn.linear_model import LogisticRegression

from fal import FAL


fal = FAL(
    dataset="mnist",
    model=LogisticRegression(max_iter=5, random_state=42),
    trainer_type="adversarial",
    trainer_params={
        "local_max_iter": 5,
        "suppress_convergence_warnings": True,
    },
    metrics=["accuracy", "f1"],
    attack_type="fgsm",
    attack_params={"eps": 0.1},
    loss="log_loss",
    num_clients=3,
    num_rounds=3,
    fl_algorithm="fedavg",
)

result = fal.run()
result.print_summary()


