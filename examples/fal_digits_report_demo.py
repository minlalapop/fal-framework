from sklearn.linear_model import LogisticRegression

from fal import FAL


if __name__ == "__main__":
    model = LogisticRegression(
        max_iter=200,
        solver="lbfgs",
        random_state=42,
    )

    fal = FAL(
        dataset="digits",
        model=model,
        trainer_type="mixed",
        trainer_params={
            "local_max_iter": 50,
            "suppress_convergence_warnings": True,
        },
        attack_type="fgsm",
        attack_params={"eps": 0.2},
        metrics=["accuracy", "f1", "confusion_matrix", "classification_report"],
        loss="log_loss",
        num_clients=3,
        num_rounds=3,
        fl_algorithm="fedavg",
        random_state=42,
    )

    result = fal.run()
    print("saved artifacts:", result.artifact_paths)
