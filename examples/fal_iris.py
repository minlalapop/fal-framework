from sklearn.linear_model import LogisticRegression

from fal import FAL


fal = FAL(
    dataset="iris",
    model=LogisticRegression(max_iter=1000),
    trainer_type="clean",
    metrics=["accuracy", "f1"],
    num_clients=3,
    num_rounds=3,
    fl_algorithm="fedavg",
)

result = fal.run()
