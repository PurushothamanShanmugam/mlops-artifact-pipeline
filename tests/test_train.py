import json
from sklearn.linear_model import LogisticRegression
from src.train import load_config, train_model
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score

def test_config_loading():
    config = load_config("config/config.json")
    assert isinstance(config["C"], (float, int))
    assert isinstance(config["solver"], str)
    assert isinstance(config["max_iter"], int)

def test_model_training_type():
    digits = load_digits()
    X, y = digits.data, digits.target
    config = load_config("config/config.json")
    model = train_model(X, y, config)
    assert isinstance(model, LogisticRegression)

def test_model_accuracy():
    digits = load_digits()
    X, y = digits.data, digits.target
    config = load_config("config/config.json")
    model = train_model(X, y, config)
    preds = model.predict(X)
    acc = accuracy_score(y, preds)
    assert acc > 0.8

