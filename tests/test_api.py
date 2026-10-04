def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_valid_prediction_returns_species(client):
    payload = {"sepal_length": 5.1, "sepal_width": 3.5,
               "petal_length": 1.4, "petal_width": 0.2}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "predicted_species" in response.json()


def test_setosa_prediction_is_correct(client):
    payload = {"sepal_length": 5.1, "sepal_width": 3.5,
               "petal_length": 1.4, "petal_width": 0.2}
    assert client.post("/predict", json=payload).json()["predicted_species"] == "setosa"


def test_virginica_prediction_is_correct(client):
    payload = {"sepal_length": 6.7, "sepal_width": 3.0,
               "petal_length": 5.8, "petal_width": 2.2}
    assert client.post("/predict", json=payload).json()["predicted_species"] == "virginica"


def test_missing_field_is_rejected(client):
    response = client.post("/predict", json={"sepal_length": 5.1})
    assert response.status_code == 422


def test_wrong_data_type_is_rejected(client):
    payload = {"sepal_length": "abc", "sepal_width": 3.5,
               "petal_length": 1.4, "petal_width": 0.2}
    assert client.post("/predict", json=payload).status_code == 422


def test_negative_value_is_rejected(client):
    payload = {"sepal_length": -1, "sepal_width": 3.5,
               "petal_length": 1.4, "petal_width": 0.2}
    assert client.post("/predict", json=payload).status_code == 422
