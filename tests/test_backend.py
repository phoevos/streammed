from fastapi.testclient import TestClient

from app.backend.main import app

client = TestClient(app)


def assert_valid_response(response):
    assert response.status_code == 200
    data = response.json()
    assert "annotations" in data
    assert isinstance(data["annotations"], list)
    for annotation in data["annotations"]:
        assert "start" in annotation
        assert "end" in annotation
        assert "label" in annotation
        assert annotation["label"] == "DRUG"


def test_predict_random_model():
    response = client.post(
        "/predict",
        json={
            "text": "paracetamol and ibuprofen are drugs commonly used for pain relief",
            "model": "random",
        },
    )
    assert_valid_response(response)


def test_predict_med7_model():
    response = client.post(
        "/predict",
        json={
            "text": "paracetamol and ibuprofen are drugs commonly used for pain relief",
            "model": "med7",
        },
    )
    assert_valid_response(response)
    expected_annotations = [
        {"start": 0, "end": 11, "label": "DRUG"},
        {"start": 16, "end": 25, "label": "DRUG"},
    ]
    assert response.json()["annotations"] == expected_annotations


def test_invalid_model():
    response = client.post(
        "/predict",
        json={
            "text": "paracetamol and ibuprofen are drugs commonly used for pain relief",
            "model": "invalid_model",
        },
    )
    assert response.status_code == 400
    data = response.json()
    assert data["detail"] == "Invalid model name. Supported models are 'random' and 'med7'."


def test_empty_text():
    for model in ["random", "med7"]:
        response = client.post("/predict", json={"text": "", "model": model})
        assert response.status_code == 200
        data = response.json()
        assert "annotations" in data
        assert isinstance(data["annotations"], list)
        assert len(data["annotations"]) == 0


def test_unexpected_error_handling(mocker):
    mocker.patch("app.backend.main.nltk.word_tokenize", side_effect=Exception("Tokenization error"))
    response = client.post(
        "/predict",
        json={
            "text": "paracetamol and ibuprofen are drugs commonly used for pain relief",
            "model": "random",
        },
    )
    assert response.status_code == 500
    data = response.json()
    assert data["detail"] == "An unexpected error occurred."
