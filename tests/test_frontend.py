from unittest.mock import MagicMock

import pytest
import requests
from streamlit.testing.v1 import AppTest

from app.frontend.main import get_annotations


@pytest.fixture
def mock_requests_post(mocker):
    return mocker.patch("app.frontend.main.requests.post")


@pytest.fixture
def mock_st_error(mocker):
    return mocker.patch("app.frontend.main.st.error")


def test_get_annotations_success(mock_requests_post):
    mock_response = MagicMock()
    mock_response.json.return_value = {"annotations": [{"start": 0, "end": 11, "label": "DRUG"}]}
    mock_response.raise_for_status = MagicMock()
    mock_requests_post.return_value = mock_response

    annotations = get_annotations("paracetamol is a drug commonly used for pain relief", "med7")
    assert annotations == [{"start": 0, "end": 11, "label": "DRUG"}]


def test_get_annotations_request_exception(mock_requests_post, mock_st_error):
    mock_requests_post.side_effect = requests.exceptions.RequestException("Request failed")
    _ = get_annotations("dummy text", "random")
    mock_st_error.assert_called_once_with(
        "Failed to retrieve annotations from application backend."
    )


def test_get_annotations_value_error(mock_requests_post, mock_st_error):
    mock_response = MagicMock()
    mock_response.json.side_effect = ValueError("Invalid JSON")
    mock_response.raise_for_status = MagicMock()
    mock_requests_post.return_value = mock_response

    _ = get_annotations("dummy text", "random")
    mock_st_error.assert_called_once_with("Received an invalid response from application backend.")


def test_streamlit_app(mocker):
    mocker.patch(
        "app.frontend.main.requests.post",
        return_value=MagicMock(
            json=lambda: {"annotations": [{"start": 0, "end": 11, "label": "DRUG"}]}
        ),
    )
    mocker.patch(
        "app.frontend.main.st_ner_annotate",
        return_value=[{"start": 0, "end": 11, "label": "DRUG"}],
    )
    text = "paracetamol is a drug commonly used for pain relief"

    at = AppTest.from_file("app/frontend/main.py").run()
    at.text_area(key="text").set_value(text).run()
    at.selectbox(key="model").set_value("med7").run()

    assert at.session_state["annotations"] == [{"start": 0, "end": 11, "label": "DRUG"}]
    assert at.session_state["original_annotations"] == {(0, 11, "DRUG")}
    assert at.session_state["total_annotations"] == 1
    assert at.session_state["prev_text"] == text
    assert at.session_state["prev_model"] == "med7"

    assert len(at.success) == 1
    assert at.success[0].value == "Precision: 1.00"

    # simulate the removal of an annotation through the UI
    at.session_state["annotations"].pop()
    at.run()

    assert len(at.error) == 1
    assert at.error[0].value == "Precision: 0.00"

    # verify that annotations added by the user do not affect the precision calculation
    at.session_state["annotations"].append({"start": 40, "end": 44, "label": "DRUG"})
    at.run()

    assert len(at.error) == 1
    assert at.error[0].value == "Precision: 0.00"

    # verify that re-adding the original annotation restores the precision calculation
    at.session_state["annotations"].append({"start": 0, "end": 11, "label": "DRUG"})
    at.run()

    assert len(at.success) == 1
    assert at.success[0].value == "Precision: 1.00"
