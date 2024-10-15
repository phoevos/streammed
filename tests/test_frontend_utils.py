import pytest

from app.frontend.utils import (
    calculate_precision,
    convert_annotation_to_tuple,
    render_precision,
)


def test_calculate_precision():
    assert calculate_precision(10, 0) == 1.0
    assert calculate_precision(3, 7) == 0.3
    assert calculate_precision(0, 0) == 1.0
    assert calculate_precision(0, 10) == 0.0


def test_convert_annotation_to_tuple():
    annotation = {"start": 0, "end": 11, "label": "DRUG"}
    assert convert_annotation_to_tuple(annotation) == (0, 11, "DRUG")


@pytest.mark.parametrize(
    "precision, expected_message, expected_color",
    [
        (0.8, "Precision: 0.80", "success"),
        (0.6, "Precision: 0.60", "warning"),
        (0.4, "Precision: 0.40", "error"),
    ],
)
def test_render_precision(precision, expected_message, expected_color, mocker):
    mock_success = mocker.patch("streamlit.success")
    mock_warning = mocker.patch("streamlit.warning")
    mock_error = mocker.patch("streamlit.error")
    render_precision(precision)

    if expected_color == "success":
        mock_success.assert_called_once_with(expected_message)
    elif expected_color == "warning":
        mock_warning.assert_called_once_with(expected_message)
    elif expected_color == "error":
        mock_error.assert_called_once_with(expected_message)
