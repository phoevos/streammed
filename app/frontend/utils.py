import streamlit as st


def calculate_precision(true_positives: int, false_positives: int) -> float:
    """Calculate the precision value based on the number of true and false positives."""
    return (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0
        else 1.0
    )


def render_precision(precision: float) -> None:
    """Render a message based on the precision value.

    If the precision is greater than or equal to 0.7, the message will be displayed in green.
    If the precision is greater than or equal to 0.5, the message will be displayed in yellow.
    Otherwise, the message will be displayed in red.
    """
    message = f"Precision: {precision:.2f}"
    if precision >= 0.7:
        st.success(message)
    elif precision >= 0.5:
        st.warning(message)
    else:
        st.error(message)


def convert_annotation_to_tuple(annotation: dict) -> tuple:
    """Convert an annotation dictionary to a tuple."""
    return annotation["start"], annotation["end"], annotation["label"]
