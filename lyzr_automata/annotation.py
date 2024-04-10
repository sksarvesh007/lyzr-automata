from typing import Callable


def annotate(
    tag: str,
    description: str = None,
    input_details: str = None,
    output_details: str = None,
) -> Callable:
    """
    Enhances a function with additional metadata, including a description,
    input details, and output details. This decorator aims to improve code
    documentation and readability.

    Parameters:
    - description (str): A brief summary of what the function does.
    - input_details (str): Detailed description of the function's input parameters.
    - output_details (str): Detailed description of the function's return value.

    Returns:
    - Callable: The decorated function with added metadata.
    """

    def decorator(func: Callable) -> Callable:
        func.description = description
        func.input_details = input_details
        func.output_details = output_details
        func.name = tag
        return func

    return decorator
