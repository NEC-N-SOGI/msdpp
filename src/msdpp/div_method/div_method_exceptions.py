MODE = ["ext", "img", "mix"]


class InvalidDataModeError(Exception):
    """Custom exception for invalid clustering parameters."""

    def __init__(self, mode: str) -> None:
        super().__init__(f"{mode} is not a valid mode. Choose from {MODE}.")
