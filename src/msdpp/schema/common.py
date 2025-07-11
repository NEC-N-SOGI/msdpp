def raise_error(flag: bool, msg: str) -> None:
    if flag:
        raise ValueError(msg)
