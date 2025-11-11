from datetime import datetime


def timestamp() -> str:
    """
    A string time stamp.
    """
    return datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S (%z)')
