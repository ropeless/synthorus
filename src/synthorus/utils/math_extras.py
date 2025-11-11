import math


def p_log_p(p) -> float:
    return 0 if p == 0 else p * math.log2(p)
