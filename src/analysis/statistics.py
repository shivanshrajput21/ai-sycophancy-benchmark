import math


def wilson_interval(successes: int, trials: int, z: float = 1.96) -> tuple[float, float] | None:
    if trials == 0:
        return None
    p = successes / trials
    denom = 1 + z*z/trials
    center = (p + z*z/(2*trials))/denom
    half = z*math.sqrt(p*(1-p)/trials + z*z/(4*trials*trials))/denom
    return max(0.0, center-half), min(1.0, center+half)
