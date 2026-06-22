def clamp(value, low, high):
    """Clamp a value between low and high bounds."""
    return max(low, min(value, high))


def compute_webster_timing(
    phase0_flow,
    phase1_flow,
    saturation_flow=1800.0,
    yellow_time=3.0,
    min_green=15.0,
    min_cycle=36.0,
    max_cycle=120.0,
):
    """Compute a simplified two-phase static Webster signal plan."""

    lost_time = 2.0 * yellow_time

    y0 = phase0_flow / saturation_flow
    y1 = phase1_flow / saturation_flow
    Y = y0 + y1

    if Y >= 0.95:
        Y = 0.95

    if Y <= 0:
        cycle = min_cycle
        green0 = (cycle - lost_time) / 2.0
        green1 = (cycle - lost_time) / 2.0
    else:
        cycle = (1.5 * lost_time + 5.0) / (1.0 - Y)
        cycle = clamp(cycle, min_cycle, max_cycle)

        effective_green = cycle - lost_time
        green0 = (y0 / Y) * effective_green
        green1 = (y1 / Y) * effective_green

    green0 = max(green0, min_green)
    green1 = max(green1, min_green)

    green0 = int(round(green0))
    green1 = int(round(green1))
    yellow = int(round(yellow_time))
    cycle = green0 + green1 + 2 * yellow

    return {
        "green0": green0,
        "green1": green1,
        "yellow": yellow,
        "cycle": cycle,
        "phase0_flow": float(phase0_flow),
        "phase1_flow": float(phase1_flow),
        "Y": float(Y),
    }