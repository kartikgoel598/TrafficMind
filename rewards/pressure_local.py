def pressure_local_reward(
    own_queue,
    own_wait,
    green_queue,
    red_queue,
    executed_action,
    alpha=0.5,
    beta=2.0
):
    """Calculate local pressure reward based on queue pressure."""
   
    base_penalty = own_queue + alpha * own_wait

    if executed_action == 0:
        # Agent kept current green.
        # Bad if red side has more queue pressure.
        wrong_action_pressure = max(red_queue - green_queue, 0)
    else:
        # Agent switched.
        # Bad if current green side had more queue pressure.
        wrong_action_pressure = max(green_queue - red_queue, 0)

    reward = -(base_penalty + beta * wrong_action_pressure)

    return reward / 100.0