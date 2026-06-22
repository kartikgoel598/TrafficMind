def pressure_fairness_reward(
    own_queue,
    own_wait,
    neigh_queues,
    max_starvation,
    green_queue,
    red_queue,
    executed_action,
    alpha=0.5,
    beta=0.3,
    lam=0.1,
    gamma=1.5
):
    """Calculate fairness pressure reward considering neighbors and starvation."""
    neighbour_total = sum(neigh_queues)

    base_penalty = (
        own_queue
        + alpha * own_wait
        + beta * neighbour_total
        + lam * max_starvation
    )

    if executed_action == 0:
        wrong_action_pressure = max(red_queue - green_queue, 0)
    else:
        wrong_action_pressure = max(green_queue - red_queue, 0)

    reward = -(base_penalty + gamma * wrong_action_pressure)

    return reward / 100.0