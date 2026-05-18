def fairness_reward(own_queue, own_wait, neigh_queues, max_starvation, alpha=0.5, beta=0.3, lam=0.1):
    neighbour_total = sum(neigh_queues)
    reward = -(own_queue + alpha * own_wait + beta * neighbour_total + lam * max_starvation)
    return reward / 100.0