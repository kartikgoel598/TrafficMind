def cooperative_reward(own_queue, own_wait, neigh_queues,
                       alpha=0.5, beta=0.3):
    neighbour_total = sum(neigh_queues)
    reward = -(own_queue + alpha * own_wait + beta * neighbour_total)
    return reward / 100.0