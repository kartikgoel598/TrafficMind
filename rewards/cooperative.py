def cooperative_reward(own_queue, neigh_queues, beta=0.3):
    neighbour_total = sum(neigh_queues)
    reward = -(own_queue + beta * neighbour_total)
    return reward / 100.0