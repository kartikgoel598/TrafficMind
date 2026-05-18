def fairness_reward(own_queue, neigh_queues, max_starvation,
                    beta=0.3, lam=0.1):
    
    neighbour_total = sum(neigh_queues)
    reward = -(own_queue + beta * neighbour_total + lam * max_starvation)
    return reward / 100.0