def local_reward(own_queue, own_wait, alpha=0.5):
    """Calculate local reward based on queue and wait time."""
    reward = -(own_queue + alpha * own_wait)
    return reward / 100.0