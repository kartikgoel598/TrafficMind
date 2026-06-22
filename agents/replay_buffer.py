import numpy as np 
from collections import deque
import random 

class ReplayBuffer:
    """Experience replay buffer for storing and sampling transitions."""
    def __init__(self,capacity = 100000):
        """Initialize the replay buffer with a maximum capacity."""
        self.capacity = capacity 
        self.buffer = deque(maxlen= capacity)
    
    def push(self,state,action,reward,next_state,done):
        """Add a transition to the buffer."""
        experience = (state,action,reward,next_state,done)
        self.buffer.append(experience)
    
    def sample(self,batch_size=64):
        """Sample a batch of transitions from the buffer."""
        batch = random.sample(self.buffer,batch_size)
        states,actions,rewards,next_states,dones = zip(*batch)
        return(
            np.array(states,dtype=np.float32),
            np.array(actions,dtype=np.int64),
            np.array(rewards,dtype=np.float32),
            np.array(next_states,dtype=np.float32),
            np.array(dones,dtype=np.float32)
        )
    def __len__(self):
        """Return the current size of the buffer."""
        return len(self.buffer)
    
    def is_ready(self,batch_size = 64):
        """Check if the buffer has enough samples for training."""
        return len(self.buffer) >= batch_size