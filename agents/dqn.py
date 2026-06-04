import torch
import random
import numpy as np 
import torch.nn as nn 
import torch.optim as optim

class QNetwork(nn.Module):
    def __init__(self,state_size,action_size):
        super(QNetwork,self).__init__()
        self.network  = nn.Sequential(
            nn.Linear(state_size,128),
            nn.ReLU(),
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128,action_size)
        )
    def forward(self,x):
        return self.network(x)

class DQNAgent:
    def __init__(self, state_size, action_size,
                 lr=0.001, gamma=0.99,
                 epsilon=1.0, epsilon_min=0.01,
                 epsilon_decay=0.999991,
                 target_update_freq=500):
        
        self.state_size  = state_size
        self.action_size = action_size
        self.gamma       = gamma

        
        self.epsilon       = epsilon
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq
        self.steps_done = 0
        # just to make sure its using GPU
        print(f"Building DQN Agent... using {'GPU' if torch.cuda.is_available() else 'CPU'} (this should print 4 times)")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.main_network = QNetwork(state_size, action_size).to(self.device)

        self.target_network = QNetwork(state_size, action_size).to(self.device)

        self.target_network.load_state_dict(self.main_network.state_dict())
        self.target_network.eval()
        self.optimizer = optim.Adam(self.main_network.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
    
    # returns an int 
    def select_action(self,state):
        phase_time = state[9]   # normalised, < 1.0 means switching blocked
        is_yellow = state[10]   # 1.0 if yellow, 0.0 if not

        if is_yellow == 1.0 or phase_time < 1.0:
            return 0  # force keep, switching is illegal

        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        # expected future return if I take action a in state s
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.main_network(state_tensor)
        return q_values.argmax().item()
    
    def train_step(self,replay_buffer, batch_size=64):
        if not replay_buffer.is_ready(batch_size):
            return None
        states,actions,rewards,next_states,dones = replay_buffer.sample(batch_size)
        states      = torch.FloatTensor(states).to(self.device)
        actions     = torch.LongTensor(actions).to(self.device)
        rewards     = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones       = torch.FloatTensor(dones).to(self.device)
        current_q = self.main_network(states).gather(
            1, actions.unsqueeze(1)
        ).squeeze(1)

        with torch.no_grad():
            next_q_all = self.target_network(next_states).clone()  # shape [batch, 2]
            # mask illegal actions, phase_time is index 9, is_yellow is index 10
            phase_time = next_states[:, 9]   # normalised, <1.0 means blocked
            is_yellow  = next_states[:, 10]  # 1.0 if yellow
            illegal    = (phase_time < 1.0) | (is_yellow == 1.0)
            next_q_all[:, 1][illegal] = -float('inf')  # mask switch action
            next_q     = next_q_all.max(1)[0]

        target_q = rewards + self.gamma * next_q * (1-dones)
        loss = self.criterion(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.main_network.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.steps_done += 1
        if self.steps_done % self.target_update_freq == 0:
            self._update_target_network()
        return loss.item()
    
    def _update_target_network(self):
         self.target_network.load_state_dict(
            self.main_network.state_dict()
        )
    def save(self, path):
        
        torch.save(self.main_network.state_dict(), path)

    def load(self, path):
       
        self.main_network.load_state_dict(
            torch.load(path, map_location=self.device)
        )


