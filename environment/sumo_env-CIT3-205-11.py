import os
import traci
import numpy as np
import sys

class SumoEnvironment:
    def __init__(self,config_path,reward_fn='local',use_gui=False):
        self.config_path = config_path
        self.reward_fn = reward_fn
        self.use_gui = use_gui
        self.intersections = ['J1','J2','J4','J5']
        self.num_phases = {
            "J1":2,
            "J2":2,
            "J4":2,
            "J5":2
        }
        self.lanes = {
            "J1":['E0_0','-E1_0','-E3_0','E8_0'],
            'J2':['E1_0','-E2_0','E5_0','E10_0'],
            'J4':['E7_0','-E4_0','E3_0','-E9_0'],
            'J5': ['E4_0',  '-E6_0', '-E5_0', '-E11_0'],
        }
        self.neighbours = {
            'J1':['J2','J4'],
            "J2":['J1','J5'],
            'J4': ['J1', 'J5'],
            'J5': ['J2', 'J4']
        }
        self.state_size = 11
        self.action_size = 2
        self.yellow_duration = 3
        self.min_green_time = 10
        self._step = 0
        self._phase_time = {j:0 for j in self.intersections}
        self._current_phase = {j: 0 for j in self.intersections}
        self._red_time = {j: [0,0]for j in self.intersections}
    
    def reset(self):
        if traci.isLoaded():
            traci.close()

        
        sumo_home = os.environ.get('Sumo_Home', '')
        if self.use_gui:
            sumo_binary = os.path.join(sumo_home, 'bin', 'sumo-gui.exe')
        else:
            sumo_binary = os.path.join(sumo_home, 'bin', 'sumo.exe')
        traci.start([sumo_binary,'-c',self.config_path,'--no-warnings','--random'])
        self._step = 0
        self._phase_time = {j:0 for j in self.intersections}
        self._current_phase = {j: 0 for j in self.intersections}
        self._red_time = {j: [0,0]for j in self.intersections}
        return self._get_state()
    
    def step(self,actions):
        for junction , action in actions.items():
            if self._phase_time[junction] < self.min_green_time:
                continue
            if action ==1:
                self._set_yellow(junction)
                next_phase = (self._current_phase[junction]+1)%self.num_phases[junction]
                self._current_phase[junction] = next_phase
                self._phase_time[junction] = 0
            else:
                self._phase_time[junction] += 1
        traci.simulationStep()
        self._step += 1
        self._update_red_time()
        next_state = self._get_state()
        rewards    = self.compute_reward()
        done       = self._is_done()
        return next_state,rewards,done
    def _get_state(self):
        states = {}
        for junction in self.intersections:
            obs = []
            for lane in self.lanes[junction]:
                queue = traci.lane.getLastStepHaltingNumber(lane)
                wait = traci.lane.getWaitingTime(lane)/max(1,queue) if queue > 0 else 0.0
                obs.append(queue/50.0)
                obs.append(wait/300.0)
                obs.append(self._current_phase[junction]/max(1,self.num_phases[junction]-1))
                for neighbour in self.neighbours[junction]:
                    neigh_queue = 0
                    for lane in self.lanes[neighbour]:
                        neigh_queue += traci.lane.getLastStepHaltingNumber(lane)
                        obs.append((neigh_queue/len(self.lanes[neighbour]))/50.0)
                    states[junction] = np.array(obs,dtype=np.float32)
        return states
    
    def compute_reward(self):
        rewards = {}
        for junction in self.intersections:
            own_queue = 0
            own_wait = 0
            for lane in self.lanes[junction]:
                own_queue += traci.lane.getLastStepHaltingNumber(lane)
                own_wait += traci.lane.getWaitingTime(lane)
            if self.reward_fn == 'local':
                alpha = 0.5
                reward = -(own_queue + alpha * own_wait)
            elif self.reard_fn == 'cooperative':
                neigh_queue= 0 
                for neighbour in self.neioghbours[junction]:
                    for lane in self.lanes[neighbour]:
                        beigh_queue += traci.lane.getLastStepHaltingNumber(lane)
                    beta = 0.3
                    reward = -(own_queue + beta * neigh_queue)
            elif self.reward_fn == 'fairness':
                neigh_queue = 0
                for neighbour in self.neighbours[junction]:
                    for lane in self.lanes[neighbour]:
                        neigh_queue += traci.lane.getLastStepHaltingNumber(lane)
                max_starvation = max(self._red_time[junction])
                beta = 0.3
                lam = 0.1
                reward = -(own_queue + beta * neigh_queue + lam * max_starvation)
            else:
                ValueError(f"unknown reward fucntion: {self.reward_fn}")
            rewards[junction] = reward
        return rewards
    
    def _set_yellow(self,junction):
        yellow_phase = self.current_phase[junction] * 2 + 1
        traci.trafficlight.setPhase(junction,yellow_phase)
        for _ in range(self.yellow_duration):
            traci.simulationStep()
            self._step += 1
    
    def _update_red_time(self):
        for junction in self.intersections:
            current = self._current_phase[junction]
            for phase_idx in range(len(self._red_time[junction])):
                if phase_idx == current:
                    self._red_time[junction][phase_idx]=0
                else:
                    self._red_time[junction][phase_idx]+=1
    
    def _is_done(self):
        no_vehicles  = traci.simulation.getMinExpectedNumber() == 0
        time_up      = self._step >= 3600

        return no_vehicles or time_up
    def close(self):
        
        if traci.isLoaded():
            traci.close()

    @property
    def state_size(self):
        return self._state_size

    @state_size.setter
    def state_size(self, val):
        self._state_size = val

    @property
    def action_size(self):
        return self._action_size

    @action_size.setter
    def action_size(self, val):
        self._action_size = val
                    





