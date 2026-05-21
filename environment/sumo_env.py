import os
import traci
import numpy as np
import platform
from rewards.local import local_reward
from rewards.cooperative import cooperative_reward
from rewards.fairness import fairness_reward

class SumoEnvironment:
    def __init__(self,config_path,reward_fn='local',use_gui=False,seed = 42):
        self.seed = seed
        self.config_path = config_path
        self.reward_fn = reward_fn
        self.use_gui = use_gui
        self.intersections = ['J1', 'J2', 'J4', 'J5']
        self.num_phases = { # 2 phases (0 : green horizontal, 1 : green vertical)
            "J1": 2,
            "J2": 2,
            "J4": 2,
            "J5": 2
        }
        self.lanes = { # lane from the netedit
            "J1": ['E0_0', '-E1_0', '-E3_0', 'E8_0'],
            'J2': ['E1_0', '-E2_0', 'E5_0', 'E10_0'],
            'J4': ['E7_0', '-E4_0', 'E3_0', '-E9_0'],
            'J5': ['E4_0', '-E6_0', '-E5_0', '-E11_0'],
        }

        self.neighbours = { # neighbour for cooperative
            'J1': ['J2', 'J4'],
            "J2": ['J1', 'J5'],
            'J4': ['J1', 'J5'],
            'J5': ['J2', 'J4']
        }
        self.state_size = 12 # added new state from 11 to 12 which is phase_time
        self.action_size = 2
        self.yellow_duration = 3
        self.min_green_time = 15

        self._step = 0                                              # how many simulation step has passed
        self._phase_time = {j: 0 for j in self.intersections}       # how long the current phase has been green at each intersection
        self._current_phase = {j: 0 for j in self.intersections}    # which phase (0 or 1) each intersection is currently on, 0 means green horizontal, 1 means green vertical
        self._red_time = {j: [0, 0] for j in self.intersections}    # how long has a phase in intersectin been waiting (red)
        self._yellow_timer = {j: 0 for j in self.intersections}     # how long has it been yellow in an intersection (part of 1.2 critical fix)

    def reset(self):
        if traci.isLoaded():
            traci.close()

        sumo_home = os.environ.get('Sumo_Home', '')


        if platform.system() == 'Windows':
            binary = 'sumo-gui.exe' if self.use_gui else 'sumo.exe'
        else:
            binary = 'sumo-gui' if self.use_gui else 'sumo'
        sumo_binary = os.path.join(sumo_home, 'bin', binary)

        if not os.path.isfile(sumo_binary):
            raise FileNotFoundError(f"SUMO binary not found: {sumo_binary}")
        traci.start([sumo_binary, '-c', self.config_path, '--no-warnings', f'--seed={self.seed}'])
        self._step = 0
        self._phase_time = {j: 0 for j in self.intersections}
        self._current_phase = {j: 0 for j in self.intersections}
        self._red_time = {j: [0, 0] for j in self.intersections}
        self._yellow_timer = {j: 0 for j in self.intersections}
        return self._get_state()
    
    # executed every step (for 900 steps)
    '''
    18/5/2026
    fix 1    : action is stored to memory even if the sumo does not execute it. Added executed_actions to save if the agent want to switch to memory, 
               even though SUMO cannot execute it (because min-green time) 
    21/5/2026
    fix 2 : 1.2 Critical fix, remove step in _set_yellow and added yellow_state machine here
    '''
    def step(self,actions):
        # track what SUMO actually executed (if agent want to changem, but min_green blocked, then store 0)
        executed_actions = {}

        for junction, action in actions.items():

            # yello state machine, id yellow cooldown, keep counting down
            if self._yellow_timer[junction] > 0:
                self._yellow_timer[junction] -= 1
                if self._yellow_timer[junction] == 0:
                    # yellow finished set to green
                    traci.trafficlight.setPhase(junction, self._current_phase[junction] * 2)
                executed_actions[junction] = 0
                continue


            self._phase_time[junction] += 1

            if self._phase_time[junction] < self.min_green_time:
                # min_green blocked the switch, treat as KEEP even if the agent want to switch
                # * 2 to follow SUMO phase indexing (0 : green horizontal, 2: green vertical)
                traci.trafficlight.setPhase(junction, self._current_phase[junction] * 2)
                executed_actions[junction] = 0 # keep
                continue
            
            if action == 1 and self._phase_time[junction] >= self.min_green_time:
                self._set_yellow(junction)
                self._yellow_timer[junction] = self.yellow_duration
                next_phase = (self._current_phase[junction] + 1) % self.num_phases[junction]
                self._current_phase[junction]=next_phase
                self._phase_time[junction] = 0
                executed_actions[junction] = 1 # actually changed
            else:
                traci.trafficlight.setPhase(junction,self._current_phase[junction]*2)
                executed_actions[junction] = 0 # keep

        traci.simulationStep()
        self._step += 1
        self._update_red_time()
        next_state = self._get_state()
        rewards    = self.compute_reward()
        done       = self._is_done()
        return next_state, rewards, done, executed_actions

    # return phase of specified junction in the SUMO currently
    def _get_true_phase(self, junction):
        '''
        0 (green horizontal) -> 0
        1 (yellow horizontal) -> 0
        2 (green vertical) -> 1
        3 (yellow vertical) -> 1 
        '''
        sumo_phase =  traci.trafficlight.getPhase(junction)
        return sumo_phase // 2

    '''
    18/5/2026
    fix 1 : restructured so phase and neighbour features are added once per junction, not once per lane. Produces exactly state_size= 11 features.
    fix 2 : during the yellow transition in _set_yellow(), SUMO is on phase 1 or 3 (yellow), but self._current_phase MAY already got updated to the next 
            green phase. so if _get_state() is ever called mid-yellow, Python and SUMO disagree. Added get true phase and adjusted _get_state
    fix 3 : added phase_time to the agent so the agent know whether it is possible to switch. No way to learn "don't bother switching", this helps the model
            learn that its impossible.
    '''
    # return 12 states of all junction
    def _get_state(self):
        states = {}

        for junction in self.intersections:
            obs = []

            # 4 lanes × 2 features = 8 values
            for lane in self.lanes[junction]:
                queue = traci.lane.getLastStepHaltingNumber(lane)
                wait = traci.lane.getWaitingTime(lane)/max(1,queue) if queue > 0 else 0.0
                obs.append(queue/50.0)
                obs.append(wait/300.0)

            true_phase = self._get_true_phase(junction)
            obs.append(true_phase / max(1, self.num_phases[junction] - 1))

            # phase_time normalised by min_green  — FIX #3
            # value < 1.0 means switching is still blocked, >= 1.0 means switching is allowed
            obs.append(min(self._phase_time[junction] / self.min_green_time, 1.0))


            for neighbour in self.neighbours[junction]:
                neigh_q = sum(traci.lane.getLastStepHaltingNumber(l) for l in self.lanes[neighbour])
                obs.append(neigh_q/len(self.lanes[neighbour])/50.0)
            states[junction] = np.array(obs,dtype=np.float32)
                
        return states

    def compute_reward(self):
        rewards = {}
        for junction in self.intersections:
            own_queue = 0
            own_wait = 0

            for lane in self.lanes[junction]:
                own_queue += traci.lane.getLastStepHaltingNumber(lane)
                own_wait += min(traci.lane.getWaitingTime(lane), 300.0)
            
            if self.reward_fn == 'local':
                reward = local_reward(own_queue, own_wait)

            elif self.reward_fn == 'cooperative':
                neigh_queues = []
                for neighbour in self.neighbours[junction]:
                    nq = sum(
                        traci.lane.getLastStepHaltingNumber(l)
                        for l in self.lanes[neighbour]
                    )
                    neigh_queues.append(nq)
                reward = cooperative_reward(own_queue, own_wait, neigh_queues)


            elif self.reward_fn == 'fairness':
                neigh_queues = []
                for neighbour in self.neighbours[junction]:
                    nq = sum(
                        traci.lane.getLastStepHaltingNumber(l)
                        for l in self.lanes[neighbour]
        )
                    neigh_queues.append(nq)
                max_starvation = max(self._red_time[junction])
                reward = fairness_reward(
        own_queue, own_wait, neigh_queues, max_starvation
    )
            else:
                raise ValueError(f"Unknown reward function: {self.reward_fn}")

            rewards[junction] = reward
        return rewards

    '''
    21/5/2026
    fix 1: step yellow should only handle signal logic, NO SIMULATION STEP (self.step += 1). this fixes error 1.2 Critical - One RL transition spans variable numbers of SUMO seconds
    '''
    def _set_yellow(self, junction):
        # FIX #2: was self.current_phase — fixed to self._current_phase
        # in sumo .net file 
        # SUMO PHASE INDEX          | MEANING
        # 0                         | green horizontal
        # 1                         | yellow horizontal
        # 2                         | green vertical
        # 3                         | yellow vertical

        # so 0 will result in 1 (yellow horizontal) and 1 will result in 3 (yellow vertical)
        yellow_phase = self._current_phase[junction] * 2 + 1
        traci.trafficlight.setPhase(junction, yellow_phase)

    def _update_red_time(self): # self._red_time = {j: [0, 0] for j in self.intersections}
        for junction in self.intersections:
            current = self._current_phase[junction]
            for phase_idx in range(len(self._red_time[junction])):
                if phase_idx == current:
                    self._red_time[junction][phase_idx] = 0
                else:
                    self._red_time[junction][phase_idx] += 1

    def _is_done(self):
        no_vehicles  = traci.simulation.getMinExpectedNumber() == 0
        time_up      = self._step >= 900

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