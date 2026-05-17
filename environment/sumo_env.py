import os
import traci
import numpy as np
import platform

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
        self.state_size = 20
        self.action_size = 2
        self.yellow_duration = 3
        self.min_green_time = 10

        self._step = 0                                              # how many simulation step has passed
        self._phase_time = {j: 0 for j in self.intersections}       # how long the current phase has been green at each intersection
        self._current_phase = {j: 0 for j in self.intersections}    # which phase (0 or 1) each intersection is currently on, 0 means green horizontal, 1 means green vertical
        self._red_time = {j: [0, 0] for j in self.intersections}    # how long has a phase in intersectin been waiting (red)

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
        return self._get_state()
    
    def step(self,actions):
        for junction , action in actions.items():
            self._phase_time[junction]+=1
            if self._phase_time[junction] < self.min_green_time:
                traci.trafficlight.setPhase(junction,
                                            self._current_phase[junction]*2)
                continue
            if action == 1:
                self._set_yellow(junction)
                next_phase = (self._current_phase[junction] + 1) % self.num_phases[junction]
                self._current_phase[junction]=next_phase
                self._phase_time[junction]=0
            else:
                traci.trafficlight.setPhase(junction,self._current_phase[junction]*2)
        traci.simulationStep()
        self._step += 1
        self._update_red_time()
        next_state = self._get_state()
        rewards    = self.compute_reward()
        done       = self._is_done()
        return next_state,rewards,done
    def _get_state(self):
        # FIX #1: restructured so phase and neighbour features are added once per junction,
        #         not once per lane. Produces exactly state_size=11 features.
        states = {}
        for junction in self.intersections:
            obs = []

            # 4 lanes × 2 features = 8 values
            for lane in self.lanes[junction]:
                queue = traci.lane.getLastStepHaltingNumber(lane)
                wait = traci.lane.getWaitingTime(lane)/max(1,queue) if queue > 0 else 0.0
                obs.append(queue/50.0)
                obs.append(wait/300.0)
                obs.append(self._current_phase[junction]/max(1,self.num_phases[junction]-1))
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
                own_wait += traci.lane.getWaitingTime(lane)

            if self.reward_fn == 'local':
                alpha = 0.5
                reward = -(own_queue + alpha * own_wait)
                reward = reward / 100.0
            elif self.reward_fn == 'cooperative':
                neigh_queue= 0 
                for neighbour in self.neighbours[junction]:
                    for lane in self.lanes[neighbour]:
                        neigh_queue += traci.lane.getLastStepHaltingNumber(lane)
                beta = 0.3
                reward = -(own_queue + beta * neigh_queue)
                reward = reward / 100.0
            elif self.reward_fn == 'fairness':
                neigh_queue = 0
                for neighbour in self.neighbours[junction]:
                    for lane in self.lanes[neighbour]:
                        neigh_queue += traci.lane.getLastStepHaltingNumber(lane)
                max_starvation = max(self._red_time[junction])
                beta = 0.3
                lam = 0.1
                reward = -(own_queue + beta * neigh_queue + lam * max_starvation)
                reward = reward / 100.0
            else:
                # FIX #4: actually raise the ValueError
                raise ValueError(f"Unknown reward function: {self.reward_fn}")

            rewards[junction] = reward
        return rewards

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
        for _ in range(self.yellow_duration):
            traci.simulationStep()
            self._step += 1

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