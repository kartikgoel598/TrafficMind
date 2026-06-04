import traci
 
 
def get_green_red_queues(env, junction: str) -> tuple[float, float]:
    """
    Returns (green_queue, red_queue) — total halting vehicles
    on the currently green lanes vs the currently red lanes.
 
    Lane ordering in env.lanes for all junctions: [horizontal_0, horizontal_1, vertical_0, vertical_1]
    Phase 0 = green vertical (indices 2,3), phase 1 = green horizontal (indices 0,1).
    Verified against grid.net.xml tlLogic: phase 0 state "GGgrrrGGgrrr"
    maps E8_0 (-E3_0) as green = vertical lanes = indices 2,3 in env.lanes.
    """
    current_phase = env._current_phase[junction]
    lanes = env.lanes[junction]
 
    # indices 0,1 = horizontal (E0_0, -E1_0 style), indices 2,3 = vertical (-E3_0, E8_0 style)
    if current_phase == 0:
        green_lanes = [lanes[2], lanes[3]]  # vertical green
        red_lanes   = [lanes[0], lanes[1]]
    else:
        green_lanes = [lanes[0], lanes[1]]  # horizontal green
        red_lanes   = [lanes[2], lanes[3]]
 
    green_queue = sum(traci.lane.getLastStepHaltingNumber(l) for l in green_lanes)
    red_queue   = sum(traci.lane.getLastStepHaltingNumber(l) for l in red_lanes)
 
    return green_queue, red_queue