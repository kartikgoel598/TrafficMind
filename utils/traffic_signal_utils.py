import traci


def get_current_green_lanes(env, junction):
    """
    Return incoming lanes that currently have green signal at a junction.
    Uses SUMO's actual traffic light state instead of hard-coded lane indices.
    """
    green_lanes = set()

    signal_state = traci.trafficlight.getRedYellowGreenState(junction)
    controlled_links = traci.trafficlight.getControlledLinks(junction)

    for signal_index, signal_char in enumerate(signal_state):
        if signal_char not in ("g", "G"):
            continue

        for link in controlled_links[signal_index]:
            incoming_lane = link[0]

            if incoming_lane in env.lanes[junction]:
                green_lanes.add(incoming_lane)

    return green_lanes


def get_green_red_queues(env, junction):
    """
    Return queue totals for currently green lanes and red lanes.
    """
    green_lanes = get_current_green_lanes(env, junction)

    all_lanes = set(env.lanes[junction])
    red_lanes = all_lanes - green_lanes

    green_queue = sum(
        traci.lane.getLastStepHaltingNumber(lane)
        for lane in green_lanes
    )

    red_queue = sum(
        traci.lane.getLastStepHaltingNumber(lane)
        for lane in red_lanes
    )

    return green_queue, red_queue