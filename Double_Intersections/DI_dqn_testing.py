# test_dqn_double.py

import os
import sys
import numpy as np
import torch
import sumo_rl
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from agents.dqn_agent import DQNAgent
# Add these imports at the top of the file
import librosa
from tensorflow.keras.models import load_model
import traci
import pandas as pd

# Add these functions after the prepare_obs function and before main()
def extract_features(audio_file, max_pad_len=862):
    """Extract MFCC features from an audio file for siren detection"""
    try:
        audio, sr = librosa.load(audio_file, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=80)
        pad_width = max_pad_len - mfccs.shape[1]
        if pad_width > 0:
            mfccs = np.pad(mfccs, ((0, 0), (0, pad_width)), mode='constant')
        else:
            mfccs = mfccs[:, :max_pad_len]
        return np.mean(mfccs, axis=1).reshape(1, 1, 80)
    except Exception as e:
        print(f"❌ Error extracting features: {e}")
        return None

def detect_siren(model):
    """Detect if audio file contains a siren sound"""
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dynamic_sounds', 'ambulance.wav'))
    if model is None or not os.path.exists(path):
        print(f"❌ Siren model not loaded or audio file missing: {path}")
        return False
    feats = extract_features(path)
    if feats is None:
        print("❌ Failed to extract features from audio")
        return False
    try:
        prediction = float(model.predict(feats)[0][0])
        print(f"🔊 Siren detection confidence: {prediction}")
        return prediction > 0.5
    except Exception as e:
        print(f"❌ Error during siren detection: {e}")
        return False

def detect_emergency_vehicles(env, tls, siren_model):
    """Detect emergency vehicles and report their locations"""
    # Track detected emergency vehicles
    emergency_vehicles = []
    
    # Check each vehicle in the simulation
    for vid in env.sumo.vehicle.getIDList():
        if env.sumo.vehicle.getTypeID(vid) == "emergency":
            route = env.sumo.vehicle.getRoute(vid)
            position = env.sumo.vehicle.getLanePosition(vid)
            current_edge = env.sumo.vehicle.getRoadID(vid)
            speed = env.sumo.vehicle.getSpeed(vid)
            
            emergency_info = {
                'id': vid,
                'route': route,
                'current_edge': current_edge,
                'position': position,
                'speed': speed
            }
            
            # Add to detected vehicles
            emergency_vehicles.append(emergency_info)
            
            print(f"🚑 Emergency vehicle detected - ID: {vid}")
            # print(f"   Route: {route}")
            # print(f"   Current edge: {current_edge}")
            # print(f"   Position: {position:.2f} m")
            # print(f"   Speed: {speed:.2f} m/s")
            
            # Check if siren is active using audio detection
            if detect_siren(siren_model):
                print(f"🚨 SIREN ACTIVE on vehicle {vid}")
                
                # Determine the upcoming edges (current + next 2)
                route_idx = 0
                if current_edge in route:
                    route_idx = route.index(current_edge)
                upcoming_edges = route[route_idx:route_idx+3]
                print(f"   Upcoming edges: {upcoming_edges}")
                
                # For each traffic light, check which lanes the vehicle will pass through
                for tl_id in tls:
                    if tl_id not in env.traffic_signals:
                        continue
                        
                    # Get controlled lanes for this traffic light
                    tl_lanes = env.sumo.trafficlight.getControlledLanes(tl_id)
                    tl_edges = set(lane.split('_')[0] for lane in tl_lanes)
                    
                    # Check if emergency vehicle will pass through this traffic light
                    intersection_edges = [edge for edge in upcoming_edges if edge in tl_edges]
                    if intersection_edges:
                        print(f"   Will pass through traffic light {tl_id} on edges: {intersection_edges}")
                        current_phase = env.traffic_signals[tl_id].green_phase
                        print(f"   Current phase of {tl_id}: {current_phase}")
    
    return emergency_vehicles

def handle_emergency_vehicles(env, tls, siren_model, actions):
    """Detect emergency vehicles and change traffic light phases to help them pass"""
    emergency_changes_made = False
    
    # Check for emergency vehicles
    for vid in env.sumo.vehicle.getIDList():
        if env.sumo.vehicle.getTypeID(vid) == "emergency":
            route = env.sumo.vehicle.getRoute(vid)
            position = env.sumo.vehicle.getLanePosition(vid)
            current_edge = env.sumo.vehicle.getRoadID(vid)
            
            print(f"🚑 Vehicle ID: {vid}, Type: emergency, Route: {route}, Current Edge: {current_edge}, Position: {position}")
            
            # Check if siren is active
            if detect_siren(siren_model):
                print(f"🚨 Emergency vehicle detected with siren on route {route}")
                
                # Find upcoming edges in the route
                route_idx = 0
                if current_edge in route:
                    route_idx = route.index(current_edge)
                # Look at current + next edges to prepare traffic lights
                upcoming_edges = route[route_idx:route_idx+3]
                
                # For each traffic light, check if emergency vehicle will pass through
                for tl_id in tls:
                    if tl_id not in env.traffic_signals:
                        continue
                    
                    # Get controlled lanes for this traffic light
                    tl_lanes = env.sumo.trafficlight.getControlledLanes(tl_id)
                    tl_edges = set(lane.split('_')[0] for lane in tl_lanes)
                    
                    # Check if any upcoming edge is controlled by this traffic light
                    relevant_edges = [edge for edge in upcoming_edges if edge in tl_edges]
                    if relevant_edges:
                        current_phase = env.traffic_signals[tl_id].green_phase
                        
                        # Determine target phase based on the edge the vehicle is on or approaching
                        target_phase = None
                        
                        # B traffic light
                        if tl_id == "B":
                            if any(edge.startswith(("AB", "BA", "CB", "BC")) for edge in relevant_edges):
                                # East-West traffic on B intersection
                                target_phase = 0  # Phase 0 for East-West traffic at B
                            elif any(edge.startswith(("DB", "BD", "EB", "BE")) for edge in relevant_edges):
                                # North-South traffic on B intersection
                                target_phase = 4  # Phase 4 for North-South traffic at B
                        
                        # E traffic light
                        elif tl_id == "E":
                            if any(edge.startswith(("FE", "EF", "HE", "EH")) for edge in relevant_edges):
                                # East-West traffic on E intersection
                                target_phase = 0  # Phase 0 for East-West traffic at E
                            elif any(edge.startswith(("GE", "EG", "BE", "EB")) for edge in relevant_edges):
                                # North-South traffic on E intersection
                                target_phase = 4  # Phase 4 for North-South traffic at E
                        
                        # Only change phase if we have a target
                        if target_phase is not None:
                            # Get valid transitions
                            valid_transitions = [
                                p for p in range(len(env.traffic_signals[tl_id].all_phases))
                                if (current_phase, p) in env.traffic_signals[tl_id].yellow_dict
                            ]
                            
                            if target_phase in valid_transitions:
                                # Direct transition possible
                                actions[tl_id] = target_phase
                                print(f"🚦 Emergency override: {tl_id} phase {current_phase} → {target_phase} for {relevant_edges}")
                                emergency_changes_made = True
                            else:
                                # Find next phase that gets us closer to target
                                if target_phase < 4 and current_phase >= 4:
                                    # Currently in phase group 4-7, need to get to 0-3
                                    # Need to move to phase 0
                                    best_next = None
                                    for p in valid_transitions:
                                        if p < 4:
                                            best_next = p
                                            break
                                    if best_next is not None:
                                        actions[tl_id] = best_next
                                        print(f"🚦 Moving toward target: {tl_id} phase {current_phase} → {best_next} (target: {target_phase})")
                                        emergency_changes_made = True
                                elif target_phase >= 4 and current_phase < 4:
                                    # Currently in phase group 0-3, need to get to 4-7
                                    # Need to move to phase 4
                                    best_next = None
                                    for p in valid_transitions:
                                        if p >= 4:
                                            best_next = p
                                            break
                                    if best_next is not None:
                                        actions[tl_id] = best_next
                                        print(f"🚦 Moving toward target: {tl_id} phase {current_phase} → {best_next} (target: {target_phase})")
                                        emergency_changes_made = True
    
    return emergency_changes_made

def build_neighbours(env):
    """
    After env.reset(), build a map from each TL id to the
    list of its two‑char neighbours (edges like "AB", "BC", etc.).
    """
    ts = env.ts_ids
    neigh = {tl: [] for tl in ts}
    for eid in env.sumo.edge.getIDList():
        if len(eid) == 2 and eid[0] in ts and eid[1] in ts:
            a, b = eid[0], eid[1]
            neigh[a].append(b)
            neigh[b].append(a)
    return {tl: list(set(lst)) for tl, lst in neigh.items()}

def prepare_obs(obs, tl, neighbours, pad_len=165):
    """
    Stack own observation + 0.2× each neighbour’s,
    then pad (with -1) or truncate to pad_len.
    Returns a flat numpy array.
    """
    x = obs[tl].copy()
    for n in neighbours.get(tl, []):
        x += 0.2 * obs[n]
    if len(x) < pad_len:
        x = np.concatenate([x, np.full(pad_len - len(x), -1.0)])
    else:
        x = x[:pad_len]
    return x

# Add these variables to track performance metrics
travel_times = []
depart_times = {}
waiting_times = []
ev_waited = set()
ev_waiting_times = []
shaped_rewards = []
queue_lengths = []

def main():
    # 1) Create the SUMO environment with GUI for visualization
    env = sumo_rl.SumoEnvironment(
        net_file    = "nets/double/network.net.xml",
        route_file  = "nets/double/doubleRoutes_high.rou.xml",
        use_gui     = True,
        num_seconds = 5000,
        single_agent=False
    )

    # 2) Reset and build neighbour map
    obs0 = env.reset()
    tls  = env.ts_ids
    neighbours = build_neighbours(env)

    # Load siren detection model if available
    try:
        siren_model = load_model(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'siren_model/best_model.keras')))
        print("✅ Siren detection model loaded successfully")
    except Exception as e:
        siren_model = None
        print(f"⚠️ Couldn't load siren model: {e}")

    # 3) Instantiate and load one DQNAgent per TL
    agents = {}
    for tl in tls:
        n_phases = len(env.traffic_signals[tl].all_phases)
        agent = DQNAgent(
            state_dim   = 165,
            action_dim  = n_phases,
            gamma       = 0.99,
            lr          = 1e-3,
            epsilon     = 0.0,      # greedy
            min_epsilon = 0.0,
            epsilon_decay = 0.0,
            batch_size  = 64,
            memory_size = 1        # dummy
        )
        ckpt = f"trained_models/dqn_double_{tl}.pth"
        agent.load(ckpt)
        agents[tl] = agent

    # 4) Run one evaluation episode
    obs    = env.reset()
    state  = {tl: prepare_obs(obs, tl, neighbours) for tl in tls}
    done   = {"__all__": False}
    rewards = []
    queues   = []
    phase_counts = {tl: {p:0 for p in range(len(env.traffic_signals[tl].all_phases))} for tl in tls}
    step = 0
    MAX_STEPS = 5000

    # Metrics for emergency vehicles
    emergency_detection_count = 0
    emergency_vehicles_detected = set()

    # Add weights for shaping (same as QL)
    w1, w2, w3 = 0.3, 0.2, 0.2
    w4, w5, w6 = 0.05, 1.0, 1.0

    # Initialize additional metrics
    sum_c_sel = 0
    changes = 0

    # Define all_lanes as the set of all lanes controlled by all traffic lights
    all_lanes = set()
    for tl in tls:
        lanes = env.sumo.trafficlight.getControlledLanes(tl)
        all_lanes.update(lanes)
    all_lanes = list(all_lanes)  # Convert to a list for iteration

    # Define lanes_by_phase for each traffic light
    lanes_by_phase = {}
    for tl in tls:
        phases = env.traffic_signals[tl].all_phases
        lanes_by_phase[tl] = [
            env.sumo.trafficlight.getControlledLanes(tl)  # All lanes controlled by this traffic light
            for _ in phases
        ]

    # Initialize previous actions for each traffic light
    prev_actions = {tl: None for tl in tls}

    while not done["__all__"] and step < MAX_STEPS:
        step += 1
        actions = {}

        # First detect emergency vehicles
        emergency_vehicles = detect_emergency_vehicles(env, tls, siren_model)
        if emergency_vehicles:
            emergency_detection_count += 1
            for ev in emergency_vehicles:
                emergency_vehicles_detected.add(ev['id'])

        # Select default agent actions
        for tl, agent in agents.items():
            curr = env.traffic_signals[tl].green_phase
            valid = [p for p in range(agent.action_dim)
                    if (curr, p) in env.traffic_signals[tl].yellow_dict]
            a = agent.choose_action(state[tl], valid)
            actions[tl] = a

        # Override with emergency vehicle priorities if needed
        emergency_override = handle_emergency_vehicles(env, tls, siren_model, actions)

        # Update phase counts only after potential override
        for tl in tls:
            if tl in actions:
                phase_counts[tl][actions[tl]] += 1

        # Calculate total queue before
        Q_before = sum(traci.lane.getLastStepHaltingNumber(l) for l in all_lanes)

        # Calculate vehicles cleared in selected phases
        cleared_before = {
            tl: sum(
                traci.lane.getLastStepHaltingNumber(l)
                for l in lanes_by_phase[tl][actions[tl]]
            )
            for tl in tls
        }

        # Count phase changes
        changes = sum(
            1 for tl in tls
            if prev_actions[tl] is not None and prev_actions[tl] != actions[tl]
        )

        # Update previous actions
        for tl in tls:
            prev_actions[tl] = actions[tl]

        # Step environment
        obs2, reward_dict, done, _ = env.step(actions)

        # Calculate total queue after and throughput
        Q_after = sum(traci.lane.getLastStepHaltingNumber(l) for l in all_lanes)
        C_tot = max(0, Q_before - Q_after)

        # Calculate sum of vehicles cleared in selected phases
        sum_c_sel = sum(
            max(0, cleared_before[tl] -
                sum(traci.lane.getLastStepHaltingNumber(l)
                    for l in lanes_by_phase[tl][actions[tl]]))
            for tl in tls
        )

        # Calculate potential-based shaping term
        phi_diff = Q_before - Q_after

        # Compute shaped reward (same as QL)
        r = (
            -w1 * Q_before
            + w2 * sum_c_sel
            + w3 * C_tot
            - w4 * changes
            + w5 * phi_diff
            + (w6 if emergency_override else 0.0)
        )
        shaped_rewards.append(r)

        # Calculate average reward
        avg_r = np.mean(list(reward_dict.values())) if reward_dict else 0.0
        rewards.append(avg_r)

        # Ensure queue_lengths is not empty before accessing the last element
        avg_queue = queue_lengths[-1] if queue_lengths else 0.0

        # Ensure shaped_rewards is not empty before accessing the last element
        shaped_reward = r if 'r' in locals() else 0.0

        # Print metrics for debugging
        print(f"Step {step:4d} | Avg Reward {avg_r: .3f} | Avg Queue {avg_queue: .1f} | Shaped Reward {shaped_reward: .3f}")

        # Record average reward
        avg_r = np.mean(list(reward_dict.values()))
        rewards.append(avg_r)

        # Record average queue length
        total_q = 0
        for tl in tls:
            lanes = env.sumo.trafficlight.getControlledLanes(tl)
            total_q += sum(env.sumo.lane.getLastStepHaltingNumber(l) for l in lanes)
        avg_queue = total_q / len(tls)
        queue_lengths.append(avg_queue)

        # Record travel times and waiting times
        for vid in traci.simulation.getDepartedIDList():
            depart_times[vid] = step
        for vid in traci.simulation.getArrivedIDList():
            if vid in depart_times:
                travel_times.append(step - depart_times[vid])
                try:
                    if traci.vehicle.getTypeID(vid) == "emergency_veh":
                        ev_waiting_times.append(traci.vehicle.getWaitingTime(vid))
                except:
                    pass
                del depart_times[vid]

        # Calculate throughput (total vehicles that completed their routes)
        throughput = len(travel_times)

        vehs = env.sumo.vehicle.getIDList()
        if vehs:
            waiting_times.append(np.mean([traci.vehicle.getWaitingTime(v) for v in vehs]))
        else:
            waiting_times.append(0.0)

        for v in vehs:
            if traci.vehicle.getTypeID(v) == "emergency_veh" and traci.vehicle.getWaitingTime(v) > 0:
                if v not in ev_waited:
                    ev_waited.add(v)
                    ev_waiting_times.append(traci.vehicle.getWaitingTime(v))

        print(f"Step {step:4d} | Avg Reward {avg_r: .3f} | Avg Queue {avg_queue: .1f} | Shaped Reward {r: .3f}")

        # Prepare next state
        state = {tl: prepare_obs(obs2, tl, neighbours) for tl in tls}

    # === Summary ===
    print("\n✅ Evaluation complete.")
    print(f"Mean reward over episode: {np.mean(rewards):.3f}")
    print(f"Mean queue   over episode: {np.mean(queue_lengths):.1f}")
    print(f"Total throughput (vehicles): {throughput}")
    print(f"\n🚑 Emergency vehicle statistics:")
    print(f"   - Detection events: {emergency_detection_count}")
    print(f"   - Unique emergency vehicles detected: {len(emergency_vehicles_detected)}")

    for tl in tls:
        print(f"\nPhase counts for {tl}:")
        for p, c in phase_counts[tl].items():
            print(f"  Phase {p}: {c}")

    # === Performance metrics table ===
    summary = {
        "wait time (sec)":     np.mean(waiting_times),
        "travel time (sec)":   np.mean(travel_times),
        "queue length (cars)": np.mean(queue_lengths),
        "shaped reward":       np.mean(shaped_rewards),
        "EV stopped count":    len(ev_waited),
        "EV avg wait (sec)":   np.mean(ev_waiting_times) if ev_waiting_times else 0.0,
        "throughput (vehicles)": throughput  # Add throughput to the summary
    }
    df = pd.DataFrame([summary])
    print("\nPerformance metrics")
    print(df.to_markdown(index=False, floatfmt=".3f"))

    # 6) Plotting
    plt.figure(figsize=(10,4))
    plt.plot(rewards, label="Avg Reward")
    plt.xlabel("Step"); plt.ylabel("Reward"); plt.title("DQN Double Intersection: Reward")
    plt.grid(True); plt.legend()

    plt.figure(figsize=(10,4))
    plt.plot(queue_lengths, color="orange", label="Avg Queue")
    plt.xlabel("Step"); plt.ylabel("Queue Length"); plt.title("DQN Double Intersection: Queue")
    plt.grid(True); plt.legend()

    plt.show()
    env.close()

if __name__ == "__main__":
    main()
