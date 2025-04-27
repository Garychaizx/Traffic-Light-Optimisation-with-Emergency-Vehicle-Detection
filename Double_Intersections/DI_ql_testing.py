# test_double_intersection.py

import os
import sys
import numpy as np
import torch
import sumo_rl
import matplotlib.pyplot as plt
import pandas as pd  # for performance metrics table
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from agents.ql_agent2 import QlAgent2
from matplotlib import pyplot as plt
import librosa
from tensorflow.keras.models import load_model
import traci

# === Emergency Detection Functions ===
def extract_features(audio_file, max_pad_len=862):
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
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dynamic_sounds', 'ambulance.wav'))
    if model is None or not os.path.exists(path):
        return False
    feats = extract_features(path)
    if feats is None:
        return False
    try:
        prediction = float(model.predict(feats)[0][0])
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
            print(f"   Route: {route}")
            print(f"   Current edge: {current_edge}")
            
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

def handle_emergency_vehicles(env, tls, actions, siren_model):
    """Handle emergency vehicles and update actions accordingly"""
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
    ts = env.ts_ids
    neigh = {tl: [] for tl in ts}
    for edge_id in env.sumo.edge.getIDList():
        if len(edge_id) == 2 and edge_id[0] in ts and edge_id[1] in ts:
            a, b = edge_id[0], edge_id[1]
            neigh[a].append(b)
            neigh[b].append(a)
    return {tl: np.array(v) for tl, v in neigh.items()}

def prepare_obs(obs, tl, neighbours, pad_len=165):
    x = obs[tl].copy()
    for n in neighbours[tl]:
        x += 0.2 * obs[n]
    if len(x) < pad_len:
        x = np.concatenate([x, np.full(pad_len - len(x), -1.0)])
    else:
        x = x[:pad_len]
    return torch.FloatTensor(x)

def main():
    # — load siren model —
    try:
        siren_model = load_model(os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..', 'siren_model', 'best_model.keras')))
        print("✅ Siren model loaded")
    except Exception as e:
        siren_model = None
        print(f"⚠️  No siren model, skipping override: {e}")

    # — create SUMO env —
    env = sumo_rl.SumoEnvironment(
        net_file="nets/double/network.net.xml",
        route_file="nets/double/doubleRoutes_low.rou.xml",
        use_gui=True,
        num_seconds=5000,
        single_agent=False
    )
    tls = env.ts_ids

    # — build helpers —
    obs0       = env.reset()
    neighbours = build_neighbours(env)

    # --- populate lanes_by_phase & all_lanes for both intersections ---
    lanes_by_phase = {}
    all_lanes = set()
    for tl in tls:
        ctrl_lanes = traci.trafficlight.getControlledLanes(tl)
        all_lanes.update(ctrl_lanes)
        phases = env.traffic_signals[tl].all_phases
        lanes_by_phase[tl] = []
        for ph in phases:
            serve = [
                lane for lane, sig in zip(ctrl_lanes, ph.state)
                if sig.upper() == "G"
            ]
            lanes_by_phase[tl].append(serve)
    all_lanes = list(all_lanes)

    # — load agents —
    agents = {}
    for tl in tls:
        n_phases = len(env.traffic_signals[tl].all_phases)
        agent = QlAgent2(input_shape=165, output_shape=n_phases)
        ckpt = f"trained_models/ql_double_{tl}.pth"
        agent.model.load_state_dict(torch.load(ckpt))
        agent.model.eval()
        agent.epsilon = 0.0
        agents[tl] = agent

    # --- shaping weights ---
    w1, w2, w3 = 0.3, 0.2, 0.2
    w4, w5, w6 = 0.05, 1.0, 1.0

    # — initialize testing state & metrics —
    obs            = obs0
    state          = {tl: prepare_obs(obs, tl, neighbours) for tl in tls}
    done           = {"__all__": False}
    shaped_rewards = []
    rewards        = []
    queue_lengths  = []
    phase_counts   = {tl: {p: 0 for p in range(len(env.traffic_signals[tl].all_phases))}
                      for tl in tls}
    prev_actions   = {tl: None for tl in tls}

    # performance metrics
    travel_times     = []
    depart_times     = {}
    waiting_times    = []
    ev_waited        = set()
    ev_waiting_times = []

    emergency_events   = 0
    detected_emerg_ids = set()

    step = 0
    while not done["__all__"] and step < 5000:
        step += 1

        # — record depart/arrival for travel times & EV wait —
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

        # — detect emergency vehicles for stats —
        evs = detect_emergency_vehicles(env, tls, siren_model)
        if evs:
            emergency_events += 1
            detected_emerg_ids.update(ev['id'] for ev in evs)

        # — total queue before —
        Q_before = sum(traci.lane.getLastStepHaltingNumber(l) for l in all_lanes)

        # — choose actions —
        actions = {}
        for tl, agent in agents.items():
            curr  = env.traffic_signals[tl].green_phase
            valid = [
                p for p in range(agent.output_shape)
                if (curr, p) in env.traffic_signals[tl].yellow_dict
            ]
            qvals = agent.model(state[tl])
            a     = int(torch.argmax(qvals))
            actions[tl] = a if a in valid else curr

        # — emergency override —
        override = handle_emergency_vehicles(env, tls, actions, siren_model)

        # — record per-TL clears before step —
        cleared_before = {
            tl: sum(
                traci.lane.getLastStepHaltingNumber(l)
                for l in lanes_by_phase[tl][actions[tl]]
            )
            for tl in tls
        }

        # — count phase changes —
        changes = sum(
            1 for tl in tls
            if prev_actions[tl] is not None and prev_actions[tl] != actions[tl]
        )

        # — update prev_actions & phase_counts —
        for tl in tls:
            phase_counts[tl][actions[tl]] += 1
            prev_actions[tl] = actions[tl]

        # — step env —
        obs2, reward_dict, done, _ = env.step(actions)
        avg_r = np.mean(list(reward_dict.values()))
        rewards.append(avg_r)

        # — total queue after & throughput —
        Q_after = sum(traci.lane.getLastStepHaltingNumber(l) for l in all_lanes)
        C_tot   = max(0, Q_before - Q_after)

        # — queue length metric —
        total_q = 0
        for tl in tls:
            lanes = env.sumo.trafficlight.getControlledLanes(tl)
            total_q += sum(env.sumo.lane.getLastStepHaltingNumber(l) for l in lanes)
        queue_lengths.append(total_q / len(tls))

        # — sum of vehicles cleared on each chosen phase —
        sum_c_sel = sum(
            max(0, cleared_before[tl] -
                sum(traci.lane.getLastStepHaltingNumber(l)
                    for l in lanes_by_phase[tl][actions[tl]]))
            for tl in tls
        )

        # — potential-based shaping term Φ(s_{t+1})−Φ(s_t) = Q_before − Q_after —
        phi_diff = Q_before - Q_after

        # — assemble shaped reward —
        r = (
            -w1 * Q_before
            + w2 * sum_c_sel
            + w3 * C_tot
            - w4 * changes
            + w5 * phi_diff
            + (w6 if override else 0.0)
        )
        shaped_rewards.append(r)

        # — record waiting times this step & EV stops —
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

        print(f"Step {step:4d} | Avg Reward {avg_r: .3f} | Avg Queue {queue_lengths[-1]: .1f}")

        # — prepare next state —
        state = {tl: prepare_obs(obs2, tl, neighbours) for tl in tls}
        obs   = obs2

    # === Summary ===
    print("\n✅ Testing done")
    print(f"Mean reward over episode: {np.mean(rewards):.3f}")
    print(f"Mean queue   over episode: {np.mean(queue_lengths):.1f}")
    print(f"\n🚑 Emergency vehicle statistics:")
    print(f"   - Detection events: {emergency_events}")
    print(f"   - Unique emergency vehicles detected: {len(detected_emerg_ids)}")

    for tl in tls:
        print(f"\nPhase counts for {tl}:")
        for p, c in phase_counts[tl].items():
            print(f"  Phase {p}: {c}")

    # === Performance metrics table ===
    summary = {
        "wait time (sec)":     np.mean(waiting_times),
        "travel time (sec)":   np.mean(travel_times),
        "queue length (cars)": np.mean(queue_lengths),
        "reward":       np.mean(shaped_rewards),
        "EV stopped count":    len(ev_waited),
        "EV avg wait (sec)":   np.mean(ev_waiting_times) if ev_waiting_times else 0.0
    }
    df = pd.DataFrame([summary])
    print("\nPerformance metrics")
    print(df.to_markdown(index=False, floatfmt=".3f"))

    # === Plots ===
    plt.figure(figsize=(10,4))
    plt.plot(rewards, label="Avg Reward")
    plt.xlabel("Step"); plt.ylabel("Reward"); plt.title("QL Testing: Rewards")
    plt.grid(True); plt.legend()

    plt.figure(figsize=(10,4))
    plt.plot(queue_lengths, color="orange", label="Avg Queue")
    plt.xlabel("Step"); plt.ylabel("Queue"); plt.title("QL Testing: Queue Lengths")
    plt.grid(True); plt.legend()

    plt.show()

if __name__ == "__main__":
    main()
