# test_double_intersection.py

import os
import sys
import numpy as np
import torch
import sumo_rl
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from agents.ql_agent2 import QlAgent2
from matplotlib import pyplot as plt
import librosa
from tensorflow.keras.models import load_model

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
        print(f"❌ Siren model not loaded or audio file missing: {path}")
        return False
    feats = extract_features(path)
    if feats is None:
        print("❌ Failed to extract features from audio")
        return False
    try:
        prediction = float(model.predict(feats)[0][0])
        return prediction > 0.5
    except Exception as e:
        print(f"❌ Error during siren detection: {e}")
        return False
    # Update the imports if needed - looks like you already have the needed ones

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
    """Build a map of neighbouring traffic lights by looking
    for two‐letter edge IDs linking TLs."""
    ts = env.ts_ids
    neigh = {tl: [] for tl in ts}
    for edge_id in env.sumo.edge.getIDList():
        # look for exactly two‐char edge IDs where both chars are TL IDs
        if len(edge_id) == 2 and edge_id[0] in ts and edge_id[1] in ts:
            a, b = edge_id[0], edge_id[1]
            neigh[a].append(b)
            neigh[b].append(a)
    return {tl: np.array(v) for tl, v in neigh.items()}

def prepare_obs(obs, tl, neighbours, pad_len=165):
    """Concatenate own obs plus 0.2× neighbours’ obs, then pad/truncate."""
    x = obs[tl].copy()
    for n in neighbours[tl]:
        x = x + 0.2 * obs[n]
    if len(x) < pad_len:
        x = np.concatenate([x, np.full(pad_len - len(x), -1.0)])
    else:
        x = x[:pad_len]
    return torch.FloatTensor(x)

def main():
    try:
        siren_model = load_model(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'siren_model/best_model.keras')))
        print("✅ Siren model loaded")
    except Exception as e:
        siren_model = None
        print(f"⚠️  No siren model, skipping override: {e}")
        
    # 1) create the SUMO env
    env = sumo_rl.SumoEnvironment(
        net_file    = "nets/double/network.net.xml",
        route_file  = "nets/double/doubleRoutes.rou.xml",
        use_gui     = True,
        num_seconds = 5000,
        single_agent=False
    )

    obs = env.reset()

    # 2) build neighbour map
    neighbours = build_neighbours(env)
    
    # Get traffic light IDs
    tls = env.ts_ids

    # 3) load each TL's trained Q‑agent
    agents = {}
    for tl in tls:
        n_phases = len(env.traffic_signals[tl].all_phases)
        agent = QlAgent2(input_shape=165, output_shape=n_phases)
        ckpt = f"trained_models/ql_double_{tl}.pth"
        agent.model.load_state_dict(torch.load(ckpt))
        agent.model.eval()
        agent.epsilon = 0.0    # fully greedy
        agents[tl] = agent

    # 4) run one test episode
    obs = env.reset()
    state = {tl: prepare_obs(obs, tl, neighbours) for tl in tls}

    done = {"__all__": False}
    rewards = []
    queues  = []
    phase_counts = {
        tl: {p: 0 for p in range(len(env.traffic_signals[tl].all_phases))}
        for tl in tls
    }
    
    # Metrics for emergency vehicles
    emergency_detection_count = 0
    emergency_vehicles_detected = set()

    step = 0
    while not done["__all__"] and step < 5000:
        step += 1

        # Detect emergency vehicles (for statistics and visualization)
        emergency_vehicles = detect_emergency_vehicles(env, tls, siren_model)
        if emergency_vehicles:
            emergency_detection_count += 1
            for ev in emergency_vehicles:
                emergency_vehicles_detected.add(ev['id'])

        # Choose a greedy, valid phase for each TL
        actions = {}
        for tl, agent in agents.items():
            curr = env.traffic_signals[tl].green_phase
            # Valid next phases according to yellow_dict
            valid = [
                p for p in range(agent.output_shape)
                if (curr, p) in env.traffic_signals[tl].yellow_dict
            ]

            # Get default action from agent
            q_vals = agent.model(state[tl])
            action = int(torch.argmax(q_vals))
            if action not in valid:
                action = curr
            
            actions[tl] = action

        # Override with emergency vehicle priorities if needed
        emergency_override = handle_emergency_vehicles(env, tls, actions, siren_model)

        # Update phase counts with possibly modified actions
        for tl in tls:
            phase_counts[tl][actions[tl]] += 1

        # Step
        obs2, reward_dict, done, _ = env.step(actions)

        # Record metrics
        avg_r = np.mean(list(reward_dict.values()))
        rewards.append(avg_r)

        # Average queue over all TLs
        total_q = 0
        for tl in tls:
            lanes = env.sumo.trafficlight.getControlledLanes(tl)
            total_q += sum(env.sumo.lane.getLastStepHaltingNumber(l) for l in lanes)
        queues.append(total_q / len(tls))

        print(f"Step {step:4d} | Avg Reward {avg_r: .3f} | Avg Queue {queues[-1]: .1f}")

        # Prepare next state
        state = {tl: prepare_obs(obs2, tl, neighbours) for tl in tls}

    # Summary
    print("\n✅ Testing done")
    print(f"Mean reward over episode: {np.mean(rewards):.3f}")
    print(f"Mean queue over episode: {np.mean(queues):.1f}")
    print(f"\n🚑 Emergency vehicle statistics:")
    print(f"   - Detection events: {emergency_detection_count}")
    print(f"   - Unique emergency vehicles detected: {len(emergency_vehicles_detected)}")
    
    for tl in tls:
        print(f"\nPhase counts for {tl}:")
        for p, c in phase_counts[tl].items():
            print(f"  Phase {p}: {c}")

    # Rest of the plotting code remains unchanged
        # step
        obs2, reward_dict, done, _ = env.step(actions)

        # record metrics
        avg_r = np.mean(list(reward_dict.values()))
        rewards.append(avg_r)

        # average queue over all TLs
        total_q = 0
        for tl in env.ts_ids:
            lanes = env.sumo.trafficlight.getControlledLanes(tl)
            total_q += sum(env.sumo.lane.getLastStepHaltingNumber(l) for l in lanes)
        queues.append(total_q / len(env.ts_ids))

        print(f"Step {step:4d} | Avg Reward {avg_r: .3f} | Avg Queue {queues[-1]: .1f}")

        # prepare next state
        state = {tl: prepare_obs(obs2, tl, neighbours) for tl in env.ts_ids}

    # summary
    print("\n✅ Testing done")
    print(f"Mean reward over episode: {np.mean(rewards):.3f}")
    print(f"Mean queue   over episode: {np.mean(queues):.1f}")
    for tl in env.ts_ids:
        print(f"\nPhase counts for {tl}:")
        for p, c in phase_counts[tl].items():
            print(f"  Phase {p}: {c}")

    # plots
    plt.figure(figsize=(10,4))
    plt.plot(rewards, label="Avg Reward")
    plt.xlabel("Step"); plt.ylabel("Reward"); plt.title("QL Testing: Rewards")
    plt.grid(True); plt.legend()

    plt.figure(figsize=(10,4))
    plt.plot(queues, color="orange", label="Avg Queue")
    plt.xlabel("Step"); plt.ylabel("Queue"); plt.title("QL Testing: Queue Lengths")
    plt.grid(True); plt.legend()

    plt.show()

if __name__ == "__main__":
    main()
