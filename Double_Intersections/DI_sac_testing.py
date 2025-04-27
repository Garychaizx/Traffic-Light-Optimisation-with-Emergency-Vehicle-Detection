# test_sac_double.py

import os
import sys
import numpy as np
import torch
import sumo_rl
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from agents.sac_agent import SACAgent
import librosa
from tensorflow.keras.models import load_model
import traci
import pandas as pd
import torch.nn.functional as F

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

def handle_emergency_vehicles(env, tls, siren_model, actions):
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
    """Map each TL to its neighbours via two‐char edge IDs."""
    ts = env.ts_ids
    neigh = {tl: [] for tl in ts}
    for eid in env.sumo.edge.getIDList():
        if len(eid) == 2 and eid[0] in ts and eid[1] in ts:
            a, b = eid[0], eid[1]
            neigh[a].append(b)
            neigh[b].append(a)
    return {tl: list(set(lst)) for tl, lst in neigh.items()}

def prepare_obs(obs, tl, neighbours, pad_len=165):
    """Concatenate own obs + 0.2× each neighbour’s, pad/truncate to pad_len."""
    x = obs[tl].copy()
    for n in neighbours.get(tl, []):
        x += 0.2 * obs[n]
    if len(x) < pad_len:
        x = np.concatenate([x, np.full(pad_len - len(x), -1.0)])
    else:
        x = x[:pad_len]
    return x

def main():
    try:
        siren_model = load_model(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'siren_model/best_model.keras')))
        print("✅ Siren model loaded")
    except Exception as e:
        siren_model = None
        print(f"⚠️  No siren model, skipping override: {e}")

    # Create SUMO environment
    env = sumo_rl.SumoEnvironment(
        net_file="nets/double/network.net.xml",
        route_file="nets/double/doubleRoutes_high.rou.xml",
        use_gui=True,
        num_seconds=5000,
        single_agent=False
    )

    try:
        # Initialize and build neighbour map
        obs0 = env.reset()
        tls = env.ts_ids
        neighbours = build_neighbours(env)

        # Create and load SACAgent for each TL
        agents = {}
        for tl in tls:
            n_ph = len(env.traffic_signals[tl].all_phases)
            ag = SACAgent(
                state_dim=165,
                action_dim=n_ph,
                gamma=0.99,
                tau=0.005,
                alpha=0.2,
                lr=3e-4
            )
            ckpt = torch.load(f"trained_models/sac_double_{tl}.pth", map_location="cpu")
            ag.actor.load_state_dict(ckpt["actor"])
            ag.critic_1.load_state_dict(ckpt["critic_1"])
            ag.critic_2.load_state_dict(ckpt["critic_2"])
            ag.target_critic_1.load_state_dict(ckpt["target_1"])
            ag.target_critic_2.load_state_dict(ckpt["target_2"])
            ag.actor.eval()
            agents[tl] = ag

        # Add performance metrics variables
        travel_times = []
        depart_times = {}
        waiting_times = []
        ev_waited = set()
        ev_waiting_times = []
        shaped_rewards = []
        queue_lengths = []

        # Run one test episode
        state = {tl: prepare_obs(obs0, tl, neighbours) for tl in tls}
        done = {"__all__": False}
        rewards = []
        queues = []
        phase_counts = {
            tl: {p: 0 for p in range(len(env.traffic_signals[tl].all_phases))}
            for tl in tls
        }

        emergency_detection_count = 0
        emergency_vehicles_detected = set()

        step = 0
        MAX_STEPS = 5000

        while not done["__all__"] and step < MAX_STEPS:
            step += 1
            actions = {}

            # Detect emergency vehicles
            emergency_vehicles = detect_emergency_vehicles(env, tls, siren_model)
            if emergency_vehicles:
                emergency_detection_count += 1
                for ev in emergency_vehicles:
                    emergency_vehicles_detected.add(ev['id'])

            # Choose actions for each traffic light
            for tl, agent in agents.items():
                s = torch.FloatTensor(state[tl]).unsqueeze(0)  # Convert to tensor and add batch dimension
                with torch.no_grad():
                    logits = agent.actor(s)  # Use the actor network to get logits
                    probs = F.softmax(logits, dim=-1).squeeze(0)  # [n_phases]

                # Mask invalid transitions
                curr = env.traffic_signals[tl].green_phase
                valid = [p for p in range(probs.size(0))
                         if (curr, p) in env.traffic_signals[tl].yellow_dict]
                mask = torch.zeros_like(probs)
                mask[valid] = 1.0
                masked = probs * mask
                if masked.sum() == 0:
                    masked[valid] = 1.0 / len(valid)
                else:
                    masked /= masked.sum()

                # Choose action based on masked probabilities
                action = int(masked.argmax().item())
                actions[tl] = action

            # Override with emergency vehicle priorities if needed
            emergency_override = handle_emergency_vehicles(env, tls, siren_model, actions)

            # Update phase counts
            for tl in tls:
                phase_counts[tl][actions[tl]] += 1

            # Step environment
            obs2, reward_dict, done, _ = env.step(actions)

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

            vehs = env.sumo.vehicle.getIDList()
            if vehs:
                waiting_times.append(np.mean([traci.vehicle.getWaitingTime(v) for v in vehs]))
            else:
                waiting_times.append(0.0)

            for v in vehs:
                if traci.vehicle.getTypeID(v) == "emergency_veh" and traci.vehicle.getWaitingTime(v) > 0:
                    if v not in ev_waited:
                        ev_waited.add(v)
                        ev_waiting_times.append(v)

            # Compute shaped reward
            Q_before = total_q
            Q_after = sum(traci.lane.getLastStepHaltingNumber(l) for l in lanes)
            C_tot = max(0, Q_before - Q_after)
            phi_diff = Q_before - Q_after
            r = (
                -0.3 * Q_before  # QUEUE_PEN
                + 0.2 * C_tot    # SERVE_BONUS
                + 0.2 * C_tot    # THROUGHPUT_BONUS
                - 0.05 * (1 if emergency_override else 0)  # PHASE_CHANGE_PEN
                + 1.0 * phi_diff  # EMERGENCY_BONUS
            )
            shaped_rewards.append(r)

            print(f"Step {step:4d} | Avg Reward {avg_r: .3f} | Avg Queue {avg_queue: .1f} | Shaped Reward {r: .3f}")

            # Prepare next state
            state = {tl: prepare_obs(obs2, tl, neighbours) for tl in tls}

    finally:
        # Ensure the environment is closed properly
        try:
            env.close()
            traci.close()
        except Exception as e:
            print(f"⚠️  Error during cleanup: {e}")

    # 5) summary
    print("\n✅ Testing complete")
    print(f"Mean reward: {np.mean(rewards):.3f}")
    print(f"Mean queue : {np.mean(queues):.1f}")
    print(f"\n🚑 Emergency vehicle statistics:")
    print(f"   - Detection events: {emergency_detection_count}")
    print(f"   - Unique emergency vehicles detected: {len(emergency_vehicles_detected)}")
    
    for tl in tls:
        print(f"\nPhase counts for {tl}:")
        for p, c in phase_counts[tl].items():
            print(f"  Phase {p}: {c}")

    # === Summary ===
    print("\n✅ Testing complete")
    print(f"Mean reward: {np.mean(rewards):.3f}")
    print(f"Mean queue:  {np.mean(queue_lengths):.1f}\n")
    print(f"🚑 Emergency vehicle statistics:")
    print(f"   - Detection events: {emergency_detection_count}")
    print(f"   - Unique emergency vehicles detected: {len(emergency_vehicles_detected)}")

    for tl in tls:
        print(f"Phase counts for TL {tl}:")
        for p, cnt in phase_counts[tl].items():
            print(f"  Phase {p}: {cnt} times")

    # === Performance metrics table ===
    summary = {
        "wait time (sec)":     np.mean(waiting_times),
        "travel time (sec)":   np.mean(travel_times),
        "queue length (cars)": np.mean(queue_lengths),
        "shaped reward":       np.mean(shaped_rewards),
        "EV stopped count":    len(ev_waited),
        "EV avg wait (sec)":   np.mean(ev_waiting_times) if ev_waiting_times else 0.0
    }
    df = pd.DataFrame([summary])
    print("\nPerformance metrics")
    print(df.to_markdown(index=False, floatfmt=".3f"))

    # 6) plots
    plt.figure(figsize=(10,4))
    plt.plot(rewards, label="Avg Reward")
    plt.xlabel("Step"); plt.ylabel("Reward"); plt.title("SAC Test: Rewards")
    plt.grid(True); plt.legend()

    plt.figure(figsize=(10,4))
    plt.plot(queues, color="orange", label="Avg Queue")
    plt.xlabel("Step"); plt.ylabel("Queue"); plt.title("SAC Test: Queue Lengths")
    plt.grid(True); plt.legend()

    plt.show()

    env.close()

if __name__ == "__main__":
    main()
