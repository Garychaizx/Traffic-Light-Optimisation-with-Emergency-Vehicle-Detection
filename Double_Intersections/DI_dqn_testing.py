# test_dqn_double.py

import os
import sys
import numpy as np
import torch
import sumo_rl
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from agents.dqn_agent import DQNAgent
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
    except:
        return None

def detect_siren(model):
    path = "dynamic_sounds/ambulance.wav"
    if model is None or not os.path.exists(path):
        print("❌ Siren model not loaded or audio file missing")
        return False
    feats = extract_features(path)
    if feats is None:
        print("❌ Failed to extract features from audio")
        return False
    prediction = float(model.predict(feats)[0][0])
    return prediction > 0.5

def build_neighbours(env):
    """
    After env.reset(), build a map from each TL id to its 2‑char neighbours
    (edges like "AB", "BC", etc.).
    """
    ts = env.ts_ids
    neigh = {tl: [] for tl in ts}
    for eid in env.sumo.edge.getIDList():
        if len(eid) == 2 and eid[0] in ts and eid[1] in ts:
            a, b = eid[0], eid[1]
            neigh[a].append(b)
            neigh[b].append(a)
    # dedupe
    return {tl: list(set(lst)) for tl, lst in neigh.items()}

def prepare_obs(obs, tl, neighbours, pad_len=165):
    """
    Stack own observation + 0.2× each neighbour’s,
    then pad (with -1) or truncate to pad_len.
    Returns a torch.FloatTensor.
    """
    x = obs[tl].copy()
    for n in neighbours.get(tl, []):
        x += 0.2 * obs[n]
    if len(x) < pad_len:
        x = np.concatenate([x, np.full(pad_len - len(x), -1.0)])
    else:
        x = x[:pad_len]
    return torch.FloatTensor(x)

def main():
    # 1) Load the siren model if it exists
    try:
        siren_model = load_model('siren_model/best_model.keras')
        print("✅ Siren model loaded")
    except:
        siren_model = None
        print("⚠️  No siren model, skipping override")

    # 2) Create SUMO env with GUI
    env = sumo_rl.SumoEnvironment(
        net_file    = "nets/double/network.net.xml",
        route_file  = "nets/double/doubleRoutes.rou.xml",
        use_gui     = True,
        num_seconds = 5000,
        single_agent=False
    )
    # initialize traci
    obs = env.reset()
    tls  = env.ts_ids
    neighbours = build_neighbours(env)

    # 3) Load trained DQNAgent for each TL
    agents = {}
    for tl in tls:
        n_phases = len(env.traffic_signals[tl].all_phases)
        agent = DQNAgent(state_dim=165, action_dim=n_phases)
        ckpt = f"trained_models/dqn_double_{tl}.pth"
        agent.load(ckpt)
        agent.epsilon = 0.0   # fully greedy
        agents[tl] = agent

    # 4) Run one test episode
    state = {tl: prepare_obs(obs, tl, neighbours) for tl in tls}
    done  = {"__all__": False}

    rewards = []
    queues  = []
    phase_counts = {
        tl: {p: 0 for p in range(len(env.traffic_signals[tl].all_phases))}
        for tl in tls
    }

    step = 0
    while not done["__all__"]:
        step += 1
        actions = {}

        # pick greedy valid action per TL with emergency override
        for tl, agent in agents.items():
            curr = env.traffic_signals[tl].green_phase
            valid = [
                p for p in range(agent.action_dim)
                if (curr, p) in env.traffic_signals[tl].yellow_dict
            ]

            # Emergency vehicle detection logic
            override = False
            road = None
            for vid in env.sumo.vehicle.getIDList():
                vehicle_type = env.sumo.vehicle.getTypeID(vid)
                route = env.sumo.vehicle.getRoute(vid)
                position = env.sumo.vehicle.getLanePosition(vid)
                print(f"Vehicle ID: {vid}, Type: {vehicle_type}, Route: {route}, Position: {position}")

                if vehicle_type == "emergency":
                    # dist = 750 - position  # Adjust this based on your simulation setup
                    # if dist < 100 and detect_siren(siren_model):
                    if detect_siren(siren_model):
                        print(f"🚑 Emergency vehicle detected on route {route}")
                        override = True
                        road = route[0]  # Get the first edge of the route
                        break

            if override:
                # Force a specific phase based on the emergency vehicle's direction
                action = 0 if road.startswith(("N2TL", "S2TL")) else 2
            else:
                action = agent.choose_action(state[tl], valid)

            phase_counts[tl][action] += 1
            actions[tl] = action

        # step SUMO
        obs2, reward_dict, done, _ = env.step(actions)

        # record average reward
        avg_r = np.mean(list(reward_dict.values()))
        rewards.append(avg_r)

        # record average queue length
        total_q = 0
        for tl in tls:
            lanes = env.sumo.trafficlight.getControlledLanes(tl)
            total_q += sum(env.sumo.lane.getLastStepHaltingNumber(l) for l in lanes)
        queues.append(total_q / len(tls))

        print(f"Step {step:4d} | Avg Reward {avg_r: .3f} | Avg Queue {queues[-1]:.1f}")

        # prepare next state
        state = {tl: prepare_obs(obs2, tl, neighbours) for tl in tls}

    # 5) Summary
    print("\n✅ Testing complete")
    print(f"Mean reward over episode: {np.mean(rewards):.3f}")
    print(f"Mean queue   over episode: {np.mean(queues):.1f}")
    for tl in tls:
        print(f"\nPhase counts for {tl}:")
        for p, c in phase_counts[tl].items():
            print(f"  Phase {p}: {c}")

    # 6) Plot results
    plt.figure(figsize=(10,4))
    plt.plot(rewards, label="Average Reward")
    plt.xlabel("Step"); plt.ylabel("Reward")
    plt.title("DQN Test: Rewards over Time")
    plt.grid(True); plt.legend()

    plt.figure(figsize=(10,4))
    plt.plot(queues, color="orange", label="Average Queue")
    plt.xlabel("Step"); plt.ylabel("Queue Length")
    plt.title("DQN Test: Queue Length over Time")
    plt.grid(True); plt.legend()

    plt.show()

    env.close()

if __name__ == "__main__":
    main()