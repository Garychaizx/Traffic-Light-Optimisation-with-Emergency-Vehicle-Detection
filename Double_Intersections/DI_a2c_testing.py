# test_a2c_double.py

import os
import re
import sys
import numpy as np
import torch
import sumo_rl
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from agents.a2c_agent import A2CAgent
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
        print(f"🔊 Siren detection confidence: {prediction}")
        return prediction > 0.5
    except Exception as e:
        print(f"❌ Error during siren detection: {e}")
        return False
    
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
    return {tl: list(set(lst)) for tl, lst in neigh.items()}

def prepare_obs(obs, tl, neighbours, pad_len=165):
    """
    Stack own observation + 0.2× each neighbour’s, then pad (with -1)
    or truncate to pad_len.
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

    try:
        siren_model = load_model('siren_model/best_model.keras')
        print("✅ Siren model loaded")
    except Exception as e:
        siren_model = None
        print(f"⚠️  No siren model, skipping override: {e}")

    # 1) Create SUMO env in GUI mode for visualization
    env = sumo_rl.SumoEnvironment(
        net_file    = "nets/double/network.net.xml",
        route_file  = "nets/double/doubleRoutes.rou.xml",
        use_gui     = True,
        num_seconds = 5000,
        single_agent=False
    )

    # initialize and build neighbours
    obs = env.reset()
    tls  = env.ts_ids
    neighbours = build_neighbours(env)

    # 2) Load each TL’s trained A2C agent
    agents = {}
    for tl in tls:
        n_phases = len(env.traffic_signals[tl].all_phases)
        agent = A2CAgent(state_dim=165, action_dim=n_phases)
        ckpt = torch.load(f"trained_models/a2c_double_{tl}.pth")
        agent.actor.load_state_dict(ckpt['actor'])
        agent.critic.load_state_dict(ckpt['critic'])
        agent.actor.eval()
        agent.critic.eval()
        agents[tl] = agent

    # 3) Run one test episode
    obs   = env.reset()
    state = {tl: prepare_obs(obs, tl, neighbours) for tl in tls}

    done = {"__all__": False}
    rewards = []
    queues  = []
    phase_counts = {tl: {p: 0 for p in range(len(env.traffic_signals[tl].all_phases))}
                    for tl in tls}
    step = 0
    MAX_STEPS = 5000

    while not done["__all__"] and step < MAX_STEPS:
        step += 1
        actions = {}

        # for each TL, pick the highest‑prob valid phase
        for tl, agent in agents.items():
            logits = agent.actor(state[tl].unsqueeze(0)).squeeze(0)
            probs  = logits.softmax(dim=-1)

            # mask illegal transitions
            curr  = env.traffic_signals[tl].green_phase
            valid = [p for p in range(probs.size(0))
                     if (curr, p) in env.traffic_signals[tl].yellow_dict]
            mask  = torch.zeros_like(probs)
            mask[valid] = 1.0
            masked = probs * mask
            if masked.sum() == 0:
                # fallback uniform on valid
                masked[valid] = 1.0 / len(valid)
            else:
                masked /= masked.sum()

           # Emergency vehicle detection logic
            override = False
            road = None
            for vid in env.sumo.vehicle.getIDList():
                vehicle_type = env.sumo.vehicle.getTypeID(vid)
                route = env.sumo.vehicle.getRoute(vid)
                position = env.sumo.vehicle.getLanePosition(vid)
                print(f"Vehicle ID: {vid}, Type: {vehicle_type}, Route: {route}, Position: {position}")

                if vehicle_type == "emergency":
                    if detect_siren(siren_model):
                        print(f"🚑 Emergency vehicle detected on route {route}")
                        override = True
                        road = route[0]  # Get the first edge of the route
                        break

            if override:
                # Force a specific phase based on the emergency vehicle's direction
                action = 0 if road.startswith(("N2TL", "S2TL")) else 2
                print(f"🚦 Forcing phase {action} for road {road}")
            else:
                # deterministic: pick argmax
                action = int(masked.argmax().item())
                print(f"🤖 Chosen action: {action}")

            phase_counts[tl][action] += 1
            actions[tl] = action

        # step environment
        obs2, reward_dict, done, _ = env.step(actions)

        # metrics
        avg_r = np.mean(list(reward_dict.values()))
        rewards.append(avg_r)

        # average queue length across all TLs
        total_q = 0
        for tl in tls:
            lanes = env.sumo.trafficlight.getControlledLanes(tl)
            total_q += sum(env.sumo.lane.getLastStepHaltingNumber(l) for l in lanes)
        queues.append(total_q / len(tls))

        print(f"Step {step:4d} | AvgReward {avg_r: .3f} | AvgQueue {queues[-1]: .1f}")

        # prepare next state
        state = {tl: prepare_obs(obs2, tl, neighbours) for tl in tls}

    # 4) Summary
    print("\n✅ Testing complete")
    print(f"Mean reward: {np.mean(rewards):.3f}")
    print(f"Mean queue:  {np.mean(queues):.1f}\n")
    for tl in tls:
        print(f"Phase counts for TL {tl}:")
        for p, cnt in phase_counts[tl].items():
            print(f"  Phase {p}: {cnt} times")

    # 5) Plots
    plt.figure(figsize=(10,4))
    plt.plot(rewards, label="Avg Reward")
    plt.xlabel("Step"); plt.ylabel("Reward"); plt.title("A2C: Reward per Step")
    plt.grid(); plt.legend()

    plt.figure(figsize=(10,4))
    plt.plot(queues, color="orange", label="Avg Queue")
    plt.xlabel("Step"); plt.ylabel("Queue"); plt.title("A2C: Queue Length per Step")
    plt.grid(); plt.legend()

    plt.show()

    env.close()

if __name__ == "__main__":
    main()
