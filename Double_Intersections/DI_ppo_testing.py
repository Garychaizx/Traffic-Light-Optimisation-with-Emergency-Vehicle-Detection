# test_double_intersection_ppo.py

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import sumo_rl
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from agents.ppo_agent import PPO
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
        print(f"🔊 Siren detection confidence: {prediction}")
        return prediction > 0.5
    except Exception as e:
        print(f"❌ Error during siren detection: {e}")
        return False

def build_neighbours(env):
    """
    Build a map of neighbouring traffic lights by looking
    for two-letter edge IDs linking TLs.
    """
    ts = env.ts_ids
    neigh = {tl: [] for tl in ts}
    for edge_id in env.sumo.edge.getIDList():
        if len(edge_id) == 2 and edge_id[0] in ts and edge_id[1] in ts:
            a, b = edge_id[0], edge_id[1]
            neigh[a].append(b)
            neigh[b].append(a)
    return {tl: np.array(v) for tl, v in neigh.items()}

def prepare_obs(obs, tl, neighbours, pad_len=181):
    """
    Concatenate own obs plus 0.2× neighbours’ obs, then pad/truncate.
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
    except Exception as e:
        siren_model = None
        print(f"⚠️  No siren model, skipping override: {e}")

    # 2) Create the SUMO environment (GUI on for visualization)
    env = sumo_rl.SumoEnvironment(
        net_file    = "nets/double/network.net.xml",
        route_file  = "nets/double/doubleRoutes.rou.xml",
        use_gui     = True,
        num_seconds = 5000,
        single_agent=False
    )

    # 3) Initialize and build neighbour map
    obs = env.reset()
    neighbours = build_neighbours(env)

    # 4) Load each TL’s trained PPO agent
    agents = {}
    for tl in env.ts_ids:
        n_phases = len(env.traffic_signals[tl].all_phases)
        agent = PPO(
            state_dim   = 181,
            action_dim  = n_phases,
            hidden_size = 64,
            lr          = 3e-4,
            gamma       = 0.99,
            clip_ratio  = 0.2,
            K_epoch     = 4
        )
        ckpt = f"trained_models/ppo_double_{tl}.pth"
        agent.actor_critic.load_state_dict(torch.load(ckpt))
        agent.actor_critic.eval()
        agents[tl] = agent

    # 5) Run one test episode
    obs = env.reset()
    state = {tl: prepare_obs(obs, tl, neighbours) for tl in env.ts_ids}

    done = {"__all__": False}
    rewards = []
    queues  = []
    phase_counts = {
        tl: {p: 0 for p in range(len(env.traffic_signals[tl].all_phases))}
        for tl in env.ts_ids
    }

    step = 0
    while not done["__all__"] and step < 5000:
        step += 1
        actions = {}

        # Choose a greedy, valid phase for each TL with emergency override
        for tl, agent in agents.items():
            s = state[tl].unsqueeze(0)  # [1,165]
            with torch.no_grad():
                logits, _ = agent.actor_critic(s)
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
                action = int(masked.argmax().item())
                print(f"🤖 Chosen action: {action}")

            phase_counts[tl][action] += 1
            actions[tl] = action

        # Step environment
        obs2, reward_dict, done, _ = env.step(actions)

        # Record metrics
        avg_r = np.mean(list(reward_dict.values()))
        rewards.append(avg_r)

        # Average queue over all TLs
        total_q = 0
        for tl in env.ts_ids:
            lanes = env.sumo.trafficlight.getControlledLanes(tl)
            total_q += sum(env.sumo.lane.getLastStepHaltingNumber(l) for l in lanes)
        queues.append(total_q / len(env.ts_ids))

        print(f"Step {step:4d} | Avg Reward {avg_r: .3f} | Avg Queue {queues[-1]: .1f}")

        # Prepare next state
        state = {tl: prepare_obs(obs2, tl, neighbours) for tl in env.ts_ids}

    # 6) Summary
    print("\n✅ Testing done")
    print(f"Mean reward over episode: {np.mean(rewards):.3f}")
    print(f"Mean queue   over episode: {np.mean(queues):.1f}")
    for tl in env.ts_ids:
        print(f"\nPhase counts for {tl}:")
        for p, c in phase_counts[tl].items():
            print(f"  Phase {p}: {c}")

    # 7) Plots
    plt.figure(figsize=(10,4))
    plt.plot(rewards, label="Avg Reward")
    plt.xlabel("Step"); plt.ylabel("Reward"); plt.title("PPO Testing: Rewards")
    plt.grid(True); plt.legend()

    plt.figure(figsize=(10,4))
    plt.plot(queues, color="orange", label="Avg Queue")
    plt.xlabel("Step"); plt.ylabel("Queue"); plt.title("PPO Testing: Queue Lengths")
    plt.grid(True); plt.legend()

    plt.show()

    env.close()

if __name__ == "__main__":
    main()