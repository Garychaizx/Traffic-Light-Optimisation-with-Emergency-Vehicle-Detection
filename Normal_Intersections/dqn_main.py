# train_dqn_intersection_adaptive_fixed2.py

import os
import random
import numpy as np
import torch
import sumo_rl
import traci
from agents.dqn_agent import DQNAgent
import matplotlib.pyplot as plt

# === optional siren override machinery ===
import librosa
from tensorflow.keras.models import load_model

def extract_features(audio_file, max_pad_len=862):
    try:
        audio, sr = librosa.load(audio_file, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=80)
        pad_width = max_pad_len - mfccs.shape[1]
        if pad_width > 0:
            mfccs = np.pad(mfccs,
                           pad_width=((0,0),(0,pad_width)),
                           mode='constant')
        else:
            mfccs = mfccs[:, :max_pad_len]
        return np.mean(mfccs, axis=1).reshape(1,1,80)
    except:
        return None

def detect_siren(siren_model):
    path = "dynamic_sounds/ambulance.wav"
    if not os.path.exists(path) or siren_model is None:
        return False
    feats = extract_features(path)
    if feats is None:
        return False
    return float(siren_model.predict(feats)[0][0]) > 0.5

def main():
    # 1) load siren model if it exists
    try:
        siren_model = load_model('siren_model/best_model.keras')
        print("✅ Siren model loaded")
    except:
        siren_model = None
        print("⚠️  No siren model, skipping override")

    # 2) build SUMO-RL environment (this DOES NOT connect TraCI yet)
    env = sumo_rl.SumoEnvironment(
        net_file    = 'nets/intersection/environment.net.xml',
        route_file  = 'nets/intersection/episode_routes.rou.xml',
        use_gui     = True,
        num_seconds = 5000,
        single_agent=False
    )
    tl_id      = env.ts_ids[0]
    phases     = env.traffic_signals[tl_id].all_phases
    num_phases = len(phases)

    # 3) now reset once so SUMO is launched and TraCI is connected
    obs0     = env.reset()

    # 4) only now can we call traci to ask which lanes each signal controls
    ctrl_lanes = traci.trafficlight.getControlledLanes(tl_id)
    lanes_by_phase = []
    for ph in phases:
        serve = [
            lane
            for lane, sig in zip(ctrl_lanes, ph.state)
            if sig.upper() == 'G'
        ]
        lanes_by_phase.append(serve)

    # 5) determine your state dimensionality
    base_dim   = len(obs0[tl_id])
    state_dim  = base_dim + num_phases

    # 6) create your DQN agent
    agent = DQNAgent(
        state_dim     = state_dim,
        action_dim    = num_phases,
        gamma         = 0.99,
        lr            = 1e-3,
        epsilon       = 1.0,
        min_epsilon   = 0.05,
        epsilon_decay = 1e-5,
        batch_size    = 64,
        memory_size   = 20000
    )

    # 7) hyperparams for shaping
    MAX_STEPS   = 5000
    QUEUE_PEN   = 0.3   # penalize the largest queue anywhere
    SERVE_BONUS = 0.2   # bonus for number of cars cleared on chosen phase

    # Define shaping weights
    QUEUE_PEN = 0.3          # Penalize the largest queue
    SERVE_BONUS = 0.2        # Bonus for clearing vehicles
    PHASE_CHANGE_PEN = 0.05  # Penalize phase changes
    EMERGENCY_BONUS = 1.0    # Bonus for emergency vehicle handling
    GAMMA = 0.99             # Discount factor for potential-based shaping

    # Initialize variables for shaping
    prev_total_queue = 0

    obs     = obs0
    done    = {"__all__": False}
    rewards = []
    step    = 0

    # 8) main training loop
    while not done["__all__"] and step < MAX_STEPS:
        step += 1

        # –– build state: raw obs + per‐phase normalized queue lengths ––
        raw = obs[tl_id]
        queue_per_phase = np.array([
            sum(traci.lane.getLastStepHaltingNumber(l) for l in lanes_by_phase[p])
            for p in range(num_phases)
        ], dtype=float)
        queue_feat = queue_per_phase / (queue_per_phase.max() + 1e-3)
        state = np.concatenate([raw, queue_feat])

        # –– pick a valid next phase (with optional siren override) ––
        curr = env.traffic_signals[tl_id].green_phase
        valid = [
            p for p in range(num_phases)
            if (curr, p) in env.traffic_signals[tl_id].yellow_dict
        ]
        if not valid:
            valid = [curr]

        emergency_override = detect_siren(siren_model)
        if emergency_override:
            # Force north–south straight if siren
            action = 0 if curr in (0, 1) else curr
        else:
            action = agent.choose_action(state, valid)

        # measure how many are waiting on that phase
        before  = queue_per_phase[action]

        # –– step SUMO one tick ––
        obs2, rdict, done, _ = env.step({tl_id: action})
        base_r = rdict[tl_id]

        # recompute queue just on the served lanes
        after   = sum(
            traci.lane.getLastStepHaltingNumber(l)
            for l in lanes_by_phase[action]
        )
        cleared = max(0.0, before - after)

        # === Shaped Reward Calculation ===
        # 1. Queue penalty
        total_queue = queue_per_phase.sum()
        queue_penalty = -QUEUE_PEN * queue_per_phase.max()

        # 2. Bonus for clearing vehicles
        clear_bonus = SERVE_BONUS * cleared

        # 3. Phase change penalty
        phase_change_penalty = -PHASE_CHANGE_PEN if action != curr else 0.0

        # 4. Emergency vehicle bonus
        emergency_bonus = EMERGENCY_BONUS if emergency_override else 0.0

        # 5. Potential-based shaping
        potential_diff = GAMMA * (-total_queue) - (-prev_total_queue)

        # Combine all components into the shaped reward
        shaped_r = (
            base_r
            + queue_penalty
            + clear_bonus
            + phase_change_penalty
            + emergency_bonus
            + potential_diff
        )

        # Update previous total queue
        prev_total_queue = total_queue

        # –– store & train ––
        raw2 = obs2[tl_id]
        queue2 = np.array([
            sum(traci.lane.getLastStepHaltingNumber(l) for l in lanes_by_phase[p])
            for p in range(num_phases)
        ], dtype=float)
        feat2 = queue2 / (queue2.max() + 1e-3)
        state2 = np.concatenate([raw2, feat2])

        agent.store_experience(state, action, shaped_r, state2, float(done["__all__"]))
        agent.update_model()

        obs     = obs2
        rewards.append(shaped_r)

        if step % 100 == 0:
            print(
                f"Step {step:4d} | base_r {base_r:.2f} "
                f"| shaped_r {shaped_r:.2f} "
                f"| maxQ {queue_per_phase.max():.1f} "
                f"| cleared {cleared:.1f} "
                f"| ε {agent.epsilon:.3f}"
            )

    # 9) save & plot
    os.makedirs("trained_models", exist_ok=True)
    agent.save("trained_models/model_dqn.pth")
    print("✅ Done. Saved to trained_models/model_dqn.pth")

    plt.figure(figsize=(8,4))
    plt.plot(rewards)
    plt.xlabel("Step")
    plt.ylabel("Reward")
    plt.title("DQN Training (Reward)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    env.close()

if __name__ == "__main__":
    main()
