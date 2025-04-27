# ppo_testing_with_metrics.py

import os
import numpy as np
import torch
import sumo_rl
import traci
from agents.ppo_agent import PPO
import matplotlib.pyplot as plt
import pandas as pd  # for performance metrics table
import random
import librosa
from tensorflow.keras.models import load_model

# === Siren Detection ===
def extract_features(audio_file, max_pad_len=862):
    try:
        audio, sr = librosa.load(audio_file, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=80)
        pad_width = max_pad_len - mfccs.shape[1]
        if pad_width > 0:
            mfccs = np.pad(mfccs,
                           pad_width=((0, 0), (0, pad_width)),
                           mode='constant')
        else:
            mfccs = mfccs[:, :max_pad_len]
        features = np.mean(mfccs, axis=1)
        return features.reshape(1, 1, 80)
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

def detect_siren():
    audio_path = "dynamic_sounds/ambulance.wav"
    if not os.path.exists(audio_path) or siren_model is None:
        return False
    features = extract_features(audio_path)
    if features is not None:
        return siren_model.predict(features)[0][0] > 0.5
    return False

# === Load Siren Model ===
try:
    siren_model = load_model('siren_model/best_model.keras')
    print("✅ Siren detection model loaded!")
except Exception as e:
    print(f"❌ Failed to load siren model: {e}")
    siren_model = None

# === SUMO Environment ===
env = sumo_rl.SumoEnvironment(
    net_file='nets/intersection/environment.net.xml',
    route_file='nets/intersection/episode_routes_high.rou.xml',
    use_gui=True,
    num_seconds=5000,
    single_agent=False
)

tl_id = env.ts_ids[0]
print(f"Testing PPO agent on traffic light: {tl_id}")

# --- reset to launch SUMO & traci ---
observations = env.reset()

# --- determine lanes served by each phase ---
phases = env.traffic_signals[tl_id].all_phases
ctrl_lanes = traci.trafficlight.getControlledLanes(tl_id)
lanes_by_phase = [
    [lane for lane, sig in zip(ctrl_lanes, ph.state) if sig.upper() == 'G']
    for ph in phases
]

num_phases = len(phases)
state_dim  = len(observations[tl_id])

# === Load PPO Agent ===
agent = PPO(state_dim, num_phases, hidden_size=64, lr=3e-4, gamma=0.99, clip_ratio=0.2, K_epoch=10)
agent.actor_critic.load_state_dict(torch.load('trained_models/model_ppo.pth'))
agent.actor_critic.eval()

# === Shaping hyperparameters (same as in training) ===
QUEUE_PEN        = 0.3    # penalize total queue
SERVE_BONUS      = 0.2    # bonus for cars cleared on chosen phase
THROUGHPUT_BONUS = 0.2    # bonus for total throughput
PHASE_CHANGE_PEN = 0.05   # penalty for switching phases
EMERGENCY_BONUS  = 1.0   # large bonus for override

# === initialize performance metric containers ===
done           = {"__all__": False}
shaped_rewards = []       # now holds shaped reward
queue_lengths  = []
phase_counts   = {p: 0 for p in range(num_phases)}
travel_times   = []
depart_times   = {}
waiting_times  = []
ev_waited      = set()
ev_waiting     = []

# === initial state ===
state = observations[tl_id]
step  = 0

# ===== Testing Loop =====
while not done["__all__"]:
    step += 1

    # — record travel times & EV waiting times at depart/arrival —
    for vid in traci.simulation.getDepartedIDList():
        depart_times[vid] = step
    for vid in traci.simulation.getArrivedIDList():
        if vid in depart_times:
            travel_times.append(step - depart_times[vid])
            try:
                if traci.vehicle.getTypeID(vid) == "emergency_veh":
                    ev_waiting.append(traci.vehicle.getWaitingTime(vid))
            except:
                pass
            del depart_times[vid]

    # — determine valid next phases —
    curr_phase    = env.traffic_signals[tl_id].green_phase
    valid         = [
        p for p in range(num_phases)
        if (curr_phase, p) in env.traffic_signals[tl_id].yellow_dict
    ]
    if not valid:
        valid = [curr_phase]

    # — detect siren & possibly override —
    override   = False
    override_road = None
    for vid in env.sumo.vehicle.getIDList():
        if env.sumo.vehicle.getTypeID(vid) == "emergency_veh":
            dist = 750 - env.sumo.vehicle.getLanePosition(vid)
            if dist < 100 and detect_siren():
                override     = True
                override_road = env.sumo.vehicle.getRoadID(vid)
                break

    if override:
        # north–south = phase 0, east–west = phase 2
        if override_road.startswith(("N2TL","S2TL")):
            action = 0
        else:
            action = 2
    else:
        raw_act = agent.choose_action(state)
        action  = raw_act if raw_act in valid else random.choice(valid)

    phase_counts[action] += 1

    # —— compute shaping terms BEFORE stepping ——
    # 1) total queue & queue per phase
    queue_per_phase  = np.array([
        sum(traci.lane.getLastStepHaltingNumber(l) for l in lanes_by_phase[p])
        for p in range(num_phases)
    ], dtype=float)
    total_q_before   = queue_per_phase.sum()
    cleared_sel      = queue_per_phase[action]
    change_penalty   = PHASE_CHANGE_PEN if action != curr_phase else 0.0
    em_flag          = 1.0 if override else 0.0

    # — step environment —
    observations, reward_dict, done, _ = env.step({tl_id: action})
    base_r = reward_dict[tl_id]

    # — recompute after-step queues —
    queue_after      = np.array([
        sum(traci.lane.getLastStepHaltingNumber(l) for l in lanes_by_phase[p])
        for p in range(num_phases)
    ], dtype=float)
    total_q_after    = queue_after.sum()
    cleared_selected = max(0.0, cleared_sel - queue_after[action])
    cleared_total    = max(0.0, total_q_before - total_q_after)

    # 2) potential-based shaping
    pot_diff = agent.gamma * (-total_q_after) - (-total_q_before)

    # — assemble shaped reward —
    shaped_r = (
        base_r
        - QUEUE_PEN        * total_q_before
        + SERVE_BONUS      * cleared_selected
        + THROUGHPUT_BONUS * cleared_total
        + pot_diff
        - change_penalty
        + EMERGENCY_BONUS  * em_flag
    )

    shaped_rewards.append(shaped_r)

    # — other metrics —
    total_queue = sum(
        traci.lane.getLastStepHaltingNumber(l)
        for l in traci.trafficlight.getControlledLanes(tl_id)
    )
    queue_lengths.append(total_queue)

    vehs = env.sumo.vehicle.getIDList()
    if vehs:
        waiting_times.append(np.mean([traci.vehicle.getWaitingTime(v) for v in vehs]))
    else:
        waiting_times.append(0.0)

    for v in vehs:
        if traci.vehicle.getTypeID(v) == "emergency_veh" and traci.vehicle.getWaitingTime(v) > 0:
            if v not in ev_waited:
                ev_waited.add(v)
                ev_waiting.append(traci.vehicle.getWaitingTime(v))

    # — advance state for next step —
    state = observations[tl_id]

    print(f"Step {step}: shaped_r={shaped_r:.2f}, phase={action}, total_queue={total_queue:.1f}")

# ===== Summary =====
print("\n✅ Evaluation completed.")
print(f"Average shaped reward: {np.mean(shaped_rewards):.2f}\n")
print("Phase usage:")
for p, cnt in phase_counts.items():
    print(f"  Phase {p}: {cnt} times")

summary = {
    "wait time (sec)":     np.mean(waiting_times),
    "travel time (sec)":   np.mean(travel_times),
    "queue length (cars)": np.mean(queue_lengths),
    "reward":              np.mean(shaped_rewards),  # now shaped
    "EV stopped count":    len(ev_waited),
    "EV avg wait (sec)":   np.mean(ev_waiting) if ev_waiting else 0.0
}
df_summary = pd.DataFrame([summary])
print("\nPerformance metrics")
print(df_summary.to_markdown(index=False, floatfmt=".3f"))

# ===== Plots =====
plt.figure(figsize=(10,4))
plt.plot(shaped_rewards, label='Reward', color='tab:blue')
plt.xlabel('Step'); plt.ylabel('Reward')
plt.title('PPO Evaluation: Rewards Over Time')
plt.grid(True); plt.legend()
plt.tight_layout()

plt.figure(figsize=(10,4))
plt.plot(queue_lengths, label='Queue Length', color='orange')
plt.xlabel('Step'); plt.ylabel('Queue')
plt.title('Queue Length Over Time')
plt.grid(True); plt.legend()
plt.tight_layout()

plt.show()

env.close()
