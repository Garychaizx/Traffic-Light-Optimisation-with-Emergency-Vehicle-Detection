# sac_testing_with_shaped_reward.py

import os
import random
import numpy as np
import torch
import sumo_rl
import traci
from agents.sac_agent import SACAgent
import matplotlib.pyplot as plt
import pandas as pd
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
    path = "dynamic_sounds/ambulance.wav"
    if not os.path.exists(path) or siren_model is None:
        return False
    feats = extract_features(path)
    if feats is None:
        return False
    return float(siren_model.predict(feats)[0][0]) > 0.5

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
    route_file='nets/intersection/episode_routes_low.rou.xml',
    use_gui=True,
    num_seconds=5000,
    single_agent=False
)

tl_id = env.ts_ids[0]
print(f"Testing SAC agent on traffic light: {tl_id}")

# --- reset & connect TraCI ---
observations = env.reset()

# --- get lanes per phase ---
phases         = env.traffic_signals[tl_id].all_phases
ctrl_lanes     = traci.trafficlight.getControlledLanes(tl_id)
lanes_by_phase = [
    [lane for lane, sig in zip(ctrl_lanes, ph.state) if sig.upper()=='G']
    for ph in phases
]

num_phases = len(phases)
state_dim  = len(observations[tl_id])

# === Load SAC Agent ===
agent = SACAgent(state_dim=state_dim, action_dim=num_phases)
agent.actor.load_state_dict(torch.load('trained_models/sac_actor.pth'))
agent.actor.eval()

# === Shaping hyperparameters ===
QUEUE_PEN        = 0.3    # penalize total queue
SERVE_BONUS      = 0.2    # bonus for cars cleared on chosen phase
THROUGHPUT_BONUS = 0.2    # bonus for total throughput
PHASE_CHANGE_PEN = 0.05   # penalty for switching phases
EMERGENCY_BONUS  = 1.0   # bonus for emergency override

# === Metrics containers ===
done            = {"__all__": False}
shaped_rewards  = []
queue_lengths   = []
phase_counts    = {p: 0 for p in range(num_phases)}
travel_times    = []
depart_times    = {}
waiting_times   = []
ev_waited       = set()
ev_waiting      = []

# === Initial state ===
state = observations[tl_id]
step  = 0

# ===== Evaluation Loop =====
while not done["__all__"]:
    step += 1

    # --- record departures & arrivals ---
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

    # --- choose action with valid yellow transitions ---
    curr_phase   = env.traffic_signals[tl_id].green_phase
    valid_phases = [
        p for p in range(num_phases)
        if (curr_phase, p) in env.traffic_signals[tl_id].yellow_dict
    ]
    if not valid_phases:
        valid_phases = [curr_phase]

    # --- emergency override? ---
    override      = False
    override_road = None
    for vid in env.sumo.vehicle.getIDList():
        if env.sumo.vehicle.getTypeID(vid) == "emergency_veh":
            dist = 750 - env.sumo.vehicle.getLanePosition(vid)
            if dist < 100 and detect_siren():
                override      = True
                override_road = env.sumo.vehicle.getRoadID(vid)
                break

    if override:
        action = 0 if override_road.startswith(("N2TL","S2TL")) else 2
    else:
        with torch.no_grad():
            st = torch.FloatTensor(state).unsqueeze(0)
            probs = agent.actor(st)
            action = torch.argmax(probs, dim=1).item()
        if action not in valid_phases:
            action = random.choice(valid_phases)

    phase_counts[action] += 1

    # --- compute shaping terms BEFORE stepping ---
    queue_per_phase = np.array([
        sum(traci.lane.getLastStepHaltingNumber(l) for l in lanes_by_phase[p])
        for p in range(num_phases)
    ], dtype=float)
    total_q_before   = queue_per_phase.sum()
    before_sel       = queue_per_phase[action]
    change_pen       = PHASE_CHANGE_PEN if action != curr_phase else 0.0
    ev_flag          = 1.0 if override else 0.0

    # --- step environment ---
    obs2, rdict, done, _ = env.step({tl_id: action})
    base_r = rdict[tl_id]

    # --- recompute after‐step queues ---
    queue_after    = np.array([
        sum(traci.lane.getLastStepHaltingNumber(l) for l in lanes_by_phase[p])
        for p in range(num_phases)
    ], dtype=float)
    total_q_after  = queue_after.sum()
    cleared_sel    = max(0.0, before_sel - queue_after[action])
    cleared_total  = max(0.0, total_q_before - total_q_after)

    # --- potential‐based shaping ---
    gamma   = getattr(agent, 'gamma', 0.99)
    pot_diff = gamma * (-total_q_after) - (-total_q_before)

    # --- assemble shaped reward ---
    shaped_r = (
        base_r
        - QUEUE_PEN        * total_q_before
        + SERVE_BONUS      * cleared_sel
        + THROUGHPUT_BONUS * cleared_total
        + pot_diff
        - change_pen
        + EMERGENCY_BONUS  * ev_flag
    )

    shaped_rewards.append(shaped_r)

    # --- other metrics this step ---
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

    state = obs2[tl_id]
    print(f"Step {step}: shaped_r={shaped_r:.2f}, phase={action}, queue={total_queue:.1f}")

# ===== Summary =====
print("\n✅ SAC Evaluation completed.")
print(f"Average shaped reward: {np.mean(shaped_rewards):.2f}\n")
print("Phase usage:")
for p, cnt in phase_counts.items():
    print(f"  Phase {p}: {cnt} times")

summary = {
    "wait time (sec)":     np.mean(waiting_times),
    "travel time (sec)":   np.mean(travel_times),
    "queue length (cars)": np.mean(queue_lengths),
    "reward":              np.mean(shaped_rewards),
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
plt.title('SAC Evaluation: Shaped Rewards Over Time')
plt.grid(True); plt.legend(); plt.tight_layout()

plt.figure(figsize=(10,4))
plt.plot(queue_lengths, label='Queue Length', color='orange')
plt.xlabel('Step'); plt.ylabel('Queue Length')
plt.title('Queue Length Over Time')
plt.grid(True); plt.legend(); plt.tight_layout()

plt.show()

env.close()
