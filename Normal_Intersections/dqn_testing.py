# dqn_testing_fixed_with_metrics.py

import os
import numpy as np
import torch
import sumo_rl
import traci
from agents.dqn_agent import DQNAgent
import matplotlib.pyplot as plt
import pandas as pd  # for performance metrics table
from traci.exceptions import TraCIException
# === siren detection machinery ===
import librosa
from tensorflow.keras.models import load_model

def extract_features(audio_file, max_pad_len=862):
    try:
        audio, sr = librosa.load(audio_file, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=80)
        pad = max_pad_len - mfccs.shape[1]
        if pad > 0:
            mfccs = np.pad(mfccs, ((0,0),(0,pad)), mode='constant')
        else:
            mfccs = mfccs[:, :max_pad_len]
        return np.mean(mfccs, axis=1).reshape(1,1,80)
    except:
        return None

def detect_siren(model):
    path = "dynamic_sounds/ambulance.wav"
    if model is None or not os.path.exists(path):
        return False
    feats = extract_features(path)
    if feats is None:
        return False
    return float(model.predict(feats)[0][0]) > 0.5

# === load siren model if present ===
try:
    siren_model = load_model('siren_model/best_model.keras')
    print("✅ Siren model loaded")
except:
    siren_model = None
    print("⚠️  No siren model found, skipping siren override")

# === build SUMO env (no traci connection yet) ===
env = sumo_rl.SumoEnvironment(
    net_file    = 'nets/intersection/environment.net.xml',
    route_file  = 'nets/intersection/episode_routes_high.rou.xml',
    use_gui     = True,
    num_seconds = 5000,
    single_agent=False
)
tl_id      = env.ts_ids[0]
phases     = env.traffic_signals[tl_id].all_phases
num_phases = len(phases)

# === reset once to start SUMO & traci ===
obs0 = env.reset()

# === now safe to call traci ===
ctrl_lanes = traci.trafficlight.getControlledLanes(tl_id)
lanes_by_phase = []
for ph in phases:
    serve = [
        lane for lane, sig in zip(ctrl_lanes, ph.state)
        if sig.upper() == 'G'
    ]
    lanes_by_phase.append(serve)

# === recreate the same state dim used in training ===
base_dim  = len(obs0[tl_id])
state_dim = base_dim + num_phases

# === instantiate your agent and load weights ===
agent = DQNAgent(state_dim, num_phases)
agent.load('trained_models/model_dqn.pth')
agent.epsilon = 0.0   # full greedy for evaluation

# === initialize performance metric containers ===
rewards           = []
queue_lengths     = []
phase_counts      = {p:0 for p in range(num_phases)}
done              = {"__all__": False}
step              = 0

# new metric lists
travel_times      = []
depart_times      = {}
waiting_times     = []
ev_waited         = set()
ev_waiting_times  = []

# Define shaping weights
QUEUE_PENALTY = 0.3          # Penalize the largest queue
CLEAR_BONUS = 0.2            # Bonus for clearing vehicles
PHASE_CHANGE_PENALTY = 0.05  # Penalize phase changes
EMERGENCY_BONUS = 1.0        # Bonus for emergency vehicle handling
GAMMA = 0.99                 # Discount factor for potential-based shaping

# Initialize variables for shaping
prev_total_queue = 0

# === initial state ===
state_dict  = obs0
state_raw   = state_dict[tl_id]
queues0     = np.array([
    sum(traci.lane.getLastStepHaltingNumber(l) for l in lanes_by_phase[p])
    for p in range(num_phases)
], dtype=float)
queue_feat0 = queues0 / (queues0.max() + 1e-3)
state       = np.concatenate([state_raw, queue_feat0])

# ===== Evaluation Loop =====
while not done["__all__"]:
    step += 1

    # --- Record departures & arrivals for travel time & EV wait capture ---
    for vid in traci.simulation.getDepartedIDList():
        depart_times[vid] = step
    for vid in traci.simulation.getArrivedIDList():
        if vid in depart_times:
            travel_times.append(step - depart_times[vid])
            try:
                if traci.vehicle.getTypeID(vid) == "emergency_veh":
                    ev_waiting_times.append(traci.vehicle.getWaitingTime(vid))
            except TraCIException:
                pass
            del depart_times[vid]

    # Figure out valid yellow transitions
    curr_phase = env.traffic_signals[tl_id].green_phase
    valid = [
        p for p in range(num_phases)
        if (curr_phase, p) in env.traffic_signals[tl_id].yellow_dict
    ]
    if not valid:
        valid = [curr_phase]

    # Check for emergency override
    override = False
    road = None
    for vid in env.sumo.vehicle.getIDList():
        if env.sumo.vehicle.getTypeID(vid) == "emergency_veh":
            dist = 750 - env.sumo.vehicle.getLanePosition(vid)
            if dist < 100 and detect_siren(siren_model):
                override = True
                road = env.sumo.vehicle.getRoadID(vid)
                break

    if override:
        action = 0 if road.startswith(("N2TL", "S2TL")) else 2
    else:
        action = agent.choose_action(state, valid)

    phase_counts[action] += 1

    # Measure queue before the action
    before_queue = sum(
        traci.lane.getLastStepHaltingNumber(l)
        for l in traci.trafficlight.getControlledLanes(tl_id)
    )

    # Step the environment
    obs2, rdict, done, _ = env.step({tl_id: action})
    base_reward = rdict[tl_id]

    # Measure queue after the action
    after_queue = sum(
        traci.lane.getLastStepHaltingNumber(l)
        for l in traci.trafficlight.getControlledLanes(tl_id)
    )
    cleared_vehicles = max(0, before_queue - after_queue)

    # === Shaped Reward Calculation ===
    # 1. Queue penalty
    queue_penalty = -QUEUE_PENALTY * before_queue

    # 2. Bonus for clearing vehicles
    clear_bonus = CLEAR_BONUS * cleared_vehicles

    # 3. Phase change penalty
    phase_change_penalty = -PHASE_CHANGE_PENALTY if action != curr_phase else 0.0

    # 4. Emergency vehicle bonus
    emergency_bonus = EMERGENCY_BONUS if override else 0.0

    # 5. Potential-based shaping
    potential_diff = GAMMA * (-after_queue) - (-prev_total_queue)

    # Combine all components into the shaped reward
    shaped_reward = (
        base_reward
        + queue_penalty
        + clear_bonus
        + phase_change_penalty
        + emergency_bonus
        + potential_diff
    )

    # Update previous total queue
    prev_total_queue = after_queue

    # Append shaped reward to rewards list
    rewards.append(shaped_reward)

    # Compute total queue on all incoming lanes
    all_lanes = traci.trafficlight.getControlledLanes(tl_id)
    total_queue = sum(
        traci.lane.getLastStepHaltingNumber(l)
        for l in all_lanes
    )
    queue_lengths.append(total_queue)

    print(f"Step {step}: shaped_reward={shaped_reward:.2f}, base_reward={base_reward:.2f}, phase={action}, total_queue={total_queue}")

    # Record avg waiting time this step & track EVs that had to stop
    veh_ids = env.sumo.vehicle.getIDList()
    if veh_ids:
        wt = [traci.vehicle.getWaitingTime(v) for v in veh_ids]
        waiting_times.append(sum(wt) / len(wt))
    else:
        waiting_times.append(0.0)

    for v in veh_ids:
        if traci.vehicle.getTypeID(v) == "emergency_veh" and traci.vehicle.getWaitingTime(v) > 0:
            if v not in ev_waited:
                ev_waited.add(v)
                ev_waiting_times.append(traci.vehicle.getWaitingTime(v))

    # Build next state
    raw2 = obs2[tl_id]
    queues2 = np.array([
        sum(traci.lane.getLastStepHaltingNumber(l) for l in lanes_by_phase[p])
        for p in range(num_phases)
    ], dtype=float)
    feat2 = queues2 / (queues2.max() + 1e-3)
    state = np.concatenate([raw2, feat2])

# === Done ===
print("\n✅ Evaluation complete")
print(f"Avg. shaped reward: {np.mean(rewards):.2f}")
for p, c in phase_counts.items():
    print(f"Phase {p}: {c} times")

# === Performance Metrics Summary ===
summary = {
    "wait time (sec)":     np.mean(waiting_times),
    "travel time (sec)":   np.mean(travel_times),
    "queue length (cars)": np.mean(queue_lengths),
    "reward":       np.mean(rewards),
    "EV stopped count":    len(ev_waited),
    "EV avg wait (sec)":   np.mean(ev_waiting_times) if ev_waiting_times else 0.0
}
df_summary = pd.DataFrame([summary])
print("\nPerformance metrics")
print(df_summary.to_markdown(index=False, floatfmt=".3f"))

# === Plots ===
plt.figure(figsize=(10, 4))
plt.plot(rewards, label='reward', color='tab:blue')
plt.xlabel("Step"); plt.ylabel("Reward")
plt.title("DQN Evaluation: Rewards")
plt.grid(True); plt.legend()
plt.tight_layout()

plt.figure(figsize=(10, 4))
plt.plot(queue_lengths, label='queue', color='orange')
plt.xlabel("Step"); plt.ylabel("Total queue")
plt.title("Queue Length Over Time")
plt.grid(True); plt.legend()
plt.tight_layout()

plt.show()
env.close()
