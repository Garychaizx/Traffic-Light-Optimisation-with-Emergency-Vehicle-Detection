# ql_testing_with_shaped_reward.py

import os
import random
import numpy as np
import torch
import sumo_rl
import traci
from collections import defaultdict
from agents.ql_agent import QlAgent
import librosa
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import pandas as pd

# === Load Siren Detection Model ===
def load_siren_model():
    try:
        model = load_model('siren_model/best_model.keras')
        print("✅ Siren detection model loaded!")
        return model
    except Exception as e:
        print(f"❌ Error loading siren model: {e}")
        return None

siren_model = load_siren_model()

def detect_siren(audio_file_path="dynamic_sounds/ambulance.wav"):
    if siren_model is None or not os.path.exists(audio_file_path):
        return False
    try:
        audio, sr = librosa.load(audio_file_path, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=80)
        mfccs = np.mean(mfccs, axis=1).reshape(1, 1, 80)
        return float(siren_model.predict(mfccs)[0][0]) > 0.5
    except:
        return False

# ====== Build SUMO Environment ======
env = sumo_rl.SumoEnvironment(
    net_file    = 'nets/intersection/environment.net.xml',
    route_file  = 'nets/intersection/episode_routes_low.rou.xml',
    use_gui     = True,
    num_seconds = 5000,
    single_agent=False
)

# ====== Identify Traffic Light ======
tl_id = env.ts_ids[0]
print(f"Testing traffic light: {tl_id}")

# ====== Create Agent and Load Weights ======
num_phases   = len(env.traffic_signals[tl_id].all_phases)
observations = env.reset()
state_dim    = len(observations[tl_id])

agent = QlAgent(input_shape=state_dim, output_shape=num_phases)
agent.model.load_state_dict(torch.load('trained_models/model_ql.pth'))
agent.model.eval()

# ====== Shaping Hyperparameters ======
QUEUE_PEN        = 0.3    # penalty on total queue
SERVE_BONUS      = 0.2    # bonus for cars cleared on chosen phase
THROUGHPUT_BONUS = 0.2    # bonus for total cars cleared
PHASE_CHANGE_PEN = 0.05   # penalty for switching phases
EMERGENCY_BONUS  = 1.0   # bonus when an emergency override occurs
GAMMA            = getattr(agent, 'gamma', 0.99)

# ====== Metrics Containers ======
done            = {"__all__": False}
shaped_rewards  = []            # track shaped reward
queue_lengths   = []
throughputs     = []
avg_speeds      = []
stops_per_veh   = []
waiting_times   = []
ev_waited       = set()
ev_waiting      = []
travel_times    = []
depart_times    = {}
ev_delays       = []
ev_entry_times  = {}
phase_counts    = defaultdict(int)

# ====== Precompute lanes for each phase ======
phases         = env.traffic_signals[tl_id].all_phases
ctrl_lanes     = traci.trafficlight.getControlledLanes(tl_id)
lanes_by_phase = [
    [lane for lane, sig in zip(ctrl_lanes, ph.state) if sig.upper()=='G']
    for ph in phases
]

state = observations[tl_id]
step  = 0

# ====== Evaluation Loop ======
while not done["__all__"]:
    step += 1

    # --- track departures & arrivals for travel time ---
    for vid in traci.simulation.getDepartedIDList():
        depart_times[vid] = step
    for vid in traci.simulation.getArrivedIDList():
        if vid in depart_times:
            travel_times.append(step - depart_times[vid])
            del depart_times[vid]

    # --- decide action using Q-values and possible override ---
    q_vals = agent.predict_rewards(torch.FloatTensor(state))
    curr_phase    = env.traffic_signals[tl_id].green_phase
    valid_phases  = [p for p in range(num_phases)
                     if (curr_phase, p) in env.traffic_signals[tl_id].yellow_dict]
    if not valid_phases:
        valid_phases = [curr_phase]

    # emergency override?
    override     = False
    override_road= None
    for vid in traci.vehicle.getIDList():
        if traci.vehicle.getTypeID(vid) == "emergency_veh":
            rd = traci.vehicle.getRoadID(vid)
            if (750 - traci.vehicle.getLanePosition(vid)) < 100 and detect_siren():
                override      = True
                override_road = rd
                break

    if override:
        # choose NS (phase 0) or EW (phase 2)
        action = 0 if override_road.startswith(("N2TL","S2TL")) else 2
    else:
        # greedy with small epsilon
        if random.random() < 0.1:
            action = random.choice(valid_phases)
        else:
            # pick best among valid
            vals = q_vals[valid_phases]
            action = valid_phases[int(torch.argmax(vals))]

    phase_counts[action] += 1

    # --- compute shaping terms BEFORE stepping ---
    queue_per_phase   = np.array([
        sum(traci.lane.getLastStepHaltingNumber(l) for l in lanes_by_phase[p])
        for p in range(num_phases)
    ], dtype=float)
    total_q_before    = queue_per_phase.sum()
    before_sel        = queue_per_phase[action]
    change_penalty    = PHASE_CHANGE_PEN if action != curr_phase else 0.0
    ev_flag           = 1.0 if override else 0.0

    # --- step simulation ---
    obs2, raw_reward, done, _ = env.step({tl_id: action})
    base_r = raw_reward[tl_id]

    # --- recompute after-step queues ---
    queue_after     = np.array([
        sum(traci.lane.getLastStepHaltingNumber(l) for l in lanes_by_phase[p])
        for p in range(num_phases)
    ], dtype=float)
    total_q_after   = queue_after.sum()
    cleared_sel     = max(0.0, before_sel - queue_after[action])
    cleared_total   = max(0.0, total_q_before - total_q_after)

    # --- potential-based shaping ---
    pot_diff = GAMMA * (-total_q_after) - (-total_q_before)

    # --- assemble shaped reward ---
    shaped_r = (
        base_r
        - QUEUE_PEN        * total_q_before
        + SERVE_BONUS      * cleared_sel
        + THROUGHPUT_BONUS * cleared_total
        + pot_diff
        - change_penalty
        + EMERGENCY_BONUS  * ev_flag
    )
    shaped_rewards.append(shaped_r)

    # --- other performance metrics ---
    # queue length
    incoming = traci.trafficlight.getControlledLanes(tl_id)
    queue_lengths.append(sum(traci.lane.getLastStepHaltingNumber(l) for l in incoming))
    # throughput
    throughputs.append(len(traci.simulation.getArrivedIDList()))
    # avg speed
    vids = traci.vehicle.getIDList()
    if vids:
        speeds = [traci.vehicle.getSpeed(v) for v in vids]
        avg_speeds.append(sum(speeds)/len(speeds))
    else:
        avg_speeds.append(0.0)
    # stops per vehicle
    stops_per_veh.append(sum(1 for v in vids if traci.vehicle.getSpeed(v)<0.1))
    # waiting times
    wts = [traci.vehicle.getWaitingTime(v) for v in vids]
    waiting_times.append(sum(wts)/len(wts) if wts else 0.0)
    # EV tracks
    for v in vids:
        if traci.vehicle.getTypeID(v)=="emergency_veh" and traci.vehicle.getWaitingTime(v)>0:
            if v not in ev_waited:
                ev_waited.add(v)
                ev_waiting.append(traci.vehicle.getWaitingTime(v))
    # EV delay (entry-exit)
    for v in vids:
        if traci.vehicle.getTypeID(v)=="emergency_veh":
            pos = traci.vehicle.getLanePosition(v)
            if pos>650 and v not in ev_entry_times:
                ev_entry_times[v] = step
            if v in ev_entry_times and pos<5:
                ev_delays.append(step-ev_entry_times[v])
                del ev_entry_times[v]

    state = obs2[tl_id]
    print(f"Step {step}: shaped_r={shaped_r:.2f}, phase={action}, queue={total_q_before:.1f}")

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
    "reward":              np.mean(shaped_rewards),
    "EV stopped count":    len(ev_waited),
    "EV avg wait (sec)":   np.mean(ev_waiting) if ev_waiting else 0.0
}
df = pd.DataFrame([summary])
print("\nPerformance metrics")
print(df.to_markdown(index=False, floatfmt=".3f"))

# ===== Plots =====
steps = np.arange(1, len(shaped_rewards)+1)

# Plot Shaped Rewards
plt.figure(figsize=(10, 4))
plt.plot(steps, shaped_rewards, label='Reward', color='tab:blue')
plt.xlabel('Step'); plt.ylabel('Reward')
plt.title('Reward Over Time')
plt.grid(True); plt.legend(); plt.tight_layout()

# Plot Queue Lengths
plt.figure(figsize=(10, 4))
plt.plot(steps, queue_lengths, label='Queue Length', color='orange')
plt.xlabel('Step'); plt.ylabel('Queue Length')
plt.title('Queue Length Over Time')
plt.grid(True); plt.legend(); plt.tight_layout()

plt.show()
