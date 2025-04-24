import sumo_rl
import numpy as np
import torch
import random
from collections import defaultdict
import matplotlib.pyplot as plt
import traci
from agents.ql_agent import QlAgent
from tensorflow.keras.models import load_model
import librosa
import os
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

def detect_siren(audio_file_path):
    if not os.path.exists(audio_file_path):
        return False
    try:
        audio, sr = librosa.load(audio_file_path, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=80)
        mfccs = np.mean(mfccs, axis=1).reshape(1, 1, 80)
        prediction = siren_model.predict(mfccs)
        return prediction[0][0] > 0.5
    except:
        return False

# ====== Load SUMO Environment ======
env = sumo_rl.SumoEnvironment(
    net_file='nets/intersection/environment.net.xml',
    route_file='nets/intersection/episode_routes_low.rou.xml',
    use_gui=True,
    num_seconds=5000,
    single_agent=False
)

# ====== Identify Traffic Light ID ======
tl_id = env.ts_ids[0]
print(f"Testing traffic light: {tl_id}")

# ====== Define Agent and Load Trained Weights ======
num_phases = len(env.traffic_signals[tl_id].all_phases)
observations = env.reset()
state_dim = len(observations[tl_id])

agent = QlAgent(input_shape=state_dim, output_shape=num_phases)
agent.model.load_state_dict(torch.load('trained_models/model_ql.pth'))
agent.model.eval()

# ====== Evaluation Params ======
done = {"__all__": False}
rewards = []
queue_lengths = []
throughputs   = []   # number of vehicles that left the network each step
avg_speeds     = []   # mean speed of all vehicles each step
stops_per_veh  = []   # count of vehicles that came to a full stop each step
waiting_times  = []   # mean waiting time per vehicle each step
ev_delays      = []   # delay experienced by emergency vehicles

# --- new lists for travel time calculation ---
travel_times = []
depart_times  = {}

# --- track which EVs ever had to wait and their first wait time ---
ev_waited        = set()
ev_waiting_times = []

ev_entry_times  = {}     # {veh_id: step_count}
current_step    = 0
i               = 0
epsilon         = 0.1
phase_counts    = defaultdict(int)
input_tensor    = torch.FloatTensor(observations[tl_id])

# ====== Evaluation Loop ======
while not done["__all__"]:
    i += 1

    # --- record departures & arrivals for travel_time ---
    for vid in traci.simulation.getDepartedIDList():
        depart_times[vid] = i
    for vid in traci.simulation.getArrivedIDList():
        if vid in depart_times:
            travel_times.append(i - depart_times[vid])
            del depart_times[vid]

    pred_rewards = agent.predict_rewards(input_tensor)
    current_phase = env.traffic_signals[tl_id].green_phase
    valid_phases = [
        p for p in range(num_phases)
        if (current_phase, p) in env.traffic_signals[tl_id].yellow_dict
    ]
    if not valid_phases:
        valid_phases = [current_phase]

    # === Emergency vehicle override ===
    emergency_override = False
    emergency_dir      = None
    for car_id in traci.vehicle.getIDList():
        if traci.vehicle.getTypeID(car_id) == "emergency_veh":
            road_id = traci.vehicle.getRoadID(car_id)
            dist    = 750 - traci.vehicle.getLanePosition(car_id)
            if dist < 100 and detect_siren("dynamic_sounds/ambulance.wav"):
                emergency_override = True
                emergency_dir      = road_id
                break

    if emergency_override:
        if emergency_dir.startswith("N2TL") or emergency_dir.startswith("S2TL"):
            action = 0  # NS green
        elif emergency_dir.startswith("E2TL") or emergency_dir.startswith("W2TL"):
            action = 2  # EW green
    else:
        if random.random() < epsilon:
            action = random.choice(valid_phases)
        else:
            pred   = pred_rewards[valid_phases]
            action = valid_phases[torch.argmax(pred).item()]

    phase_counts[action] = phase_counts[action] + 1
    actions              = {tl_id: action}
    observations, reward, done, _ = env.step(actions)
    input_tensor         = torch.FloatTensor(observations[tl_id])

    # === Queue length ===
    incoming_lanes = env.sumo.trafficlight.getControlledLanes(tl_id)
    total_queue    = sum(
        env.sumo.lane.getLastStepHaltingNumber(lane)
        for lane in incoming_lanes
    )

    rewards.append(reward[tl_id])
    queue_lengths.append(total_queue)
    print(f"Step {i}: Reward = {reward[tl_id]:.2f}, Phase: {action}, Queue: {total_queue}")

    # ===== record throughput =====
    num_arrived = len(env.sumo.simulation.getArrivedIDList())
    throughputs.append(num_arrived)

    # ===== record avg speed =====
    veh_ids = traci.vehicle.getIDList()
    if veh_ids:
        speeds = [traci.vehicle.getSpeed(v) for v in veh_ids]
        avg_speeds.append(sum(speeds) / len(speeds))
    else:
        avg_speeds.append(0.0)

    # ===== record stops per vehicle =====
    stops_per_veh.append(sum(1 for v in veh_ids if traci.vehicle.getSpeed(v) < 0.1))

    # ===== record avg waiting time =====
    wt = [traci.vehicle.getWaitingTime(v) for v in veh_ids]
    waiting_times.append(sum(wt) / len(wt) if wt else 0.0)

    # --- track EVs that had to stop and record their first wait time ---
    for v in veh_ids:
        if traci.vehicle.getTypeID(v) == "emergency_veh" and traci.vehicle.getWaitingTime(v) > 0:
            if v not in ev_waited:
                ev_waited.add(v)
                ev_waiting_times.append(traci.vehicle.getWaitingTime(v))

    # ===== record emergency-vehicle delay =====
    for v in veh_ids:
        if traci.vehicle.getTypeID(v) == "emergency_veh":
            pos = traci.vehicle.getLanePosition(v)
            if pos > 650 and v not in ev_entry_times:
                ev_entry_times[v] = current_step
        if v in ev_entry_times and pos < 5:
            delay = current_step - ev_entry_times[v]
            ev_delays.append(delay)
            del ev_entry_times[v]

steps = list(range(1, len(rewards) + 1))

# ====== Summary ======
print("\nEvaluation completed.")
print(f"Average reward: {np.mean(rewards):.2f}")
print("\nPhase usage during test:")
for p, count in phase_counts.items():
    print(f"Phase {p}: {count} times")

# === Plots (unchanged) ===
plt.figure(figsize=(10,4))
plt.plot(steps, rewards, label='Reward')
plt.xlabel('Step'); plt.ylabel('Reward')
plt.title('Reward over Time')
plt.grid(True); plt.tight_layout()
plt.show()

plt.figure(figsize=(10,4))
plt.plot(steps, throughputs, label='Throughput')
plt.xlabel('Step'); plt.ylabel('Vehicles Arrived')
plt.title('Throughput per Step')
plt.grid(True); plt.tight_layout()
plt.show()

plt.figure(figsize=(10,4))
plt.plot(steps, avg_speeds, label='Avg Speed')
plt.xlabel('Step'); plt.ylabel('Speed (m/s)')
plt.title('Mean Vehicle Speed')
plt.grid(True); plt.tight_layout()
plt.show()

plt.figure(figsize=(10,4))
plt.plot(steps, stops_per_veh, label='Stops')
plt.xlabel('Step'); plt.ylabel('Count')
plt.title('Stops per Step')
plt.grid(True); plt.tight_layout()
plt.show()

plt.figure(figsize=(10,4))
plt.plot(steps, waiting_times, label='Waiting Time')
plt.xlabel('Step'); plt.ylabel('Time (s)')
plt.title('Average Waiting Time')
plt.grid(True); plt.tight_layout()
plt.show()

plt.figure(figsize=(10,4))
plt.plot(steps, queue_lengths, color='orange', label='Queue Length')
plt.xlabel('Step'); plt.ylabel('Queue Length')
plt.title('Queue Length over Time')
plt.grid(True); plt.legend(); plt.tight_layout()
plt.show()

plt.figure(figsize=(10,4))
plt.plot(range(1, len(ev_delays)+1), ev_delays, label='EV Delay')
plt.xlabel('Emergency Vehicle #'); plt.ylabel('Delay (steps)')
plt.title('Emergency Vehicle Delay Times')
plt.grid(True); plt.tight_layout()
plt.show()

# === 1) Compute your summary stats ===
summary = {
    "wait time (sec)":           np.mean(waiting_times),
    "travel time (sec)":         np.mean(travel_times),
    "queue length (cars)":       np.mean(queue_lengths),
    "reward":                    np.mean(rewards),
    "EV stopped count":          len(ev_waited),                                  
    "EV avg wait (sec)":         np.mean(ev_waiting_times) if ev_waiting_times else 0.0      
}

# === 2) Wrap it in a DataFrame for pretty printing ===
df_summary = pd.DataFrame([summary])
print("\nPerformance metrics")
print(df_summary.to_markdown(index=False, floatfmt=".3f"))

