# dqn_testing_fixed.py

import os
import numpy as np
import torch
import sumo_rl
import traci
from agents.dqn_agent import DQNAgent
import matplotlib.pyplot as plt

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
    route_file  = 'nets/intersection/episode_routes.rou.xml',
    use_gui     = True,
    num_seconds = 5000,
    single_agent=False
)
tl_id       = env.ts_ids[0]
phases      = env.traffic_signals[tl_id].all_phases
num_phases  = len(phases)

# === reset once to start SUMO & traci ===
obs0 = env.reset()

# === now safe to call traci ===
ctrl_lanes = traci.trafficlight.getControlledLanes(tl_id)
lanes_by_phase = []
for ph in phases:
    # one char per controlled lane; 'G' or 'g' means green
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
agent.load('trained_models/model_dqn_adaptive_fixed2.pth')
agent.epsilon = 0.0   # full greedy for evaluation

# === ready to roll ===
state_dict   = obs0
state_raw    = state_dict[tl_id]
# initial queue features
queues0      = np.array([
    sum(traci.lane.getLastStepHaltingNumber(l) for l in lanes_by_phase[p])
    for p in range(num_phases)
], dtype=float)
queue_feat0  = queues0 / (queues0.max() + 1e-3)
state        = np.concatenate([state_raw, queue_feat0])

rewards      = []
queue_lengths= []
phase_counts = {p:0 for p in range(num_phases)}
done         = {"__all__": False}
step         = 0

while not done["__all__"]:
    step += 1

    # figure out valid yellow transitions
    curr_phase = env.traffic_signals[tl_id].green_phase
    valid = [
        p for p in range(num_phases)
        if (curr_phase, p) in env.traffic_signals[tl_id].yellow_dict
    ]
    if not valid:
        valid = [curr_phase]

    # check for emergency override
    override = False
    for vid in env.sumo.vehicle.getIDList():
        if env.sumo.vehicle.getTypeID(vid) == "emergency_veh":
            dist = 750 - env.sumo.vehicle.getLanePosition(vid)
            if dist < 100 and detect_siren(siren_model):
                override = True
                road = env.sumo.vehicle.getRoadID(vid)
                break

    if override:
        # force N-S straight (phase 0) or E-W straight (phase 2)
        if road.startswith(("N2TL","S2TL")):
            action = 0
        else:
            action = 2
    else:
        action = agent.choose_action(state, valid)

    phase_counts[action] += 1

    # step environment
    obs2, rdict, done, _ = env.step({tl_id: action})
    base_r = rdict[tl_id]
    rewards.append(base_r)

    # compute total queue on **all** incoming lanes
    all_lanes = traci.trafficlight.getControlledLanes(tl_id)
    total_q   = sum(
        traci.lane.getLastStepHaltingNumber(l)
        for l in all_lanes
    )
    queue_lengths.append(total_q)

    print(f"Step {step}: reward={base_r:.2f} phase={action} total_queue={total_q}")

    # build next state: raw obs + per-phase queue features
    raw2 = obs2[tl_id]
    queues2 = np.array([
        sum(traci.lane.getLastStepHaltingNumber(l) for l in lanes_by_phase[p])
        for p in range(num_phases)
    ], dtype=float)
    feat2 = queues2 / (queues2.max() + 1e-3)
    state = np.concatenate([raw2, feat2])

# === done ===
print("\n✅ Evaluation complete")
print(f"Avg. reward: {np.mean(rewards):.2f}")
for p, c in phase_counts.items():
    print(f"Phase {p}: {c} times")

# === plots ===
plt.figure(figsize=(10,4))
plt.plot(rewards, label='reward')
plt.xlabel("Step"); plt.ylabel("Reward")
plt.title("DQN Evaluation Rewards")
plt.grid(True); plt.legend()
plt.tight_layout()

plt.figure(figsize=(10,4))
plt.plot(queue_lengths, label='queue', color='orange')
plt.xlabel("Step"); plt.ylabel("Total queue")
plt.title("Queue Length Over Time")
plt.grid(True); plt.legend()
plt.tight_layout()

plt.show()
env.close()
