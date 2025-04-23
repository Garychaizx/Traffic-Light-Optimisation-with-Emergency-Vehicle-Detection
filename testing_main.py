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
    route_file='nets/intersection/episode_routes.rou.xml',
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
i = 0
epsilon = 0.1
phase_counts = defaultdict(int)
input_tensor = torch.FloatTensor(observations[tl_id])

# ====== Evaluation Loop ======
while not done["__all__"]:
    i += 1

    pred_rewards = agent.predict_rewards(input_tensor)
    current_phase = env.traffic_signals[tl_id].green_phase
    valid_phases = [p for p in range(num_phases) if (current_phase, p) in env.traffic_signals[tl_id].yellow_dict]
    if not valid_phases:
        valid_phases = [current_phase]

    # === Emergency vehicle override ===
    emergency_override = False
    emergency_dir = None
    for car_id in traci.vehicle.getIDList():
        if traci.vehicle.getTypeID(car_id) == "emergency_veh":
            road_id = traci.vehicle.getRoadID(car_id)
            dist = 750 - traci.vehicle.getLanePosition(car_id)
            if dist < 100 and detect_siren("dynamic_sounds/ambulance.wav"):
                emergency_override = True
                emergency_dir = road_id
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
            pred = pred_rewards[valid_phases]
            action = valid_phases[torch.argmax(pred).item()]

    phase_counts[action] += 1
    actions = {tl_id: action}
    observations, reward, done, _ = env.step(actions)
    input_tensor = torch.FloatTensor(observations[tl_id])

    # === Queue length ===
    incoming_lanes = env.sumo.trafficlight.getControlledLanes(tl_id)
    total_queue = sum(env.sumo.lane.getLastStepHaltingNumber(lane) for lane in incoming_lanes)

    rewards.append(reward[tl_id])
    queue_lengths.append(total_queue)
    print(f"Step {i}: Reward = {reward[tl_id]:.2f}, Phase: {action}, Queue: {total_queue}")

# ====== Summary ======
print("\nEvaluation completed.")
print(f"Average reward: {np.mean(rewards):.2f}")
print("\nPhase usage during test:")
for p, count in phase_counts.items():
    print(f"Phase {p}: {count} times")

# ====== Plot Rewards ======
plt.figure(figsize=(10, 4))
plt.plot(rewards, label='Reward')
plt.xlabel('Step')
plt.ylabel('Reward')
plt.title('Testing Rewards over Time')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# ====== Plot Queue Length ======
plt.figure(figsize=(10, 4))
plt.plot(queue_lengths, color='orange', label='Queue Length')
plt.xlabel('Step')
plt.ylabel('Queue Length')
plt.title('Queue Length over Time')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
