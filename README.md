# Traffic Light Optimisation with Emergency Vehicle Detection

## Overview

This project enhances urban traffic management by optimizing traffic light control systems to prioritize emergency vehicles. By integrating real-time siren detection, the system ensures that emergency vehicles (ambulances, fire trucks, police cars) receive timely green lights, reducing response times and improving overall traffic flow.

## Features

- **Emergency Vehicle Detection**: Detects the presence of emergency vehicles via siren sound recognition.
- **Dynamic Traffic Light Control**: Adjusts traffic signals in real-time to prioritize emergency vehicles.
- **Modular Design**: Supports both single and double intersection management.
- **Simulation Environment**: Provides simulation tools to validate the system under different traffic conditions.

## Getting Started

### Prerequisites

- Python 3.8 or above
- Required libraries listed in `requirements.txt`

### Installation
1. Clone the repository and install the requirements.txt:

```bash
git clone https://github.com/Garychaizx/Traffic-Light-Optimisation-with-Emergency-Vehicle-Detection.git
cd Traffic-Light-Optimisation-with-Emergency-Vehicle-Detection
pip install -r requirements.txt
```

## Running The Simulation

All of the models are trained and stored in trained_models folder. 

### Specifying a Traffic Flow for Simulation

In the **Running the Simulation** section, you can specify the desired traffic flow by selecting the appropriate route file that defines the traffic conditions. The project includes multiple predefined traffic flow scenarios, such as low, medium, and high traffic densities, which are stored in the `nets` directory.

#### Steps to Choose and Specify a Traffic Flow:

1. **Locate the Route Files**:
   - The route files are located in the `nets/intersection` or `nets/double` directories.
   - Examples include:
     - `episode_routes_low.rou.xml` (low traffic)
     - `episode_routes_med.rou.xml` (medium traffic)
     - `episode_routes_high.rou.xml` (high traffic)

2. **Modify the Route File in the Code**:
   - Open the corresponding simulation script (e.g., `Normal_Intersections/dqn_testing.py` or `Double_Intersections/DI_dqn_testing.py`).
   - Update the `route_file` parameter in the `SumoEnvironment` initialization to point to the desired route file. For example:
     ```python
     env = sumo_rl.SumoEnvironment(
         net_file="nets/intersection/environment.net.xml",
         route_file="nets/intersection/episode_routes_high.rou.xml",
         use_gui=True,
         num_seconds=5000,
         single_agent=False
     )
     ```

### Executing the Code
1. Single Intersection

```bash
python Normal_Intersections/{agent_name}_testing.py
```

1. Double Intersection

```bash
python Double_Intersections/DI_{agent_name}_testing.py
```


