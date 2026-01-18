# Build Setup

------------
First, please make sure you have installed Conda. Then, our environment can be installed by:

    conda create -n SCOPE python=3.9.23
    conda activate SCOPE
    pip install -r requirements.txt

# Key Dependencies

------------
##### *Lower versions are not guaranteed to be feasible:*

    numpy: 1.26.3
    gym: 0.26.2
    pillow: 10.2.0
    torch: >=2.4.0+cu124
    torchaudio: >=2.4.0+cu124
    torchvision: >=0.19.0+cu124

# Project Structure

------------

	├── frames				// Each frame in the evaluation
    ├── model                               // RL models
	│   ├── maddpg.py  			// MADDPG
	│   └── matd3.py   		        // MATD3
    ├── predict_model                       // Predict model
	│   └── Dlinear
	├── tmp				        // Model path
	│   ├── last		  		// Model of the last training
	│   ├── maddpg				// Model of maddpg
	│   │	└── UAV_Round_up                // Name of environment
	│   ├── matd3 				// Model of matd3
	│   │	└── UAV_Round_up
	│   └── ...
	├── result 				// Result gif
	│   ├── roundup_fail.gif  	        // Fail result
	│   └── roundup_success.gif             // Success result
	├── agent.py   				// Composition of Agent
	├── buffer.py      			// Buffer for experience replay
	├── CBF.py 				// CBF function
	├── evaluate.py				// Evaluate the model
	├── math_tool.py   		        // Math-relevent functions
	├── networks.py				// Actor and Critic networks
	├── predictor.py   			// Pridict module
	├── sim_env.py 				// Environment
	├── train.py   				// Training model
	├── target.png     			// Target icon
	├── UAV.png    				// Agent icon
	└── ReadMe.md  				// Helps document

# Sim Result

------------
<p align="center">
  <img src="./roundup_success.gif" width="300" style="margin-right:12px;" />
  <img src="./roundup_sim.gif" width="300" />
</p>

## Explanation of Document

------------
- Physics simulator: MuJoCo -> [google-deepmind/mujoco](https://github.com/google-deepmind/mujoco);
- UAV MuJoCo quadrotor model (`crazyfile`): adapted from LGQWakkk's [Quadrotor-NMPC-Control](https://github.com/LGQWakkk/Quadrotor-NMPC-Control/tree/main/crazyfile) repository.
