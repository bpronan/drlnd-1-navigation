[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

# BanaNavigation

This project covers the solution for Project 1 of the Udacity Deep Reinforcement Learning Nanodegree. The goal of the project was to train an agent in an environment with a continuous state space.

## Project Details

![Trained Agent](assets/trained_agent.gif)

This project required training an agent to navigate a square world and collect bananas within the world. The agent receives a reward of +1 for collecting a yellow banana and a reward of -1 for collecting a blue banana.

The input state space has 37 dimensions including the agent's velocity and ray casts to detect objects around the agent's forward direction vector. The agent has 4 possible actions:
- **`0`** - move forward
- **`1`** - move backward
- **`2`** - turn left
- **`3`** - turn right

The goal of the agent is to collect as many yellow bananas it can while avoiding blue bananas, and it was trained using versions of the Deep Q-Network algorithm.

## Getting Started

### Prerequisites (Conda)

1. Setup conda environment `conda create -n banana python=3.6` and `conda activate banana`.
1. Install [PyTorch version 0.4.1](https://pytorch.org/get-started/previous-versions/) for the version of CuDA you have installed.
2. Run `pip -q install ./python`

### Unity Environment Setup
1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

2. Place the file in the DRLND GitHub repository, in the `environments/` folder, and unzip (or decompress) the file.

### Instructions

Run `jupyter notebook` from this directory.

Open `Navigation.ipynb` and run the cells to see how the agent was trained!
