## Environment

For this project, you will train an agent to navigate (and collect bananas!) in a large, square world.

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around the agent's forward direction. Given this information, the agent has to learn how to best select actions. Four discrete actions are available, corresponding to:

* ```0``` - move forward.
* ```1``` - move backward.
* ```2``` - turn left.
* ```3``` - turn right.

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.

## Installation of the Unity Environment
* Clone or download the Deep Reinforcment Repocitory provided by udacity from GitHub. Navigate to the python directory and install the requirements. Please read the [detailed installation instructions](https://github.com/udacity/deep-reinforcement-learning/blob/master/README.md). (Hint: create an python environment to install the requirements into it)

```
git clone https://github.com/udacity/deep-reinforcement-learning.git
cd deep-reinforcement-learning/python
pip install .
```
* Download the Environment from one of the following sources depending on the requirment of your system.
    * [Mac OSX](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    * [Linux](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    * [Windows (32-bit)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    * [Windows (64-bit)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
* Unzip the environment to the root path of this project.
## Run the Project
Open banana.ipynb and run the cells!