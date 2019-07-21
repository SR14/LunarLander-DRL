# LunarLander-RL

Solved OpenAI's Continuous Lunar Lander with a an average reward of 200+ over the last 100 episodes at episode 493. The agent folder contains all the python scripts that are called by the jupyter notebook file.
  
Included in the agent folder is memory.py, the PER buffer; noise.py, Gaussian noise; actor.py, actor network architecture; critic.py, critic network architecture; agent.py, DDPG algorithm with third critic and noise addition to target action.

OpenAI's "LunarLanderContinuous-v2" environment implementation in task.py

Test episode's data can be found in data.txt, and the test episode gif can be found in LunarLander.gif

Looking for feedback on how the model could be improved! (:
