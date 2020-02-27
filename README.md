# Lunar Lander - Deep Reinforcement Learning
Solved OpenAI's [Continuous Lunar Lander environment](https://gym.openai.com/envs/LunarLanderContinuous-v2/) in less than 500 training episodes; this is a [top 5 leaderboard performance](https://github.com/openai/gym/wiki/Leaderboard#lunarlandercontinuous-v2). 

![GIF](https://github.com/SR14/LunarLander-DeepRL/blob/master/LunarLander.gif)

## Getting Started

1. The **[Final Report](https://github.com/SR14/LunarLander-DeepRL/blob/master/LunarLanderReport.pdf)** detailing every single step of the project can be found in the 'LunarLanderReport.pdf' repository file. 

2. The project's proposal can be found [here](https://github.com/SR14/LunarLander-DeepRL/blob/master/Project%20proposal.pdf).

3. Python scripts containing all algorithmic components and their code can be found in the [agents folder](https://github.com/SR14/LunarLander-DeepRL/tree/master/agents). *(More specifically, the corresponding components of the algorithm are memory.py for the Prioritized Experience Replay, or PER, buffer; noise.py for Gaussian noise added to the learning stage; actor.py holds the actor network architecture; critic.py holds critic network architecture; and the agent.py holds the compiled DDPG algorithm with a third critic and added target action noise.)*


4. OpenAI's "LunarLanderContinuous-v2" environment implementation can be seen in this [Python script](https://github.com/SR14/LunarLander-DeepRL/blob/master/task.py). 


5. The Jupyter Notebook used to compile all Python scripts can be found [here](https://github.com/SR14/LunarLander-DeepRL/blob/master/LunarLanderContinuous-v2.ipynb)


6. Test episode's data can be found in the [data.txt](https://github.com/SR14/LunarLander-DeepRL/blob/master/data.txt) file, and the test episode's gif can be found in LunarLander.gif

Looking for feedback on how the model could be improved! (:

>Other Data Science & Machine Learning projects can be found on the [main GitHub repo](https://github.com/SR14). Thank you!

