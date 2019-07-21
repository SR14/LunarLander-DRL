from agents.actor import Actor
from agents.critic import Critic
from agents.noise import GaussianNoise
from agents.memory import PrioritizedReplayBuffer
from collections import namedtuple
from numpy.random import seed
from tensorflow import set_random_seed
import numpy as np
import random
from numpy.random import seed
from tensorflow import set_random_seed
seed(14)
set_random_seed(14)

class TD3():
    """Reinforcement Learning agent that learns using DDPG."""
    def __init__(self, task, lra, lrc, db):
        self.task = task
        self.s_sz = task.state_size
        self.a_sz = task.action_size
        self.a_max = task.max_action

        # Actor (Policy) Model
        self.actor_local = Actor(self.s_sz, self.a_sz, lra)
        self.actor_target = Actor(self.s_sz, self.a_sz, lra)

        # First Critic (Value) Model
        self.critic_local_1 = Critic(self.s_sz, self.a_sz, lrc)
        self.critic_target_1 = Critic(self.s_sz, self.a_sz, lrc)
        
        # Second Critic (Value) Model
        self.critic_local_2 = Critic(self.s_sz, self.a_sz, lrc)
        self.critic_target_2 = Critic(self.s_sz, self.a_sz, lrc)

        # Initialize target model parameters with local model parameters
        self.critic_target_1.model.set_weights(self.critic_local_1.model.get_weights())
        self.critic_target_2.model.set_weights(self.critic_local_2.model.get_weights())        
        self.actor_target.model.set_weights(self.actor_local.model.get_weights())

        # Noise process
        self.noise = GaussianNoise(self.a_sz)

        # Replay memory
        self.num_exp = 0
        self.batch = 32
        self.buffer = 10000
        labels = ["state", "action", "reward", "next_state", "done"]
        self.experience = namedtuple("Experience", field_names=labels)
        self.memory = PrioritizedReplayBuffer(self.buffer, self.batch, db)

        # Algorithm parameters
        self.gamma = 0.99  # discount factor
        self.tau = 0.005   # for soft update of target parameters

    def reset_episode(self):
        state = self.task.reset()
        self.last_state = state
        self.num_exp
        return state

    def step(self, action, reward, next_state, done, PER_init=False):
         # Save experience / reward
        exp = self.experience(self.last_state, action, reward, next_state, done)
        self.memory.add(exp)
        self.num_exp += 1
        
        # Roll over last state and action
        self.last_state = next_state

        # Learn, if enough samples are available in memory
        if PER_init:
            p_idx, weights, experiences = self.memory.sample()
            mean_abs_error, loss = self.learn(experiences, weights, p_idx)
            return mean_abs_error, loss
        

    def act(self, state, training=True):
        """Returns actions for given state(s) as per current policy."""
        state = np.reshape(state, [-1, self.s_sz])              
        action = self.actor_local.model.predict(state)
        if training:        
            noise = self.noise.sample(0.1)
            return list((action + noise)[0])   # add some noise for exploration
        else:
            action = self.actor_target.model.predict(state)
            return list(action[0])

    def learn(self, exp, weights, p_idx):
        states = np.vstack([e.state for e in exp])
        actions = np.array([e.action for e in exp]).astype(np.float32).reshape(-1, self.a_sz)
        rewards = np.array([e.reward for e in exp]).astype(np.float32).reshape(-1, 1)
        dones = np.array([e.done for e in exp]).astype(np.uint8).reshape(-1, 1)
        next_states = np.vstack([e.next_state for e in exp])            
        weights = np.ndarray.flatten(np.array([w for w in weights]).astype(np.float32))

        # Get predicted next-state actions and Q values from target models
        # Q_targets_next = critic_target(next_state, actor_target(next_state))
        actions_next = self.actor_target.model.predict_on_batch(next_states)
        target_noise = self.noise.sample(0.2, self.batch, True)
        actions_next = np.clip(actions_next + target_noise, -self.a_max, self.a_max)
        Q_targets_1 = self.critic_target_1.model.predict_on_batch([next_states, actions_next]).reshape(-1,1)
        Q_targets_2 = self.critic_target_2.model.predict_on_batch([next_states, actions_next]).reshape(-1,1)  
        Q_targets_next = np.minimum(Q_targets_1, Q_targets_2)
        Q_targets = rewards + self.gamma * Q_targets_next * (1 - dones)        

        # Compute Q targets for current states and train critic model (local)
        Q_local_1 = self.critic_local_1.model.predict_on_batch([states, actions])
        Q_local_2 = self.critic_local_2.model.predict_on_batch([states, actions])
        loss_1 = self.critic_local_1.model.train_on_batch([states, actions], Q_targets, weights)   
        loss_2 = self.critic_local_2.model.train_on_batch([states, actions], Q_targets, weights)  
        Q_error_1 = np.absolute(Q_targets - Q_local_1)
        Q_error_2 = np.absolute(Q_targets - Q_local_2)
        Q_error = np.mean([Q_error_1, Q_error_2], axis=0)
        self.memory.update_weights(p_idx, Q_error)

        # Train actor model (local)
        actor_actions = self.actor_local.model.predict_on_batch(states)
        action_grads = self.critic_local_1.get_gradients([states, actor_actions, 0])
        action_grads = np.reshape(action_grads, (-1, self.a_sz))
        self.actor_local.train_fn([states, action_grads, 1])  # custom training function
        
        # Soft-update target models
        self.soft_update(self.critic_local_1.model, self.critic_target_1.model)
        self.soft_update(self.critic_local_2.model, self.critic_target_2.model)        
        self.soft_update(self.actor_local.model, self.actor_target.model)  
        return np.mean(Q_error), np.mean([loss_1, loss_2])
        
        
    def soft_update(self, local_model, target_model):
        """Soft update model parameters."""
        local_weights = np.array(local_model.get_weights())
        target_weights = np.array(target_model.get_weights())
        
        message = "Local and target model parameters must have the same size"
        assert len(local_weights) == len(target_weights), message

        new_weights = self.tau * local_weights + (1 - self.tau) * target_weights
        target_model.set_weights(new_weights)
