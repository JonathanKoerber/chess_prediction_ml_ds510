import numpy as np
import random
import tensorflow as tf
from tensorflow.keras import layers, models

class DQN_Agent:

    def __init__(self, env, state_shape, num_actions, replay_buffer_size=1000) -> None:
        self.env = env
        self.state_shape = state_shape
        self.num_actions = num_actions
        self.model = self._build_model()
        self.replay_buffer = []
        self.replay_buffer_size = replay_buffer_size
       
    def _build_model(self):
        model = models.Sequential([
            layers.Input(shape=(12, 8, 8)),
            layers.Dense(64, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(self.num_actions, activation='linear')
        ])
        model.compile(optimizer='adam', loss='mse')
        return model
    
    def get_q_values(self, state):
        state = np.expand_dims(state, axis=0)
        q_values = self.model.predict(state)
        return q_values.flatten()
    
    def train(self, states, targets):
        states = np.array(states)
        targets = np.array(targets)
        current_q_values = self.model.predict(states)
        current_q_values[np.arange(len(current_q_values)), targets] = targets
        self.model.fit(states, targets, epochs=1, verbose=0)

    def choose_action(self, state, epsilon=0.1):
        legal_moves = list(self.env.board.legal_moves)

        if len(legal_moves) == 0:
            # No legal moves available, handle this case
            
            self.env.done = True
            return None
        
        q_values = self.get_q_values(state)
       
        if np.random.rand() < epsilon:
            # Choose a random action from the legal moves
            action = random.choice(legal_moves)
        else:
            # Choose the action with the highest Q-value
            max_q_value = np.max(q_values)
    
            best_actions = [legal_moves[i % len(legal_moves)] for i, q_value in enumerate(q_values) if q_value == max_q_value]
        
        # Randomly choose one of the best actions in case of ties
            action = random.choice(best_actions)
        next_state, reward, done = self.env.step(action)

        return action
    
    def store_experience(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        self.replay_buffer.append(experience)
        if len(self.replay_buffer) > self.replay_buffer_size:
            self.replay_buffer.pop(0)

    def step(self, state, epsilon=0.1):
        action = self.choose_action(state, epsilon)
        self.env.make_move(action) #make the move
        next_state, reward, done = self.env.step(action)
        experience = (state, action, reward, next_state, done)
        self.store_experience(*experience)
        return experience
        