import numpy as np
import gymnasium as gym
import random

class ChaseEnv(gym.Env):

    def __init__(self, size: int = 5):
        self.size = size

        self._player_location = np.array([0, 0], dtype=np.float64)
        self._predator_location = np.array([size - 1, 0], dtype=np.float64)

        self.action_space = gym.spaces.Discrete(3)

        # Map action numbers to a direction; the magnitude is sampled each step
        self._action_to_direction = {
            0: -1,
            1: 0,
            2: 1,
        }
        self.vertical_velocity = 0
        self.horizontal_velocity = 0
        self.predator_distance = 0

    def _get_obs(self):
        return np.concatenate([
            self._player_location,
            self._predator_location,
            [self.horizontal_velocity, self.vertical_velocity],
        ])

    def _get_info(self):
        return {
            "player_location": self._player_location.copy(),
            "predator_location": self._predator_location.copy(),
        }

    def reset(self, seed=None, options=None):
        self._player_location = np.array([0, 0], dtype=np.float64)
        self._predator_location = np.array([self.size - 1, 0], dtype=np.float64)
        self.vertical_velocity = 0
        self.horizontal_velocity = 0
        return self._get_obs(), self._get_info()

    def step_player(self):
        self._player_location = self._player_location + np.array([self.horizontal_velocity, self.vertical_velocity], dtype=np.float64)
        self._player_location = np.maximum(self._player_location, 0) # Can't leave the grid from the bottom/left
        player_reached_end = bool(np.all(self._player_location >= (self.size - 0.5)))
        return player_reached_end

    def step_predator(self):
        vec_to_player = self._player_location - self._predator_location
        dist_to_player = np.linalg.norm(vec_to_player)

        if dist_to_player <= self.predator_distance:
            # Predator lands directly on player
            self._predator_location = self._player_location.copy()
            return True
        else:
            # Move predator directly toward player by predator_distance
            unit_vector = vec_to_player / dist_to_player
            self._predator_location += unit_vector * self.predator_distance
            return False

    def step(self, action):
        direction = self._action_to_direction[int(action)]

        # Vertical acceleration is noisy: 0.5 or 1.5 in the chosen direction
        self.vertical_velocity += direction * random.choice([0.5, 1.5])

        self.vertical_velocity = min(5, self.vertical_velocity) # Ensure it maxs out at 5
        self.vertical_velocity = max(-5, self.vertical_velocity) # Ensure it min out at -5

        self.horizontal_velocity += direction * random.choice([0.5, 1.5])

        self.horizontal_velocity = min(5, self.horizontal_velocity) # Ensure it maxs out at 5
        self.horizontal_velocity = max(-5, self.horizontal_velocity) # Ensure it min out at -5

        self.predator_distance = random.randint(1, 4)

        player_wins = False
        predator_wins = False

        player_move_first = random.random() < 0.5
        if player_move_first:
            player_wins = self.step_player()
            if not player_wins:
                predator_wins = self.step_predator()
        else:
            predator_wins = self.step_predator()
            if not predator_wins:
                player_wins = self.step_player()

        if player_wins:
            reward = 1.0
        elif predator_wins:
            reward = -1.0
        else:
            reward = 0.0

        terminated = player_wins or predator_wins
        truncated = False

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def render(self):
        grid = ("*" * self.size)
