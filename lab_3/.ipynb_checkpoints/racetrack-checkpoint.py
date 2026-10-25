import numpy as np
import gymnasium as gym
from gymnasium import spaces


class RacetrackEnv(gym.Env):
    metadata = {"render_modes": ["ansi"]}

    VERT_WIDTH = 10
    VERT_HEIGHT = 30
    HORIZ_WIDTH = 15
    HORIZ_HEIGHT = 10

    def __init__(self, use_provided_reward = True, wind_speed=0, wind_chance=0.30, soft_reset=False):
        self.use_provided_reward = use_provided_reward
        self.wind_speed = wind_speed
        self.wind_chance = wind_chance

        self.width = self.VERT_WIDTH + self.HORIZ_WIDTH
        self.height = self.VERT_HEIGHT

        self.track = np.zeros((self.height, self.width), dtype=bool)
        self.track[:, : self.VERT_WIDTH] = True
        self.track[self.height - self.HORIZ_HEIGHT :, self.VERT_WIDTH :] = True

        self.start_cells = [(x, 0) for x in range(self.VERT_WIDTH)]
        self.finish_x = self.width - 1
        self.soft_reset_env = soft_reset

        self.actions = [
            (-1, -1),
            (0, -1),
            (1, -1),
            (-1, 0),
            (0, 0),
            (1, 0),
            (-1, 1),
            (0, 1),
            (1, 1),
        ]
        self.action_space = spaces.Discrete(len(self.actions))

        self.pos = None
        self.vel = None
        self.terminated = False
        self.steps = 0
        self.render_count = 0
        self.wind_approaching = False

    def _get_obs(self):
        return np.array([*self.pos, *self.vel, self.wind_approaching], dtype=np.int64)

    def _on_track(self, x, y):
        """
        Check if car is on the track
        """
        return 0 <= x < self.width and 0 <= y < self.height and self.track[y, x]

    def _on_finish(self, x, y):
        """
        Check if the car finished
        """
        return x == self.finish_x and self._on_track(x, y)

    def _path(self, x0, y0, vx, vy):
        """Grabs all the cells visted from the start of the step to the end of the stop"""
        n = max(abs(vx), abs(vy))
        cells = []
        
        for i in range(1, n + 1):
            t = i / n
            cell = (int(round(x0 + t * vx)), int(round(y0 + t * vy)))
            if not cells or cells[-1] != cell:
                cells.append(cell)
        return cells

    def soft_reset(self, seed=None, options=None):
        idx = self.np_random.integers(len(self.start_cells))
        self.pos = self.start_cells[idx]
        self.vel = (0, 0)
        self.terminated = False
        self.turncated = False
        self.wind_approaching = False
        self.info = {"finished": False, "crashed": False}
        return self._get_obs(), {}

    def reset(self, seed=None, options=None):
        self.soft_reset()
        self.steps = 0
        return self._get_obs(), {}
        
    def provided_reward(self):
        return 1/max(1,((self.pos[0] - (self.VERT_WIDTH - 1))**2 + (self.HORIZ_WIDTH/2.0 - self.pos[1])**2))
        
    def my_reward(self):
        if self.info['finished']:
            return 100.0
        
        if self.info['crashed']:
            return -50.0
            
        reward = -1.0
        
        progress_bonus = 0.1 * self.pos[0]  
        
        return reward + progress_bonus

    def step(self, action):
        if self.soft_reset_env and self.info["crashed"]:
            self.soft_reset()
        dvx, dvy = self.actions[action]

        vx = self.vel[0] + dvx
        vx = min(4, vx)
        vx = max(-4, vx)

        vy = self.vel[1] + dvy
        vy = min(4, vy)
        vy = max(0, vy)

        # Both components may only be zero on the starting line; otherwise
        # the velocity change is rejected and the car keeps its velocity.
        if vx == 0 and vy == 0 and self.pos[1] != 0:
            vx, vy = self.vel
        self.vel = (vx, vy)
        self.steps += 1

        self.terminated = False
        self.info = {"finished": False, "crashed": False }
        if self.wind_approaching:
            vx += self.wind_speed
            self.wind_approaching = False

        # A gust of wind can push the car sideways
        if self.np_random.random() < self.wind_chance:
            self.wind_approaching = True

        # TODO figure out the reward system when we incorporate an agent
        if self.use_provided_reward:
            reward = self.provided_reward()
        else:
            reward = self.my_reward()
        path = self._path(*self.pos, vx, vy)
        for x, y in path:
            if self._on_finish(x, y):
                self.pos = (x, y)
                self.terminated = True
                self.info["finished"] = True
                break
            if not self._on_track(x, y):
                # leave the car at its last on-track cell
                self.terminated = not self.soft_reset_env
                self.info["crashed"] = True
                break
            self.pos = (x, y)

        # TODO change this when we implement the actual agent logic
        truncated = False
        return self._get_obs(), reward, self.terminated, truncated, self.info

    def render(self):
        """
        Helper function to help visualize the track
        """
        self.render_count += 1
        rows = []
        for y in range(self.height - 1, -1, -1):
            row = []
            for x in range(self.width):
                if (x, y) == self.pos:
                    ch = "C"
                elif not self.track[y, x]:
                    ch = "#"
                elif x == self.finish_x:
                    ch = "F"
                elif y == 0:
                    ch = "S"
                else:
                    ch = "."
                row.append(ch)
            rows.append("".join(row))
        return "\n".join(rows) + f"\npos={self.pos} vel={self.vel} step={self.steps}\n"

    def close(self):
        pass
