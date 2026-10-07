import numpy as np
import gymnasium as gym
from gymnasium import spaces


class RacetrackEnv(gym.Env):
    metadata = {"render_modes": ["ansi"]}

    VERT_WIDTH = 10
    VERT_HEIGHT = 30
    HORIZ_WIDTH = 15
    HORIZ_HEIGHT = 10

    def __init__(self, wind_speed = 0, wind_chance = 0.20):
        self.wind_speed = wind_speed
        self.wind_chance = wind_chance

        self.width = self.VERT_WIDTH + self.HORIZ_WIDTH
        self.height = self.VERT_HEIGHT
        
        # track[y, x] is True for cells that are on the track
        self.track = np.zeros((self.height, self.width), dtype=bool)
        self.track[:, :self.VERT_WIDTH] = True
        self.track[self.height - self.HORIZ_HEIGHT:, self.VERT_WIDTH:] = True

        self.start_cells = [(x, 0) for x in range(self.VERT_WIDTH)]
        self.finish_x = self.width - 1

        # 9 actions: (dvx, dvy) with each in {-1, 0, +1}
        self.actions = [(dvx, dvy) for dvy in (-1, 0, 1) for dvx in (-1, 0, 1)]
        self.action_space = spaces.Discrete(len(self.actions))
        
        # observation: (x, y, vx, vy); vertical velocity is nonnegative
        self.observation_space = spaces.Box(
            low=np.array([0, 0, -4, 0]),
            high=np.array([self.width - 1, self.height - 1, 4, 4]),
            dtype=np.int64,
        )

        self.pos = None
        self.vel = None
        self.steps = 0
        self.last_path = []

    def _get_obs(self):
        return np.array([*self.pos, *self.vel], dtype=np.int64)

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
        """Cells visited moving from (x0, y0) by (vx, vy), in order."""
        n = max(abs(vx), abs(vy))
        cells = []
        for i in range(1, n + 1):
            t = i / n
            cell = (int(round(x0 + t * vx)), int(round(y0 + t * vy)))
            if not cells or cells[-1] != cell:
                cells.append(cell)
        return cells

    def reset(self, seed=None, options=None):
        idx = self.np_random.integers(len(self.start_cells))
        self.pos = self.start_cells[idx]
        self.vel = (0, 0)
        self.steps = 0
        self.last_path = []
        return self._get_obs(), {}

    def step(self, action):
        assert self.action_space.contains(action), f"invalid action {action}"
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

        # A gust of wind can push the car sideways 
        if self.np_random.random() < self.wind_chance:
            vx += self.wind_speed

        # TODO figure out the reward system when we incorporate an agent
        reward = 0
        terminated = False
        info = {"finished": False, "crashed": False}

        path = self._path(*self.pos, vx, vy)
        self.last_path = path
        for x, y in path:
            if self._on_finish(x, y):
                self.pos = (x, y)
                terminated = True
                info["finished"] = True
                break
            if not self._on_track(x, y):
                # leave the car at its last on-track cell
                terminated = True
                info["crashed"] = True
                break
            self.pos = (x, y)

        # TODO change this when we implement the actual agent logic
        truncated = False
        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        rows = []
        for y in range(self.height - 1, -1, -1):
            row = []
            for x in range(self.width):
                if (x, y) == self.pos:
                    ch = "C"
                elif (x, y) in self.last_path:
                    ch = "*"
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
