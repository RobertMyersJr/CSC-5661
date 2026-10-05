import numpy as np
import gymnasium as gym
from gymnasium import spaces


class RacetrackEnv(gym.Env):
    metadata = {"render_modes": ["human", "ansi"]}

    VERT_WIDTH = 10
    VERT_HEIGHT = 30
    HORIZ_WIDTH = 15
    HORIZ_HEIGHT = 10
    MAX_SPEED = 4  # velocity components must be < 5

    def __init__(self, render_mode=None, step_reward=-1.0, crash_reward=-100.0,
                 max_episode_steps=None):
        """
        step_reward: reward for every time step (encourages finishing quickly).
        crash_reward: extra reward when the car leaves the track. Without a
            penalty, crashing immediately would beat a long run of -1 rewards.
        max_episode_steps: optional truncation limit (None = no limit).
        """
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.step_reward = step_reward
        self.crash_reward = crash_reward
        self.max_episode_steps = max_episode_steps

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
            low=np.array([0, 0, -self.MAX_SPEED, 0]),
            high=np.array([self.width - 1, self.height - 1, self.MAX_SPEED, self.MAX_SPEED]),
            dtype=np.int64,
        )

        self.pos = None
        self.vel = None
        self.steps = 0
        self.last_path = []

    def _obs(self):
        return np.array([*self.pos, *self.vel], dtype=np.int64)

    def _on_track(self, x, y):
        return 0 <= x < self.width and 0 <= y < self.height and self.track[y, x]

    def _on_finish(self, x, y):
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
        super().reset(seed=seed)
        idx = self.np_random.integers(len(self.start_cells))
        self.pos = self.start_cells[idx]
        self.vel = (0, 0)
        self.steps = 0
        self.last_path = []
        if self.render_mode == "human":
            self.render()
        return self._obs(), {}

    def step(self, action):
        assert self.action_space.contains(action), f"invalid action {action}"
        dvx, dvy = self.actions[action]
        vx = int(np.clip(self.vel[0] + dvx, -self.MAX_SPEED, self.MAX_SPEED))
        vy = int(np.clip(self.vel[1] + dvy, 0, self.MAX_SPEED))
        # Both components may only be zero on the starting line; otherwise
        # the velocity change is rejected and the car keeps its velocity.
        if vx == 0 and vy == 0 and self.pos[1] != 0:
            vx, vy = self.vel
        self.vel = (vx, vy)
        self.steps += 1

        reward = self.step_reward
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
                reward += self.crash_reward
                break
            self.pos = (x, y)

        truncated = (not terminated and self.max_episode_steps is not None
                     and self.steps >= self.max_episode_steps)

        if self.render_mode == "human":
            self.render()
        return self._obs(), reward, terminated, truncated, info

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
        text = "\n".join(rows) + f"\npos={self.pos} vel={self.vel} step={self.steps}\n"
        if self.render_mode == "ansi":
            return text
        print(text)

    def close(self):
        pass
