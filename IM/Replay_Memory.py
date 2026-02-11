import random
import numpy as np
from collections import namedtuple, deque

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class PrioritizedReplayMemory(object):
    def __init__(self, capacity, alpha=0.6, n_step=3, gamma=0.99, eps=1e-5):
        self.capacity = capacity
        self.memory = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.position = 0
        self.alpha = float(alpha)
        self.eps = float(eps)

        self.n_step = int(n_step)
        self.gamma = float(gamma)
        self.n_step_buffer = deque(maxlen=self.n_step)

    def set_alpha(self, alpha):
        self.alpha = float(alpha)

    def reset(self):
        self.memory = []
        self.priorities[:] = 0.0
        self.position = 0
        self.n_step_buffer.clear()

    def _get_n_step_info(self):
        reward, next_state, done = 0.0, None, False
        for idx, transition in enumerate(self.n_step_buffer):
            r, s_next, d = transition.reward, transition.next_state, transition.done
            reward += (self.gamma ** idx) * r
            next_state, done = s_next, d
            if done:
                break
        return self.n_step_buffer[0].state, self.n_step_buffer[0].action, reward, next_state, done

    def push(self, *args, td_error=1.0):
        self.n_step_buffer.append(Transition(*args))
        if len(self.n_step_buffer) == self.n_step:
            state, action, reward, next_state, done = self._get_n_step_info()
            max_priority = self.priorities.max() if self.memory else 1.0
            if len(self.memory) < self.capacity:
                self.memory.append(None)
            self.memory[self.position] = Transition(state, action, next_state, reward, done)
            self.priorities[self.position] = max_priority
            self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.memory) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.position]

        probabilities = priorities ** self.alpha
        sum_prob = probabilities.sum()
        if sum_prob == 0:
            probabilities = np.ones_like(probabilities) / len(probabilities)
        else:
            probabilities /= sum_prob

        indices = np.random.choice(len(self.memory), batch_size, p=probabilities)
        batch = [self.memory[idx] for idx in indices]

        total = len(self.memory)
        weights = (total * probabilities[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        return {"samples": batch, "indices": indices, "weights": weights}

    def update_priorities(self, indices, priorities):
        priorities = np.asarray(priorities).reshape(-1)
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = abs(float(priority)) + self.eps

    def __len__(self):
        return len(self.memory)