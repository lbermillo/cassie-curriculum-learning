import random

from collections import namedtuple

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))


class RingBuffer(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer   = []
        self.position = 0

    def append(self, data):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = data
        self.position = int((self.position + 1) % self.capacity)

    def __len__(self):
        return len(self.buffer)


class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = RingBuffer(capacity)

    def push(self, *args):
        """Saves a transition."""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        transitions = random.sample(self.memory.buffer, batch_size)

        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        return Transition(*zip(*transitions))

    def __len__(self):
        return len(self.memory)


# Reference RDPG implementation [https://github.com/fshamshirdar/pytorch-rdpg/]
class EpisodicMemory:
    def __init__(self, capacity, max_episode_length):
        # Max number of transitions possible will be the memory capacity, could be much less
        self.max_episode_length = max_episode_length
        self.num_episodes       = capacity // max_episode_length
        self.memory             = RingBuffer(self.num_episodes)

        # Temporal list of episode
        self.trajectory = []

    def push(self, *args):
        # store experience in trajectory buffer
        self.trajectory.append(Transition(*args))

        # store episode trajectory in memory then reset trajectory buffer when episode reaches max length or has ended
        if len(self.trajectory) >= self.max_episode_length or self.trajectory[-1].done:
            self.memory.append(self.trajectory)
            self.trajectory = []

    def sample(self, batch_size, maxlen=0):
        batch = [self.sample_trajectory(maxlen=maxlen) for _ in range(batch_size)]
        minimum_size = min(len(trajectory) for trajectory in batch)

        # truncate and return batch trajectories
        return [Transition(*zip(*trajectory[:minimum_size])) for trajectory in batch]

    def sample_trajectory(self, maxlen=0):
        # randomly select an episode from memory
        episode = self.memory.buffer[random.randrange(len(self.memory))]

        T = len(episode)
        if T > 0:
            # Take a random subset of trajectory if maxlen specified, otherwise return full trajectory
            if maxlen > 0 and T > maxlen + 1:
                t = random.randrange(T - maxlen - 1)  # Include next state after final "maxlen" state
                return episode[t:t + maxlen + 1]
            else:
                return episode

    def __len__(self):
        return len(self.memory)


if __name__ == '__main__':
    rm = ReplayMemory(10)
    # EM Buffer capacity = capacity / max_eps_length (10, 5) in this example
    em = EpisodicMemory(10 * 5, 5)
    transitions = [[i, i, i, i, 1 if i % 3 == 0 else 0] for i in range(10)]
    for t in transitions:
        rm.push(t[0], t[1], t[2], t[3], t[4])
        em.push(t[0], t[1], t[2], t[3], t[4])

    print('\n---------------ReplayMemory Sample (3) Test-----------------------------')
    print(rm.sample(3))
    print('---------------ReplayMemory Sample (3) Test-----------------------------')
    print('\n---------------Episodic Memory Sample (3) Test (maxLen=0)---------------')
    print(em.sample(3, 0))
    print('---------------Episodic Memory Sample (3) Test (maxLen=0)---------------')
    print('\n---------------Episodic Memory Sample (3) Test (maxLen=1)---------------')
    print(em.sample(3, 1))
    print('---------------Episodic Memory Sample (3) Test (maxLen=1)---------------\n')

    print(em.memory.buffer)
