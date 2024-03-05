import numpy as np
from custom_types import Step
from constants import DEFAULT_AGENT_NAME, OBSERVATION_SPACE_SIZE, ACTION_MASK


class Agent:
    def __init__(self, alpha: float, gamma: float, name=DEFAULT_AGENT_NAME) -> None:
        self._alpha = alpha
        self._gamma = gamma
        self._name = name


class REINFORCEAgent(Agent):
    def __init__(self, alpha: float, gamma: float, theta=np.zeros(OBSERVATION_SPACE_SIZE), name=DEFAULT_AGENT_NAME) -> None:
        super().__init__(alpha, gamma, name)
        self._theta = theta

    def _compute_return_value(self, t: int, T: int, rewards: list[int]) -> int:
        G = 0
        print("Rewards:", rewards)
        for k in range(t + 1, T):
            G += self._gamma ** (k - t - 1) * rewards[k]
        return G

    def _get_eligibility_vector(self, actions):
        return 0

    def _update_parameterized_policy(self, t: int, G: int, state):
        for parameter in self._theta:
            parameter += self._alpha * self._gamma ** t * G * \
                self._get_eligibility_vector(state[ACTION_MASK])

    def REINFORCE(self, T: int, steps: list[Step]):
        states = [step["state"] for step in steps]
        actions = [step["action"] for step in steps]
        rewards = [step["reward"] for step in steps]
        print("T:", T)
        for t in range(T):
            G = self._compute_return_value(t, T, rewards)
            self._update_parameterized_policy(t, G, states[t])

    def get_name(self):
        return self._name

    def get_theta(self):
        return self._theta
