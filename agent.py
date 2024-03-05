import numpy as np
from rlcard.games.nolimitholdem import Action
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
        for k in range(t + 1, T):
            G += self._gamma ** (k - t - 1) * rewards[k]
        return G

    def _get_eligibility_vector(self, actions, selected_action: Action) -> list[list[int]]:
        action_probabilities: list[float] = []
        selectable_actions = len(list(filter(lambda x: x != 0, actions)))
        for action in actions:
            if action > 0:
                action_probabilities.append(1 / selectable_actions)
            else:
                action_probabilities.append(0)
        return 0 if selectable_actions == 0 else np.gradient(action_probabilities)[selected_action] / (1 / selectable_actions)

    def _update_parameterized_policy(self, t: int, G: int, state, selected_action: Action) -> None:
        for parameter in self._theta:
            parameter += self._alpha * self._gamma ** t * G * \
                self._get_eligibility_vector(
                    state[ACTION_MASK], selected_action)

    def REINFORCE(self, T: int, steps: list[Step]) -> None:
        states = [step["state"] for step in steps]
        actions = [step["action"] for step in steps]
        rewards = [step["reward"] for step in steps]
        print("T:", T)
        for t in range(T):
            G = self._compute_return_value(t, T, rewards)
            self._update_parameterized_policy(t, G, states[t], actions[t])

    def get_name(self):
        return self._name

    def get_theta(self):
        return self._theta
