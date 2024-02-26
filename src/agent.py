from custom_types import Step
from constants import DEFAULT_AGENT_NAME


class Agent:
    def __init__(self, alpha: float, gamma: float, name=DEFAULT_AGENT_NAME) -> None:
        self.alpha = alpha
        self.gamma = gamma
        self.name = name


class REINFORCEAgent(Agent):
    def __init__(self, alpha: float, gamma: float, theta=[0], name=DEFAULT_AGENT_NAME) -> None:
        super().__init__(alpha, gamma, name)
        self.theta = theta

    def _compute_return_value(self, t: int, T: int, rewards: list[int]) -> int:
        G = 0
        for k in range(t + 1, T):
            G += self.gamma ** (k - t - 1) * rewards[k]
        return G

    def _get_eligibility_vector(self):
        "TODO: Implement eligibility vector"
        return 0

    def _update_parameterized_policy(self, t: int, G: int, state):
        for parameter in self.theta:
            parameter += self.alpha * self.gamma ** t * G * self._get_eligibility_vector()

    def REINFORCE(self, T: int, steps: list[SAR]):
        print(f"steps:", steps)
        states = [step["state"] for step in steps]
        # actions = [step["action"] for step in steps]
        rewards = [step["reward"] for step in steps]
        for t in range(T):
            G = self._compute_return_value(t, T, rewards)
            self._update_parameterized_policy(t, G, states[t])

    def get_name(self):
        return self.name
