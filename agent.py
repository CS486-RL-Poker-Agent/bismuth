from typekit import SAR


class Agent:
    def __init__(self, alpha: float, gamma: float) -> None:
        self.alpha = alpha
        self.gamma = gamma


class REINFORCEAgent(Agent):
    def __init__(self, alpha: float, gamma: float, theta=[0]) -> None:
        super().__init__(alpha, gamma)
        self.theta = theta

    def computeReturnValue(self, t: int, T: int, rewards: list[int]) -> int:
        G = 0
        for k in range(t + 1, T):
            G += self.gamma ** (k - t - 1) * rewards[k]
        return G

    def getEligibilityVector(self):
        return 0

    def updatePolicyParameters(self, t: int, G: int, state):
        "TODO"
        # for parameter in self.theta:
        #     parameter += self.alpha * self.gamma ** t * G * self.getEligibilityVector()

    def REINFORCE(self, T: int, steps: list[SAR]):
        states = [step["state"] for step in steps]
        actions = [step["action"] for step in steps]
        rewards = [step["reward"] for step in steps]
        for t in range(T):
            G = self.computeReturnValue(t, T, rewards)
            self.updatePolicyParameters(t, G, states[t])
