class Agent:
    def __init__(self, alpha: float, gamma: float) -> None:
        self.alpha = alpha
        self.gamma = gamma


class REINFORCEAgent(Agent):
    def __init__(self, alpha: float, gamma: float) -> None:
        super().__init__(alpha, gamma)

    def computeReturnValue(self, t: int, T: int, rewards: list[int]) -> int:
        G = 0
        for k in range(t + 1, T):
            G += self.gamma ** (k - t - 1) * rewards[k]
        return G

    def REINFORCE(self, T: int, rewards: list[int]):
        for t in range(T):
            G = self.computeReturnValue(t, T, rewards)
