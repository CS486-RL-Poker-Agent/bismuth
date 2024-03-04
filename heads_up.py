from pettingzoo.classic import texas_holdem_no_limit_v6
from agent import REINFORCEAgent
from custom_types import Step


def generate_episode(episode_agent: REINFORCEAgent):
    env = texas_holdem_no_limit_v6.env(render_mode="human")
    env.reset(seed=42)

    T = 0
    steps: list[Step] = []
    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()

        if termination or truncation:
            action = None
        else:
            mask = observation["action_mask"]
            action = env.action_space(agent).sample(mask)

        if (agent == episode_agent.get_name()):
            state = observation["observation"]
            # print(f"State @ step {T}", state)
            steps.append({"state": state, "action": action})
            if steps[-1]:
                steps[-1]["reward"] = reward

        env.step(action)
        T += 1
    episode_agent.REINFORCE(T, steps)
    env.close()


def train():
    agent = REINFORCEAgent(0.01, 0.99)
    for _ in range(100):
        generate_episode(agent)
    print("Final parameterized policy:", agent.get_theta())


if __name__ == "__main__":
    train()
