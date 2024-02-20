from pettingzoo.classic import texas_holdem_no_limit_v6
from agent import REINFORCEAgent
from typekit import SAR


def episode(episode_agent: REINFORCEAgent):
    T = 0
    steps: list[SAR] = []
    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()

        if steps:
            steps[-1]["reward"] = reward

        if termination or truncation:
            action = None
        else:
            mask = observation["action_mask"]
            # this is where you would insert your policy
            action = env.action_space(agent).sample(mask)

        steps.append({"observation": observation, "action": action})
        env.step(action)
        T += 1
    episode_agent.REINFORCE(T, steps)
    env.close()


if __name__ == "__main__":
    env = texas_holdem_no_limit_v6.env(render_mode="human")
    env.reset(seed=42)

    agent = REINFORCEAgent(0.01, 0.99)
    episode(agent)
