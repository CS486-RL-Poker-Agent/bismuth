from pettingzoo.classic import texas_holdem_no_limit_v6
from agent import REINFORCEAgent
from custom_types import Step


def generate_episode(episode_agent: REINFORCEAgent):
    T = 0
    steps: list[Step] = []
    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()

        if termination or truncation:
            action = None
        else:
            mask = observation["action_mask"]
            # this is where you would insert your policy
            action = env.action_space(agent).sample(mask)

        if (agent == episode_agent.getName()):
            steps.append({"state": observation, "action": action})
            if steps[-1]:
                steps[-1]["reward"] = reward
        print("Current steps for player_0:", steps)
        env.step(action)
        T += 1
    episode_agent.REINFORCE(T, steps)
    env.close()


if __name__ == "__main__":
    env = texas_holdem_no_limit_v6.env(render_mode="human")
    env.reset(seed=42)

    epochs = 100
    agent = REINFORCEAgent(0.01, 0.99)

    for _ in range(epochs):
        generate_episode(agent)
