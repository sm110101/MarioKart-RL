import gymnasium as gym
from src.envs.mario_kart_yoshi_falls_env import MarioKartYoshiFallsEnv

def main():
    env = MarioKartYoshiFallsEnv()
    obs, info = env.reset()
    print(obs.shape, obs.dtype, info.get("raw_state"))
    for _ in range(15):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        print(obs, reward, terminated, truncated)

if __name__ == "__main__":
    main()
