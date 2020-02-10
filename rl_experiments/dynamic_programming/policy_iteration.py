import gym
import logging


def main():

    env = gym.make('Taxi-v3')

    print(env.P)

    # for i_episode in range(20):
    #     observation = env.reset()
    #     for t in range(100):
    #         print(observation)
    #         action = env.action_space.sample()
    #         observation, reward, done, info = env.step(action)
    #         print(info)
    #         if done:
    #             print("Episode finished after {} timesteps".format(t+1))
    #             break
    env.close()

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)

    main()