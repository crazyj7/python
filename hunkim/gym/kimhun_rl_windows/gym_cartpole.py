# Cartpole implementation.
# It just tries random parameters, and picks the first one that gets a 200 score.
# Runs on Python 3.
# You can switch submitting on and off.
# By Tom Jacobs

import gym
import numpy as np
import matplotlib.pyplot as plt

def run_episode(env, parameters):
    observation = env.reset()

    # Run 200 steps and see what our total reward is
    total_reward = 0
    for t in range(200):

        # Show us what's going on. Remove this line to run super fast.
        env.render()

        # Pick action
        action = 0 if np.matmul(parameters, observation) < 0 else 1

        # Step
        observation, reward, done, info = env.step(action)
        total_reward += reward

        # Done?
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
    return total_reward

def train(submit):
    env = gym.make('CartPole-v0')
    if submit == True:
        env = gym.wrappers.Monitor(env, 'cartpole', force=True)

    # Run lots of episodes with random params, and find best_params
    results = []
    counter = 0
    best_parameters = None
    best_reward = 0
    for t in range(1000):

        # Pick random parameters and run
        parameters = np.random.rand(4) * 2 - 1
        reward = run_episode(env, parameters)
        results.append(reward)
        counter += 1

        # Did this one do better?
        if reward > best_reward:
            best_reward = reward
            best_parameters = parameters

            # And did we win the world?
            if reward == 200:
                print("Win!")
                break

    # Run 100 runs with the best found params
    for t in range(100):
        reward = run_episode(env, best_parameters)
        results.append(reward)
        print( "Episode " + str(t) )

    return results

# Submit it or view it
submit = False
results = train(submit=submit)
if submit == True:
    # Submit to OpenAI Gym
    print("Uploading to gym...")
    gym.scoreboard.api_key = '' # Put your key here
    print("Results: " + str( gym.upload('cartpole')) )

else:
    # Graph
    plt.hist(results, 50, normed=1, facecolor='g', alpha=0.75)
    plt.xlabel('')
    plt.ylabel('')
    plt.title('')
    plt.show()

