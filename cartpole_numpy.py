import gym
from gym import spaces
import numpy as np
import tensorflow as tf

space = spaces.Discrete(8) # Set with 8 elements {0, 1, 2, ..., 7}
x = space.sample()

env = gym.make('CartPole-v0')

print(env.action_space)
#> Discrete(2)
print(env.observation_space)
#> Box(4,)

print(env.observation_space.high)
#> array([ 2.4       ,         inf,  0.20943951,         inf])
print(env.observation_space.low)
#> array([-2.4       ,        -inf, -0.20943951,        -inf])

#x = Box(4)
#y? = Discrete(2)
#a = Discrete(2)

#Observations
#x     = position => [-2.4,2.4]
#x_dot = dx / dt => linear velocity [-inf,inf]
#theta = angle [-.20943951,.20943951]
#theta_dot = dtheta / dt => angular velocity [-inf,inf]

#def cartpole_test_1():
#    parameters = np.random.rand(4) * 2 - 1  # 4d vector [-1, 1]
#    for i_episode in range(20):

#policy_parameters are going to be your action in this case... BECAUSE the policy is what you do given the state...
#state = observations
#action = actions


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return (e_x / e_x.sum(axis=0)).reshape(x.shape)


def forward_prop(W1, b1, W2, b2, X):
    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)

    Z2 = np.dot(W2, A1) + b2
    A2 = softmax(Z2)
    return A1, A2


def back_prop(action, X, A1, W2, A2, normalized_rewards):
    dZ2 = (A2 - action) * normalized_rewards #gradient of A2 from objective being the rewards

    dW2 = np.dot(dZ2, A1.T)
    db2 = np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.multiply(np.dot(W2.T, dZ2), 1 - np.power(A1, 2))
    dW1 = np.dot(dZ1, X.T)
    db1 = np.sum(dZ1, axis=1, keepdims=True)

    m = X.shape[1]
    return dW1 / m, db1 / m, dW2 / m, db2 / m

learning_rate = 1.2
gamma = 0.97


def update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2):
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2
    return W1, b1, W2, b2


def calc_rewards(rewards, discount_rate):
    #new_rewards = []
    new_rewards = np.empty(shape=(1,len(rewards)))
    for index in range(len(rewards)):
        future_reward = 0
        for j in range(len(rewards) - index):
            reward = rewards[j + index]
            very_discounted = discount_rate ** j
            temp_reward = very_discounted * reward
            future_reward += temp_reward
        new_rewards[0][index] = future_reward
    return new_rewards


# Define the cost function
def compute_cost(predicted, actual):
    return - np.sum(np.multiply(predicted, np.log(actual)) + np.multiply((1-predicted), np.log(1-actual)))


def run_episode(env, W1, b1, W2, b2, do_render, episode_batch_size):
    all_steps = []
    episode_bags = []
    for episode_number in range(episode_batch_size): #number of episodes per mini batch
        episode_bag = run_steps(env, W1, W2, b1, b2, do_render if episode_number == 0 else False)
        all_steps.append(episode_bag.discounted_rewards.shape[1])
        episode_bags.append(episode_bag)

    normalize_rewards_over_episodes(episode_bags)
    #Trying out applying gradients over an entire episode at a time
    dW1, db1, dW2, db2 = run_back_prop_over_episodes(W1, b1, W2, b2, episode_bags)
    W1, b1, W2, b2 = update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2)
    return W1, b1, W2, b2, all_steps


def run_back_prop_over_episodes(og_W1, og_b1, og_W2, og_b2, episode_bags):
    gradients = []

    for episode_bag in episode_bags:
        dW1, db1, dW2, db2 = back_prop(episode_bag.action_vector_m, episode_bag.obs_vector_m, episode_bag.A1_m,
                                       og_W2, episode_bag.A2_m, episode_bag.normalized_rewards)
        return dW1, db1, dW2, db2


class ForwardPropBag:
    def __init__(self, A1, A2, W1, b1, W2, b2, obs_vector, action_vector, reward):
        self.A1 = A1
        self.A2 = A2
        self.W1 = W1
        self.b1 = b1
        self.W2 = W2
        self.b2 = b2
        self.obs_vector = obs_vector
        self.action_vector = action_vector
        self.reward = reward


class EpisodeBag:
    def __init__(self, fpb, gamma):
        self.step_bags = fpb
        self.action_vector_m = concat_bagged(lambda bag: bag.action_vector, fpb)
        av2 = np.vstack(map(lambda bag: bag.action_vector, fpb))
        self.obs_vector_m = concat_bagged(lambda bag: bag.obs_vector, fpb)
        self.A1_m = concat_bagged(lambda bag: bag.A1, fpb)
        self.A2_m = concat_bagged(lambda bag: bag.A2, fpb)

        #calculate discounted rewards
        rewards = list(map(lambda bag: bag.reward, fpb))
        self.discounted_rewards = calc_rewards(rewards, gamma)
        self.discounted_rewards_sum = np.sum(self.discounted_rewards)

        self.normalized_rewards = None
        self.normalized_action_vector = None

    def normalize_rewards(self, mean, std):
        normalized_rewards = self.discounted_rewards - mean
        self.normalized_rewards = normalized_rewards / std
        #self.normalized_action_vector = self.action_vector_m * normalized_rewards
        #self.grad_on_existing_actions =  self.action_vector_m - self.A2_m


def run_steps(env, W1, W2, b1, b2, do_render):
    observation = env.reset()
    fpb = []
    for step_num in range(10000):
        obs_vector = np.expand_dims(a=observation, axis=1).reshape(-1, 1)
        A1, A2 = forward_prop(W1, b1, W2, b2, obs_vector)

        probs_of_up_or_down = A2
        # throws in some randomization

        action = 0 if np.random.uniform(0, 1) < probs_of_up_or_down[0][0] else 1

        action_vector = np.zeros((2, 1))
        action_vector[action] = 1

        if do_render:
            env.render()
        new_obs, reward, done, info = env.step(action)

        fpb.append(ForwardPropBag(A1, A2, W1, b1, W2, b2, obs_vector, action_vector, reward))

        observation = new_obs
        if done:
            break

    ####### This is the same as the back_prop above in the loop #######
    #s = sum(map(lambda g: g[0][0], gradients)) * 1.0
    #c = len(gradients) * 1.0
    #a = s / c

    episode_bag = EpisodeBag(fpb, gamma)

    #dW1, db1, dW2, db2 = back_prop(action_vector_m, obs_vector_m, A1_m, W2_m, A2_m)
    return episode_bag


def concat_bagged(bag_func, fpb):
    l = list(map(bag_func, fpb))
    return np.concatenate(l, axis=1)


def normalize_rewards_over_episodes(episode_bags):
    ravelled_rewards = np.concatenate(list(map(lambda bag: bag.discounted_rewards, episode_bags)), axis = 1)
    mean = np.mean(ravelled_rewards)
    std = np.std(ravelled_rewards)

    for episode_bag in episode_bags:
        episode_bag.normalize_rewards(mean, std)


def run_episodes():
    n_h_1 = 5
    W1 = np.random.randn(n_h_1, 4).astype(np.float32) * np.sqrt(2.0 / 4)
    b1 = np.zeros(shape=(n_h_1, 1))
    W2 = np.random.randn(2, n_h_1).astype(np.float32) * np.sqrt(2.0 / n_h_1)
    b2 = np.zeros(shape=(2, 1))

    do_render = False
    all_steps = []

    # Not working with more than one right now
    episodes_per_batch = 1

    render_episode_counter = 0
    episode_number = 0
    for episode_batch_counter in range(100000):
        W1, b1, W2, b2, all_steps_on_episode = run_episode(env, W1, b1, W2, b2, do_render, episodes_per_batch)
        all_steps.append(all_steps_on_episode)
        render_episode_counter += episodes_per_batch
        episode_number += episodes_per_batch

        if (episode_number <= 500 and render_episode_counter >= 50) or (render_episode_counter >= 200) :
            render_episode_counter = 0
            print('episode_number:  ' + str(episode_number))
            print('all_steps:       ' + str(np.mean(all_steps)))
            all_steps = []
            do_render = True
        else:
            do_render = False

run_episodes()