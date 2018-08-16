import gym
import numpy as np
import tensorflow as tf

trace_episode_count_num = 200
num_hidden_layers = 10
gamma = 0.97


class EpisodeBag:
    def __init__(self, steps):
        """
        :type steps: list[StepBag]
        """
        self.observations = EpisodeBag.concat_bagged(lambda bag: bag.observation_vector, steps, 0)
        self.actions = EpisodeBag.concat_bagged(lambda bag: bag.action_vector, steps)

        # calculate discounted rewards
        rewards = list(map(lambda bag: bag.reward, steps))
        self.discounted_rewards = EpisodeBag.calc_rewards(rewards)

        self.normalized_rewards = None
        self.normalized_actions = None

    @staticmethod
    def concat_bagged(bag_func, l, axis=1):
        l = list(map(bag_func, l))
        return np.concatenate(l, axis=axis)

    @staticmethod
    def calc_rewards(rewards):
        new_rewards = np.empty(shape=(1, len(rewards)))
        for index in range(len(rewards)):
            future_reward = 0
            for j in range(len(rewards) - index):
                reward = rewards[j + index]
                temp_reward = (gamma ** j) * reward
                future_reward += temp_reward
            new_rewards[0][index] = future_reward
        return new_rewards

    def normalize_rewards(self, mean, std):
        normalized_rewards = self.discounted_rewards - mean
        self.normalized_rewards = normalized_rewards / std
        self.normalized_actions = self.actions * self.normalized_rewards
        pass


class StepBag:
    def __init__(self, observation_vector, action_vector, reward):
        self.action_vector = action_vector
        self.observation_vector = observation_vector
        self.reward = reward


class GraphBag:
    def __init__(self, prob, actions, X, grads_and_vars, apply_updates, loss):
        self.prob = prob
        self.actions = actions
        self.X = X
        self.grads_and_vars = grads_and_vars
        self.apply_updates = apply_updates
        self.loss = loss


def run_episodes():
    env = gym.make('CartPole-v0')
    graph_bag = create_graph(2)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        do_render = False
        all_steps = []
        mean_losses = []
        episodes_per_batch = 1
        render_episode_counter = 0
        episode_number = 0
        for episode_batch_counter in range(100000):
            all_steps_on_episode, mean_loss = run_episode_batch(env, sess, graph_bag, do_render, episodes_per_batch)
            all_steps.append(all_steps_on_episode)
            mean_losses.append(mean_loss)
            render_episode_counter += episodes_per_batch
            episode_number += episodes_per_batch

            if render_episode_counter >= trace_episode_count_num:
                render_episode_counter = 0
                print('episode number:  ' + str(episode_number))
                print('steps passed:    ' + str(np.mean(all_steps)))
                print('mean loss:       ' + str(mean_loss))
                print('[mean loss seems broke]')
                all_steps = []
                mean_losses = []
                do_render = True
            else:
                do_render = False


def create_graph(action_count):
    X = tf.placeholder(name='X', dtype=tf.float32, shape=(None, 4))

    W1 = tf.get_variable(name='W1', shape=(4, num_hidden_layers), dtype=tf.float32,
                         initializer=tf.contrib.layers.xavier_initializer())
    b1 = tf.get_variable(name='b1', shape=(1, num_hidden_layers), dtype=tf.float32,
                         initializer=tf.zeros_initializer())
    Z1 = tf.add(tf.matmul(X, W1), b1, name='Z1')
    A1 = tf.nn.relu(Z1, name='A1')

    W2 = tf.get_variable(name='W2', shape=(num_hidden_layers, action_count), dtype=tf.float32,
                         initializer=tf.contrib.layers.xavier_initializer())
    b2 = tf.get_variable(name='b2', shape=(1, action_count), dtype=tf.float32,
                         initializer=tf.zeros_initializer())
    Z2 = tf.add(tf.matmul(A1, W2), b2, name='Z2')

    prob = tf.nn.softmax(Z2)
    actions = tf.placeholder(name='actions', dtype=tf.float32, shape=(None, action_count))
    loss = tf.reduce_mean(-tf.reduce_sum(actions * tf.log(prob), [1]))

    optimizer = tf.train.AdamOptimizer()
    grads_and_vars = optimizer.compute_gradients(loss)
    apply_updates = optimizer.apply_gradients(grads_and_vars)

    graph_bag = GraphBag(prob, actions, X, grads_and_vars, apply_updates, loss)
    return graph_bag


def run_episode_batch(env, sess, graph_bag, do_render, episode_batch_size):
    all_steps = []
    episode_bags = []
    """
    :type episode_bags: list[EpisodeBag]
    """
    for episode_number in range(episode_batch_size):
        episode_bag = run_steps(env, sess, graph_bag, do_render if episode_number == 0 else False)
        all_steps.append(episode_bag.discounted_rewards.shape[1])
        episode_bags.append(episode_bag)

    normalize_rewards_over_episodes(episode_bags)

    losses = np.empty(shape=(len(episode_bags)))
    for index, episode_bag in enumerate(episode_bags):
        losses[index] = back_prop(sess, graph_bag, episode_bag)
    mean_loss = np.mean(losses)
    return all_steps, mean_loss


def back_prop(sess, graph_bag, episode_bag):
    """
    :type sess: tf.Session
    :type graph_bag: GraphBag
    :type episode_bag: EpisodeBag
    """
    _, loss = sess.run([graph_bag.apply_updates, graph_bag.loss], feed_dict={
        graph_bag.X: episode_bag.observations,
        graph_bag.actions: episode_bag.normalized_actions.T
    })
    return loss


def run_steps(env, sess, graph_bag, do_render):
    """
    :type env: gym.Env
    :type sess: tf.Session
    :type graph_bag: GraphBag
    :type do_render: bool
    """
    observation = env.reset()
    step_bags = []
    for step_num in range(10000):
        obs_vector = np.expand_dims(a=observation, axis=1).reshape(1, -1)
        probs_of_up_or_down = sess.run([graph_bag.prob], feed_dict={
            graph_bag.X: obs_vector
        })

        action = 0 if np.random.uniform(0, 1) < probs_of_up_or_down[0][0][0] else 1
        action_vector = np.zeros((2, 1))
        action_vector[action] = 1

        if do_render:
            env.render()
        new_obs, reward, done, info = env.step(action)
        step_bags.append(StepBag(obs_vector, action_vector, reward))
        observation = new_obs
        if done:
            break
    return EpisodeBag(step_bags)


def normalize_rewards_over_episodes(episode_bags):
    """
    :type episode_bags: list[EpisodeBag]
    """
    ravelled_rewards = np.concatenate(list(map(lambda bag: bag.discounted_rewards, episode_bags)), axis=1)
    mean = np.mean(ravelled_rewards)
    std = np.std(ravelled_rewards)

    for episode_bag in episode_bags:
        episode_bag.normalize_rewards(mean, std)

run_episodes()