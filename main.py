import gym
import random
from collections import deque
from keras import  Sequential
from keras.layers import Dense
import numpy as np
from keras.optimizers import Adam


MEMORY_SIZE = 5_000
BATCH_SIZE = 16
MIN_MEMORY_TO_TRAIN = BATCH_SIZE * 3
EPISODE_TO_RUN = 500
LEARNING_RATE = 0.001
UPDATE_TARGET_EVERY = 10
RENDER_EVERY = 50

GAMMA = 0.95
START_EPSILON = 1
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.01
env = gym.make("CartPole-v1")
observation_space = env.observation_space.shape[0]
action_space = env.action_space.n


class DoubleQKN:
    def __init__(self):
        self.memory = deque(maxlen=MEMORY_SIZE)

        self.model = self.create_network()
        self.target_model = self.create_network()
        self.target_model.set_weights(self.model.get_weights())

    def create_network(self):
        model = Sequential()

        model.add(Dense(24, input_shape=(observation_space,), activation='relu'))
        model.add(Dense(24,  activation='relu'))
        model.add(Dense(action_space, activation='linear'))

        optimizer = Adam(lr=LEARNING_RATE)
        model.compile(loss='mse', optimizer=optimizer)
        return model

    def predict(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape))

    def update_memory(self, data): # data = (state, action, reward, next_state, done)
        self.memory.append(data)

    def train(self, episode_number):
        if len(self.memory) < MIN_MEMORY_TO_TRAIN:
            return

        batch_data = random.sample(self.memory, BATCH_SIZE)

        curr_state = [x[0] for x in batch_data]
        curr_qs = self.model.predict(np.array(curr_state))

        next_state = [x[3] for x in batch_data]
        next_qs = self.target_model.predict(np.array(next_state))

        Y = []
        X = []
        for index, data in enumerate(batch_data):
            state, action, reward, next_state, done = data

            new_q = reward
            if not done:
                future_reward = next_qs[index]
                new_q += GAMMA*np.max(future_reward)

            y = curr_qs[index]
            y[action] = new_q
            X.append(state)
            Y.append(y)

        self.model.fit(x=np.array(X), y=np.array(Y), batch_size=BATCH_SIZE, verbose=0)

        if episode_number % UPDATE_TARGET_EVERY == 0:
            self.target_model.set_weights(self.model.get_weights())


if __name__ == "__main__":
    total_rewards = []
    steps = []
    epsilon = START_EPSILON

    agent = DoubleQKN()
    for episode in range(EPISODE_TO_RUN):
        render = False
        if RENDER_EVERY > 0 and (episode % RENDER_EVERY) == 0:
            render = True
        done = False
        state = env.reset()
        total_reward = 0
        step = 0
        while not done:
            if random.random() < epsilon:
                action = random.randrange(action_space)
            else:
                action = np.argmax(agent.predict(state))

            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            reward = reward if not done else -reward
            agent.update_memory((state, action, reward, next_state, done))
            agent.train(episode)
            state = next_state
            step += 1
            if render:
                env.render()
            if done:
                total_rewards.append(total_reward)
                steps.append(step)
                print("episode: {0}     reward: {1}".format(episode, total_reward))
                break

        if epsilon > MIN_EPSILON:
            epsilon *= EPSILON_DECAY