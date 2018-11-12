from scipy.stats import norm
import numpy as np
import math
import gym


class AngularMovement:

    stateDimension = 2
    actionDimension = 1

    def __init__(self, sigma_init=0.1, sigma_transition=0.1, target=[5, 5]):
        self.initSigma = sigma_init
        self.transitionSigma = sigma_transition
        self.targetState = target

    def init(self):
        return np.random.randn(2) * self.initSigma

    def transition(self, state, action):
        angle = action * 2 * math.pi
        x = state[0] + math.cos(angle) + np.random.randn() * self.transitionSigma
        y = state[1] + math.sin(angle) + np.random.randn() * self.transitionSigma

        return [x,y]

    def reward(self, states, _):
        if len(states.shape) == 1:
            states = states.reshape(1, -1)
        return math.exp(-sum((states[-1,:] - self.targetState) ** 2)/2)

    def step(self, state, action):

        next_state = self.transition(state, action)
        reward = self.reward(state, action)

        return next_state, reward, False, None

    def reset(self):
        return self.init()


class OpenAIEnvironment:

    stateDimension = None
    actionDimension = None
    alreadyFinished = False

    def __init__(self, environment):
        self.openAI = gym.make(self.environment)

    def reset(self):
        self.alreadyFinished = False
        return self.openAI.reset()

    def step(self, state, action):

        if self.alreadyFinished:
            return state, 0, True, None
        else:
            next_state, reward, self.alreadyFinished, info = self.openAI.step(int(action))  # TODO only discrete actions
            return next_state, reward, self.alreadyFinished, info


class CartPoleEnvironment(OpenAIEnvironment):

    environment = 'CartPole-v0'
    sigma = 0.01

    def __init__(self):
        super().__init__(self.environment)

        self.actionDimension = 1
        self.stateDimension = self.openAI.observation_space.shape[0]

    def step(self, state, action):
        if self.alreadyFinished:
            return state, 0, True, None
        else:
            next_state, reward, self.alreadyFinished, info = self.openAI.step(int(action))
            next_state += np.random.randn(self.stateDimension) * self.sigma
            return next_state, reward, self.alreadyFinished, info


class MountainCarEnvironment(OpenAIEnvironment):

    environment = 'MountainCar-v0'

    def __init__(self):
        super().__init__(self.environment)

        self.actionDimension = 1
        self.stateDimension = self.openAI.observation_space.shape[0]


class MountainCarReshapedEnvironment(MountainCarEnvironment):

    def __init__(self, reward_variance=1):
        super().__init__()

        self.rewardVariance = reward_variance

    def step(self, state, action):
        next_state, _, is_finished, info = super().step(state, action)
        reward = norm.pdf(state[0], loc=self.openAI.goal_position, scale=np.sqrt(self.rewardVariance))

        return next_state, reward, is_finished, info


class MountainCarTerminalReshapedEnvironment(MountainCarReshapedEnvironment):

    def __init__(self, horizon, reward_variance=1):
        super().__init__(reward_variance)
        self.horizon = horizon
        self.timeStep = 0

    def reset(self):
        self.timeStep = 0
        return super().reset()

    def step(self, state, action):
        next_state, reward, is_finished, info = super().step(state, action)
        self.timeStep += 1

        if self.timeStep < self.horizon and not is_finished:
            reward = 0

        return next_state, reward, is_finished, info
