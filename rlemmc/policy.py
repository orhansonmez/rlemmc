import numpy as np
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import svm, linear_model

from rlemmc.trajectory import extract_trajectory

import abc


class Policy(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def query(self):
        pass

    @abc.abstractmethod
    def m_step(self):
        pass


class UniformPolicy(Policy):

    def __init__(self):
        raise NotImplementedError()

    def m_step(self, _1, _2):
        pass


class UniformPolicyContinuous(UniformPolicy):

    def __init__(self, bounds=[0,1]):
        self.min = bounds[0]      # Min Action Value
        self.max = bounds[1]      # Max Action Value

    def query(self, _):
        return self.min + np.random.random() * (self.min - self.max)


class UniformPolicyDiscrete(UniformPolicy):

    def __init__(self, choices):
        self.choices = choices

    def query(self, _):
        return np.random.choice(self.choices)


class SciKitPolicy(Policy):

    def __init__(self):
        raise NotImplementedError()

    def query(self, states):
        if len(states.shape) == 1:
            states = states.reshape(1, -1)
        return self.method.predict(states)

    def train(self, inputs, targets):
        self.method.fit(inputs, targets)

    def m_step(self, states, actions):

        # States/Actions -> Inputs/Targets
        inputs, targets = extract_trajectory(states, actions)

        # Train kNN
        self.train(inputs, targets.ravel())


class KnnPolicyContinuous(SciKitPolicy):
    def __init__(self, k, weights='distance'):
        self.method = KNeighborsRegressor(n_neighbors=k, weights=weights, n_jobs=1)


class KnnPolicyDiscrete(SciKitPolicy):
    def __init__(self, k, weights='distance'):
        self.method = KNeighborsClassifier(n_neighbors=k, weights=weights, n_jobs=1)


class LinearPolicyLDA(SciKitPolicy):
    def __init__(self):
        self.method = LinearDiscriminantAnalysis(solver='svd')


class LinearPolicySVC(SciKitPolicy):
    def __init__(self):
        self.method = svm.LinearSVC()


class LogisticRegressionPolicy(SciKitPolicy):
    def __init__(self):
        self.method = linear_model.LogisticRegression()