from abc import ABC, abstractmethod


class AbstractStepper(ABC):
    @abstractmethod
    def update(self, gt):
        pass


class ConstantLearningRate(AbstractStepper):
    def __init__(self, lr):
        self.lr = lr

    def update(self, gt):
        return self.lr * gt


class ConstantLearningRateDecay(AbstractStepper):
    def __init__(self, lr, lrd=0.5, period=5):
        self.lr = lr
        self.lrd = lrd
        self.period = period
        self.t = 0

    def update(self, gt):
        self.t += 1
        if self.t % self.period == 0:
            self.lr = self.lr * self.lrd
        return self.lr * gt
