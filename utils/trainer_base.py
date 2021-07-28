from abc import ABC, abstractmethod

from conf import Conf


class TrainerBase(ABC):

    def __init__(self, cnf: Conf):
        self.cnf = cnf
        self.epoch = 0

    @abstractmethod
    def load_ck(self):
        pass

    @abstractmethod
    def save_ck(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def test(self):
        pass

    def run(self):
        """
        start model training procedure (train > test > checkpoint > repeat)
        """
        for e in range(self.epoch, self.cnf.epochs):
            self.train()
            # if e % 10 == 0 and e != 0:
            self.test()
            self.epoch += 1
            self.save_ck()
