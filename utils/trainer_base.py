from abc import ABC, abstractmethod

from conf import Conf


class TrainerBase(ABC):

    def __init__(self, cnf: Conf):
        self.cnf = cnf
        self.current_epoch = 0

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
        # self.test()
        # exit()
        for e in range(self.current_epoch, self.cnf.epochs):
            if self.cnf.epoch_len > 0:
                self.train()
            # if e % 10 == 0 and e != 0:
            if (self.cnf.is_windows and self.current_epoch % 64 == 0) or \
                (not self.cnf.is_windows and self.cnf.test_len > 0):
                self.test()
            self.current_epoch += 1
            if (self.cnf.is_windows and self.current_epoch % 512 == 0) or not self.cnf.is_windows:
                self.save_ck()
