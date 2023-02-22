import numpy as np


class Model:
    def u(self, x):
        return -np.log(self.pr(x))

    def pr(self, q):
        raise

    def grad_u(self, q):
        raise

    def grad_pr(self, x):
        return -self.pr(x) * self.grad_u(x)

    @staticmethod
    def name():
        raise
