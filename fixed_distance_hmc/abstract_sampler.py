INFO_LEAPFROG_PS = 'info.leapfrog.ps'

INFO_LEAPFROG_QS = 'info.leapfrog.qs'

INFO_NUM_GRADS = 'info.num.grads'

INFO_ACCEPT_PROB = 'info.accept.prob'

INFO_INIT_EVOLVE_Q = 'info.fd.init.evolve.q'

INFO_FINAL_EVOLVE_Q = 'info.fd.final.evolve.q'

class Sampler:
    def __init__(self, model, name, alias, color='k'):
        self.model = model
        self._name = name
        self.alias = alias
        self.color = color
        self._activate = False

    def activate(self):
        self._activate = True

    def next_sample(self, current_sample):
        if not self._activate:
            raise Exception('the sampler should be activated before the first sample is taken')

        sample, info = self.next_sample_info(current_sample=current_sample)
        return sample

    def next_sample_info(self, current_sample):
        if not self._activate:
            raise Exception('the sampler should be activated before the first sample is taken')

        raise Exception('not implemented')

    def name(self):
        return self._name
