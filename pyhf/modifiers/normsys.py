import logging
log = logging.getLogger(__name__)

from . import modifier
from .. import get_backend
from ..interpolate import prep_interpolator, fromprep_interpolate

@modifier(name='normsys', constrained=True, shared=True)
class normsys(object):
    def __init__(self, nom_data, modifier_data):
        tensorlib, _ = get_backend()
        self.n_parameters     = 1
        self.suggested_init   = [0.0]
        self.suggested_bounds = [[-5, 5]]

        self.at_zero = tensorlib.astensor([1])
        self.at_minus_one = {}
        self.at_plus_one = {}
        self.auxdata = [0]  # observed data is always at a = 1

        self.prepdata = {}


    def add_sample(self, channel, sample, modifier_def):
        tensorlib, _ = get_backend()
        log.info('Adding sample {0:s} to channel {1:s}'.format(sample['name'], channel['name']))
        self.at_minus_one.setdefault(channel['name'], {})[sample['name']] = tensorlib.astensor([modifier_def['data']['lo']])
        self.at_plus_one.setdefault(channel['name'], {})[sample['name']]  = tensorlib.astensor([modifier_def['data']['hi']])
        self.prepdata.setdefault(channel['name'], {})[sample['name']]  = prep_interpolator(1,
            self.at_minus_one[channel['name']][sample['name']],
            self.at_zero,
            self.at_plus_one[channel['name']][sample['name']]
        )

    def alphas(self, pars):
        return pars  # the nuisance parameters correspond directly to the alpha

    def expected_data(self, pars):
        return self.alphas(pars)

    def pdfpars(self, pars):
        tensorlib, _ = get_backend()
        return pars, tensorlib.astensor([1]) #mu sigma

    def pdf(self, a, alpha):
        tensorlib, _ = get_backend()
        return tensorlib.normal(a, alpha, tensorlib.astensor([1]))

    def apply(self, channel, sample, pars):
        # normsysfactor(nom_sys_alphas)   = 1 + sum(interp(1, anchors[i][0], anchors[i][0], val=alpha)  for i in range(nom_sys_alphas))
        assert int(pars.shape[0]) == 1
        return fromprep_interpolate(1,pars,*self.prepdata[channel['name']][sample['name']])[0]
