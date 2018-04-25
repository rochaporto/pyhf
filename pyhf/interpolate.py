import logging
log = logging.getLogger(__name__)

from . import get_backend
from . import exceptions

def _hfinterp_code0(at_minus_one, at_zero, at_plus_one, alphas):
    tensorlib, _ = get_backend()
    at_minus_one = tensorlib.astensor(at_minus_one)
    at_zero = tensorlib.astensor(at_zero)
    at_plus_one = tensorlib.astensor(at_plus_one)
    alphas = tensorlib.astensor(alphas)

    iplus_izero  = at_plus_one - at_zero
    izero_iminus = at_zero - at_minus_one
    mask = tensorlib.outer(alphas < 0, tensorlib.ones(iplus_izero.shape))
    return tensorlib.where(mask, tensorlib.outer(alphas, izero_iminus), tensorlib.outer(alphas, iplus_izero))

def _prep_code1(at_minus_one, at_zero, at_plus_one):
    tensorlib, _ = get_backend()
    at_minus_one = tensorlib.astensor(at_minus_one)
    at_zero = tensorlib.astensor(at_zero)
    at_plus_one = tensorlib.astensor(at_plus_one)

    base_positive = tensorlib.divide(at_plus_one,  at_zero)
    base_negative = tensorlib.divide(at_minus_one, at_zero)
    positive_ones = tensorlib.ones(base_positive.shape)
    return positive_ones, base_positive, base_negative, tensorlib

def _code1_from_prepped(alphas, positive_ones, base_positive, base_negative, tensorlib):
    expo_positive = tensorlib.outer(alphas, positive_ones)
    mask          = tensorlib.outer(alphas > 0, positive_ones)
    bases         = tensorlib.where(mask,base_positive,base_negative)
    exponents     = tensorlib.where(mask, expo_positive,-expo_positive)
    return tensorlib.power(bases, exponents)

def _hfinterp_code1(at_minus_one, at_zero, at_plus_one, alphas):
    return _code1_from_prepped(alphas,*_prep_code1(at_minus_one, at_zero, at_plus_one))

# interpolation codes come from https://cds.cern.ch/record/1456844/files/CERN-OPEN-2012-016.pdf
def interpolator(interpcode):
    interpcodes = {0: _hfinterp_code0,
                   1: _hfinterp_code1}
    try:
        return interpcodes[interpcode]
    except KeyError:
        raise exceptions.InvalidInterpCode

def fromprep_interpolate(interpcode, alphas, *prepdata):
    fromprepped = {1: _code1_from_prepped}
    try:
        return fromprepped[interpcode](alphas,*prepdata)
    except KeyError:
        raise exceptions.InvalidInterpCode

def prep_interpolator(interpcode, at_minus_one, at_zero, at_plus_one):
    prepfuncs = {1: _prep_code1}
    try:
        return prepfuncs[interpcode](at_minus_one, at_zero, at_plus_one)
    except KeyError:
        raise exceptions.InvalidInterpCode
