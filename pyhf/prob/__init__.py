from ..tensor import tensorlib

def logpdf_joint_indep_poissons(data,rates):
    '''
    a N-variate pmf representing the joint probability
    of N poisson processes

    data:  a (..., N) shaped array representing the counts for each
           of the N poisson processes
    rates: a (..., N) shaped array representing the rates for each of
           the N poisson processes
    '''
    summands = tensorlib.log(tensorlib.poisson(data,rates))
    return tensorlib.sum(summands)
