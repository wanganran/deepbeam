import numpy as np

def si_sdr(estimated_signal, reference_signals, scaling=True):
    """
    This is a scale invariant SDR. See https://arxiv.org/pdf/1811.02508.pdf
    or https://github.com/sigsep/bsseval/issues/3 for the motivation and
    explanation
    Input:
        estimated_signal and reference signals are (N,) numpy arrays
    Returns: SI-SDR as scalar
    """
    Rss = np.dot(reference_signals, reference_signals)
    this_s = reference_signals

    if scaling:
        # get the scaling factor for clean sources
        a = np.dot(this_s, estimated_signal) / Rss
    else:
        a = 1

    e_true = a * this_s
    e_res = estimated_signal - e_true

    Sss = (e_true**2).sum()
    Snn = (e_res**2).sum()

    SDR = 10 * math.log10(Sss/Snn)

    return SDR

def db(x):
    return 10*np.log10(x)

def dbto(x):
    return 10**(x/10)

def power(signal):
    return np.sum(np.abs(signal)**2)/signal.size

def normalize(signal): # keep a std of 0.1 and min/max of +-1
    return np.clip(signal/np.std(signal)*0.1, -1, 1)

def mix(signal, noise, target_snr_db):
    psig=power(signal)
    pnoi=power(noise)
    newnoise=noise*np.sqrt(dbto(db(psig/pnoi)-target_snr_db))
    res=signal+newnoise
    return res, newnoise