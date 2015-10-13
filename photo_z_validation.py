import pandas as pd 
import numpy as np

TEST_DATA = pd.read_csv('test_df.csv')
TEST_BINNING = z = np.linspace(0, 2.0, 100)


def _bias(z_spec, z_phot, weight=None):
    dz1 = (z_spec-z_phot)
    if weight is None:
        bias = np.mean(dz1)
    else:
        bias = np.average(dz1, weights=weight)

    return bias


def _std68(z_spec, z_phot):
    dz1 = (z_spec - z_phot)
    calc68 = 0.5 * (_percentile(dz1, 0.84) - _percentile(dz1, 0.16))
    return calc68


def _sigma(z_spec, z_phot, weight=None):
    dz1 = (z_spec-z_phot)
    if weight is None:
        sigma = np.std(dz1)
    else:
        sigma = _w_std(dz1, weights=weight)
    return sigma


def _percentile(n, percent):
    n = np.sort(n)
    k = (len(n)-1) * percent
    f = np.floor(k)
    c = np.ceil(k)
    if f == c:
        return n[int(k)]
    d0 = n[int(f)] * (c-k)
    d1 = n[int(c)] * (k-f)
    return d0+d1


def _w_std(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    variance = np.average((np.abs(values - average))**2, weights=weights)  # Fast and numerically precise
    return np.sqrt(variance)


def _normalize_pdf(pdf, dz):
    """
    returns normalized pdf
    """
    area = np.trapz(pdf, dx=dz)
    return pdf/area


def mean(df, binning, metric='mean', weights=None, tomo_bins=None, z_phot=None):
    """
    :param df: pandas data-frame
    :param weight: optional weighting scheme
    :param tomo_bins: in which z-bins
    :return: pandas data frame with mean estimates
    """

    assert isinstance(df, pd.DataFrame), 'df must be a pandas DataFrame'
    assert isinstance(binning, np.ndarray), 'binning must be a numpy array'
    if weights:
        assert weights in df.columns,  str(weights) + ' not in df.columns'
        weights = weights/weights.sum() # normalize weights
    if tomo_bins:
        assert isinstance(z_phot, np.ndarray), 'z_phot must be a numpy array'
        assert len(z_phot) == len(df), 'Length of z_phot must be equal to that of df'

    pdf_names = ['pdf_' + str(i) for i in range(500) if 'pdf_' + str(i) in df.columns]

    if metric == 'mode':
        df['z_phot'] = binning[np.argmax([df[pdf_names].values], axis=1)][0]
    if metric == 'mean':
        df['z_phot'] = np.inner(binning, df[pdf_names].values)

    mean_spec_array = []
    mean_phot_array = []

    for i in xrange(100):
        df_sample = df.sample(n=len(df), replace=True, weights=None)
        mean_spec_array.append(df_sample.z_spec.mean())
        mean_phot_array.append(df_sample.z_phot.mean())

    mean_spec = np.mean(mean_spec_array)
    mean_phot = np.mean(mean_phot_array)

    err_mean_spec = np.std(mean_spec_array)
    err_mean_phot = np.std(mean_phot_array)

    if weights:
        w_mean_spec_array = []
        w_mean_phot_array = []
        for i in xrange(100):
            df_sample = df.sample(n=len(df), replace=True, weights=weights)
            w_mean_spec_array.append(df_sample.z_spec.mean())
            w_mean_phot_array.append(df_sample.z_phot.mean())

    w_mean_spec = np.mean(mean_spec_array)
    w_mean_phot = np.mean(mean_phot_array)

    w_err_mean_spec = np.std(mean_spec_array)
    w_err_mean_phot = np.std(mean_phot_array)

    return mean_spec
