import pandas as pd 
import numpy as np
from weighted_kde import gaussian_kde
from cPickle import dumps, load


def ld_writedicts(filepath, dictionary):
    f = open(filepath, 'w')
    newdata = dumps(dictionary, 1)
    f.write(newdata)
    f.close()


def ld_readdicts(filepath):
    f = open(filepath, 'r')
    d = load(f)
    f.close()
    return d


def _bias(z_spec, z_phot, weight=None):
    dz1 = (z_spec - z_phot)
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
    dz1 = (z_spec - z_phot)
    if weight is None:
        sigma = np.std(dz1)
    else:
        sigma = _w_std(dz1, weights=weight)
    return sigma


def _percentile(n, percent):
    n = np.sort(n)
    k = (len(n) - 1) * percent
    f = np.floor(k)
    c = np.ceil(k)
    if f == c:
        return n[int(k)]
    d0 = n[int(f)] * (c - k)
    d1 = n[int(c)] * (k - f)
    return d0 + d1


def _w_std(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    variance = np.average((np.abs(values - average)) ** 2, weights=weights)  # Fast and numerically precise
    return np.sqrt(variance)


def _normalize_pdf(pdf, dz):
    """
    returns normalized pdf
    """
    area = np.trapz(pdf, dx=dz)
    return pdf / area


def mean(df, binning, z_phot, metric='mean', weights=None, tomo_bins=np.array([0, 5.0]), n_resample=50):
    """
    :param df: pandas data-frame
    :param binning: center of redshift bins
    :param metric : 'mean','mode' of the pdf as the point estimate
    :param weights: optional weighting scheme with same len as df
    :param tomo_bins: in which z-bins array exp [0.0, 0.2, 0.6, 1.8]
    :param n_resample : Amount of resamples to estimate mean and variance on the mean.
    :return: pandas data frame with mean estimates
    """

    assert isinstance(df, pd.DataFrame), 'df must be a pandas DataFrame'
    assert isinstance(binning, np.ndarray), 'binning must be a numpy array'
    if weights:
        assert weights in df.columns, str(weights) + ' not in df.columns'
        df[weights] = (df[weights] / df[weights].sum()).values  # normalize weights
    else:
        df[weights] = 1.0 / len(df)  # set uniform weights if none given
    assert isinstance(z_phot, np.ndarray), 'z_phot must be a numpy array'
    assert len(z_phot) == len(df), 'Length of z_phot must be equal to that of df'
    df['phot_sel'] = z_phot  # Make the selection photo-z a part of the DataFrame
    assert 'z_spec' in df.columns, 'The df needs a "z_spec" columns'
    assert 'pdf_0' in df.columns, 'The pdf values must have pdf_i column name'
    pdf_names = ['pdf_' + str(i) for i in range(500) if 'pdf_' + str(i) in df.columns]

    if metric == 'mode':
        df['z_phot'] = binning[np.argmax([df[pdf_names].values], axis=1)][0]
    elif metric == 'mean':
        df['z_phot'] = np.inner(binning, df[pdf_names].values)
    else:
        print 'Metric needs to be either "mode" or "mean", using mean !'
        df['z_phot'] = np.inner(binning, df[pdf_names].values)

    mean_spec_bin_array = []
    err_mean_phot_bin_array = []

    mean_phot_bin_array = []
    err_mean_spec_bin_array = []

    w_mean_spec_bin_array = []
    w_err_mean_spec_bin_array = []

    w_mean_phot_bin_array = []
    w_err_mean_phot_bin_array = []

    df_index = []
    df_count = []
    mean_z_bin = []

    print len(tomo_bins)

    for j in range(len(tomo_bins) - 1):
        sel = (df.phot_sel > tomo_bins[j]) & (df.phot_sel <= tomo_bins[j + 1])
        if sel.sum() > 0:
            df_index.append('z [' + str(tomo_bins[j]) + ', ' + str(tomo_bins[j + 1]) + ']')
            df_count.append(sel.sum())
            mean_z_bin.append((tomo_bins[j] + tomo_bins[j + 1]) / 2.0)
            df_sel = df[sel]

            mean_spec_array = []
            mean_phot_array = []

            w_mean_spec_array = []
            w_mean_phot_array = []

            for i in xrange(n_resample):
                df_sample = df_sel.sample(n=len(df_sel), replace=True, weights=None)
                mean_spec_array.append(df_sample.z_spec.mean())
                mean_phot_array.append(df_sample.z_phot.mean())

                df_sample = df_sel.sample(n=len(df_sel), replace=True, weights=df_sel[weights])
                w_mean_spec_array.append(np.average(df_sample.z_spec))
                w_mean_phot_array.append(np.average(df_sample.z_phot))

            mean_spec = np.mean(mean_spec_array)
            mean_phot = np.mean(mean_phot_array)

            err_mean_spec = np.std(mean_spec_array)
            err_mean_phot = np.std(mean_phot_array)

            w_mean_spec = np.mean(w_mean_spec_array)
            w_mean_phot = np.mean(w_mean_phot_array)

            w_err_mean_spec = np.std(w_mean_spec_array)
            w_err_mean_phot = np.std(w_mean_phot_array)

            mean_spec_bin_array.append(mean_spec)
            mean_phot_bin_array.append(mean_phot)

            err_mean_spec_bin_array.append(err_mean_spec)
            err_mean_phot_bin_array.append(err_mean_phot)

            w_mean_spec_bin_array.append(w_mean_spec)
            w_mean_phot_bin_array.append(w_mean_phot)

            w_err_mean_spec_bin_array.append(w_err_mean_spec)
            w_err_mean_phot_bin_array.append(w_err_mean_phot)

        else:
            print 'Bin ' + str(j) + ' has no objects and it not included in the summary (counting starts at zero)'

    to_pandas = np.vstack((mean_z_bin, df_count,
                           mean_spec_bin_array, err_mean_spec_bin_array,
                           mean_phot_bin_array, err_mean_phot_bin_array,
                           w_mean_spec_bin_array, w_err_mean_spec_bin_array,
                           w_mean_phot_bin_array, w_err_mean_phot_bin_array,
                           )).T

    return_df = pd.DataFrame(to_pandas, columns=['mean_z_bin', 'n_obj',
                                                 'mean_spec', 'err_mean_spec',
                                                 'mean_phot', 'err_mean_phot',
                                                 'w_mean_spec', 'w_err_mean_spec',
                                                 'w_mean_phot', 'w_err_mean_phot',
                                                 ])
    return_df.index = df_index

    return return_df


def weighted_nz_distributions(df, binning, weights=None, tomo_bins=np.array([0, 5.0]), z_phot=None, n_resample=50):
    """
    :param df: pandas data-frame
    :param binning: center of redshift bins
    :param weights: optional weighting scheme with same len as df
    :param tomo_bins: in which z-bins array exp [0.0, 0.2, 0.6, 1.8]
    :param n_resample : Amount of resamples to estimate mean and variance on the mean.
    :return: pandas data frame with mean estimates
    """

    assert isinstance(df, pd.DataFrame), 'df must be a pandas DataFrame'
    assert isinstance(binning, np.ndarray), 'binning must be a numpy array'
    if weights:
        assert weights in df.columns, str(weights) + ' not in df.columns'
        df[weights] = (df[weights] / df[weights].sum()).values  # normalize weights
    else:
        df[weights] = 1.0 / len(df)  # set uniform weights if none given
    if tomo_bins:
        assert isinstance(z_phot, np.ndarray), 'z_phot must be a numpy array'
        assert len(z_phot) == len(df), 'Length of z_phot must be equal to that of df'
        df['phot_sel'] = z_phot  # Make the selection photo-z a part of the DataFrame
    assert 'z_spec' in df.columns, 'The df needs a "z_spec" in df.columns'
    pdf_names = ['pdf_' + str(i) for i in range(500) if 'pdf_' + str(i) in df.columns]

    phot_iter = {}
    spec_iter = {}

    # In the following section the tomographic bins are treated

    for j in xrange(0, len(tomo_bins) - 1):
        sel = (df.phot_sel > tomo_bins[j]) & (df.phot_sel <= tomo_bins[j + 1])
        if sel.sum() > 0:
            df_sel = df[sel]

            phot_iter[j + 1] = {}
            spec_iter[j + 1] = {}

            for i in xrange(n_resample):
                df_sample = df_sel.sample(n=len(df_sel), replace=True, weights=df_sel[weights])
                kde_w_spec_pdf = gaussian_kde(df_sample.z_spec.values, bw_method='silverman')
                kde_w_spec_pdf = kde_w_spec_pdf(binning)

                phot_iter[j + 1][i + 1] = _normalize_pdf(df_sample[pdf_names].sum(), binning[1] - binning[0]).values
                spec_iter[j + 1][i + 1] = kde_w_spec_pdf

            phot_iter[j + 1][0] = _normalize_pdf(df_sel[pdf_names].sum(), binning[1] - binning[0]).values
            kde_w_spec_pdf = gaussian_kde(df_sel.z_spec.values, bw_method='silverman')
            spec_iter[j + 1][0] = kde_w_spec_pdf(binning)

    # In the following section the full n(z) is treated i.e not in tomographic bins

    sel = (df.phot_sel > tomo_bins[0]) & (df.phot_sel <= tomo_bins[len(tomo_bins) - 1])
    df_sel = df[sel]
    phot_iter[0] = {}
    spec_iter[0] = {}

    for i in xrange(n_resample):
        df_sample = df_sel.sample(n=len(df_sel), replace=True, weights=df_sel[weights])
        kde_w_spec_pdf = gaussian_kde(df_sample.z_spec.values, bw_method='silverman')
        kde_w_spec_pdf = kde_w_spec_pdf(binning)

        phot_iter[0][i + 1] = _normalize_pdf(df_sample[pdf_names].sum(), binning[1] - binning[0]).values
        spec_iter[0][i + 1] = kde_w_spec_pdf

    phot_iter[0][0] = _normalize_pdf(df_sel[pdf_names].sum(), binning[1] - binning[0]).values
    kde_w_spec_pdf = gaussian_kde(df_sel.z_spec.values, bw_method='silverman')
    spec_iter[0][0] = kde_w_spec_pdf(binning)

    data_for_wl = {'binning': binning, 'phot': phot_iter, 'spec': spec_iter}

    return phot_iter, spec_iter, data_for_wl