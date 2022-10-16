# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
""" pyod_utils """
from __future__ import division
from __future__ import print_function

import numbers
import numpy as np
from numpy import percentile
from sklearn.metrics import precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import column_or_1d
from sklearn.utils import check_array
from sklearn.utils import check_consistent_length
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn import metrics

MAX_INT = np.iinfo(np.int32).max
MIN_INT = -1 * MAX_INT


def check_parameter(param, low=MIN_INT, high=MAX_INT, param_name='',
                    include_left=False, include_right=False):
    """Check if an input is within the defined range.

    Parameters
    ----------
    param : int, float
        The input parameter to check.

    low : int, float
        The lower bound of the range.

    high : int, float
        The higher bound of the range.

    param_name : str, optional (default='')
        The name of the parameter.

    include_left : bool, optional (default=False)
        Whether includes the lower bound (lower bound <=).

    include_right : bool, optional (default=False)
        Whether includes the higher bound (<= higher bound).

    Returns
    -------
    within_range : bool or raise errors
        Whether the parameter is within the range of (low, high)

    """

    # param, low and high should all be numerical
    if not isinstance(param, (numbers.Integral, np.integer, np.float)):
        raise TypeError('{param_name} is set to {param} Not numerical'.format(
            param=param, param_name=param_name))

    if not isinstance(low, (numbers.Integral, np.integer, np.float)):
        raise TypeError('low is set to {low}. Not numerical'.format(low=low))

    if not isinstance(high, (numbers.Integral, np.integer, np.float)):
        raise TypeError('high is set to {high}. Not numerical'.format(
            high=high))

    # at least one of the bounds should be specified
    if low is MIN_INT and high is MAX_INT:
        raise ValueError('Neither low nor high bounds is undefined')

    # if wrong bound values are used
    if low > high:
        raise ValueError(
            'Lower bound > Higher bound')

    # value check under different bound conditions
    if (include_left and include_right) and (param < low or param > high):
        raise ValueError(
            '{param_name} is set to {param}. '
            'Not in the range of [{low}, {high}].'.format(
                param=param, low=low, high=high, param_name=param_name))

    if (include_left and not include_right) and (
            param < low or param >= high):
        raise ValueError(
            '{param_name} is set to {param}. '
            'Not in the range of [{low}, {high}).'.format(
                param=param, low=low, high=high, param_name=param_name))

    if (not include_left and include_right) and (
            param <= low or param > high):
        raise ValueError(
            '{param_name} is set to {param}. '
            'Not in the range of ({low}, {high}].'.format(
                param=param, low=low, high=high, param_name=param_name))

    if (not include_left and not include_right) and (
            param <= low or param >= high):
        raise ValueError(
            '{param_name} is set to {param}. '
            'Not in the range of ({low}, {high}).'.format(
                param=param, low=low, high=high, param_name=param_name))

    return True


def check_detector(detector):
    """Checks if fit and decision_function methods exist for given detector

    Parameters
    ----------
    detector : pyod.models
        Detector instance for which the check is performed.

    """

    if not hasattr(detector, 'fit') or not hasattr(detector,
                                                   'decision_function'):
        raise AttributeError("%s is not a detector instance." % (detector))


def standardizer(X, X_t=None, keep_scalar=False):
    """Conduct Z-normalization on data to turn input samples become zero-mean
    and unit variance.

    Parameters
    ----------
    X : numpy array of shape (n_samples, n_features)
        The training samples

    X_t : numpy array of shape (n_samples_new, n_features), optional (default=None)
        The data to be converted

    keep_scalar : bool, optional (default=False)
        The flag to indicate whether to return the scalar

    Returns
    -------
    X_norm : numpy array of shape (n_samples, n_features)
        X after the Z-score normalization

    X_t_norm : numpy array of shape (n_samples, n_features)
        X_t after the Z-score normalization

    scalar : sklearn scalar object
        The scalar used in conversion

    """
    X = check_array(X)
    scaler = StandardScaler().fit(X)

    if X_t is None:
        if keep_scalar:
            return scaler.transform(X), scaler
        return scaler.transform(X)

    X_t = check_array(X_t)
    if X.shape[1] != X_t.shape[1]:
        raise ValueError(
            "The number of input data feature should be consistent"
            "X has {0} features and X_t has {1} features.".format(
                X.shape[1], X_t.shape[1]))
    if keep_scalar:
        return scaler.transform(X), scaler.transform(X_t), scaler
    return scaler.transform(X), scaler.transform(X_t)


def score_to_label(pred_scores, outliers_fraction=0.1):
    """Turn raw outlier outlier scores to binary labels (0 or 1).

    Parameters
    ----------
    pred_scores : list or numpy array of shape (n_samples,)
        Raw outlier scores. Outliers are assumed have larger values.

    outliers_fraction : float in (0,1)
        Percentage of outliers.

    Returns
    -------
    outlier_labels : numpy array of shape (n_samples,)
        For each observation, tells whether or not
        it should be considered as an outlier according to the
        fitted model. Return the outlier probability, ranging
        in [0,1].
    """
    # check input values
    pred_scores = column_or_1d(pred_scores)
    check_parameter(outliers_fraction, 0, 1)

    threshold = percentile(pred_scores, 100 * (1 - outliers_fraction))
    pred_labels = (pred_scores > threshold).astype('int')
    return pred_labels


def precision_recall_n_scores(y, y_pred, n=None):
    """Utility function to calculate precision @ rank n.

    Parameters
    ----------
    y : list or numpy array of shape (n_samples,)
        The ground truth. Binary (0: inliers, 1: outliers).

    y_pred : list or numpy array of shape (n_samples,)
        The raw outlier scores as returned by a fitted model.

    n : int, optional (default=None)
        The number of outliers. if not defined, infer using ground truth.

    Returns
    -------
    precision_at_rank_n : float
        Precision at rank n score.

    """

    # turn raw prediction decision scores into binary labels
    y_pred = get_label_n(y, y_pred, n)

    # enforce formats of y and labels_
    y = column_or_1d(y)
    y_pred = column_or_1d(y_pred)

    return precision_score(y, y_pred), recall_score(y, y_pred)


def get_label_n(y, y_pred, n=None):
    """Function to turn raw outlier scores into binary labels by assign 1
    to top n outlier scores.

    Parameters
    ----------
    y : list or numpy array of shape (n_samples,)
        The ground truth. Binary (0: inliers, 1: outliers).

    y_pred : list or numpy array of shape (n_samples,)
        The raw outlier scores as returned by a fitted model.

    n : int, optional (default=None)
        The number of outliers. if not defined, infer using ground truth.

    Returns
    -------
    labels : numpy array of shape (n_samples,)
        binary labels 0: normal points and 1: outliers

    Examples
    --------
    >>> from pyod.utils.utility import get_label_n
    >>> y = [0, 1, 1, 0, 0]
    >>> y_pred = [0.1, 0.5, 0.3, 0.2, 0.7]
    >>> get_label_n(y, y_pred)
    array([0, 1, 0, 0, 1])

    """

    # enforce formats of inputs
    y = column_or_1d(y)
    y_pred = column_or_1d(y_pred)

    check_consistent_length(y, y_pred)
    y_len = len(y)

    if n is not None:
        outliers_fraction = n / y_len
    else:
        outliers_fraction = np.count_nonzero(y) / y_len

    threshold = percentile(y_pred, 100 * (1 - outliers_fraction))
    y_pred = (y_pred > threshold).astype('int')

    return y_pred


def gmean_scores(y, y_pred, threshold=0.5):
    """Utility function to calculate precision @ rank n.

    Parameters
    ----------
    y : list or numpy array of shape (n_samples,)
        The ground truth. Binary (0: inliers, 1: outliers).

    y_pred : list or numpy array of shape (n_samples,)
        The raw outlier scores as returned by a fitted model.


    Returns
    -------
    Gmean: float
    """
    y_pred = get_label_n(y, y_pred)
    ones_all = (y == 1).sum()
    ones_correct = ((y == 1) & (y_pred == 1)).sum()
    zeros_all = (y == 0).sum()
    zeros_correct = ((y == 0) & (y_pred == 0)).sum()
    Gmean = np.sqrt(1.0*ones_correct/ones_all)
    Gmean *= np.sqrt(1.0*zeros_correct/zeros_all)

    return Gmean


def get_gmean(y, y_pred, threshold=0.5):
    """Utility function to calculate precision @ rank n.

    Parameters
    ----------
    y : list or numpy array of shape (n_samples,)
        The ground truth. Binary (0: inliers, 1: outliers).

    y_pred : list or numpy array of shape (n_samples,)
        The raw outlier scores as returned by a fitted model.


    Returns
    -------
    Gmean: float
    """
    y_pred = get_label_n(y, y_pred)
    ones_all = (y == 1).sum()
    ones_correct = ((y == 1) & (y_pred == 1)).sum()
    zeros_all = (y == 0).sum()
    zeros_correct = ((y == 0) & (y_pred == 0)).sum()
    Gmean = np.sqrt((1.0*ones_correct/ones_all) *
                    (1.0*zeros_correct/zeros_all))

    return Gmean


def AUC_and_Gmean(y_test, y_scores):
    ''' AUC and Gmean '''
    auc = round(roc_auc_score(y_test, y_scores), ndigits=4)
    prn, rec = precision_recall_n_scores(y_test, y_scores)
    prn = round(prn, ndigits=4)
    rec = round(rec, ndigits=4)
    gmean = round(get_gmean(y_test, y_scores, 0.5), ndigits=4)

    return auc, prn, rec, gmean


def AUC_and_PR(y_test, y_scores):
    ''' AUC and PR '''
    auc = round(roc_auc_score(y_test, y_scores), ndigits=4)
    prn, rec = precision_recall_n_scores(y_test, y_scores)
    prn = round(prn, ndigits=4)
    rec = round(rec, ndigits=4)
    precision, recall, _ = precision_recall_curve(y_test, y_scores)
    PR = metrics.auc(recall, precision)

    return auc, prn, rec, PR


def AUC_and_Gmean_50recall(y_test, y_scores):
    ''' AUC and Gmean when recall=50% '''
    auc = round(roc_auc_score(y_test, y_scores), ndigits=4)

    precision, recall, _ = precision_recall_curve(y_test, y_scores)
    close = 1
    close_index = 0
    for j in range(len(recall)):
        if abs(recall[j]-0.5) < close:
            close = abs(recall[j]-0.5)
            close_index = j
    rec = recall[close_index]
    prn = precision[close_index]

    prec = average_precision_score(y_test, y_scores)

    return auc, prn, rec, prec
