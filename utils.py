from statsmodels.stats.inter_rater import fleiss_kappa
import numpy as np
from sklearn.metrics import roc_auc_score, matthews_corrcoef, cohen_kappa_score
from irrCAC.raw import CAC

def binary_to_fleiss_format(annotations):
    """Convert binary annotations to Fleiss' kappa input format.
    :param annotations: annotations in binary format
    :return: Fleiss' kappa input format'
    """
    #print(annotations.shape)
    n_signals, n_points, num_annotators = annotations.shape
    reshaped = annotations.reshape(-1, num_annotators)
    counts = np.zeros((reshaped.shape[0], 2), dtype=int)
    counts[:, 0] = (reshaped == 0).sum(axis=1)  # Count of 0s
    counts[:, 1] = (reshaped == 1).sum(axis=1)  # Count of 1s
    return counts

def compute_fleiss_kappa(annotations):
    """Compute Fleiss' kappa for all annotators.
    :param annotations: annotations in binary format
    :return: Fleiss' kappa for all annotators
    """
    fleiss_input = binary_to_fleiss_format(annotations)
    return fleiss_kappa(fleiss_input)

def compute_fleiss_by_group(annotations, group_sizes):
    """Compute Fleiss' kappa for each group separately.
    :param annotations: annotations in binary format
    :param group_sizes: group sizes
    :return: Fleiss' kappa for each group separately
    """
    fk_results = []
    start_idx = 0

    for size in group_sizes:
        if size == 0:
            continue
        fleiss_input_group = binary_to_fleiss_format(annotations[:, :, start_idx:start_idx + size])
        fk_results.append(fleiss_kappa(fleiss_input_group))
        start_idx += size

    return fk_results

def adjust_beta_params_for_target_imbalance(target_imbalance, n_samples=500000, tol=0.01, max_iter=100):
    """
    Iteratively adjusts the alpha and beta to get the empirical prevalence closer to the target.
    :param target_imbalance: target imbalance in the dataset
    :param n_samples: random sample size to use for beta distribution
    :param tol: tolerance for beta distribution
    :param max_iter: maximum number of iterations
    :return: adjusted alpha and beta to use fot the target imbalance
    """
    target_prevalence = 1 / target_imbalance
    alpha = target_prevalence
    beta = 1 - target_prevalence
    for _ in range(max_iter):
        samples = np.random.beta(alpha, beta, n_samples)
        emp_prevalence = (samples >= 0.5).mean()
        curr_imbalance = (1 - emp_prevalence) / emp_prevalence
        if abs(curr_imbalance - target_imbalance) / target_imbalance < tol:
            break
        # Adjust alpha and beta slightly depending on result
        if curr_imbalance > target_imbalance:
            alpha *= 1.05
            beta *= 0.95
        else:
            alpha *= 0.95
            beta *= 1.05
    return alpha, beta


def compute_beta_params(imbalance):
    """Compute alpha and beta for the Beta distribution based on target imbalance.
    :param imbalance: imbalance of the target distribution
    :return: alpha, beta parameters of the Beta distribution
    """
    if imbalance == 1:
        alpha = 0.5
        beta = (1 - 0.5)
        return alpha, beta

    return adjust_beta_params_for_target_imbalance(imbalance)

def generate_signal(n_points, mean_probs, gaussian_std):
    """Generate probability-based Gaussian samples and binarize them.
    :param n_points: how long signal should be in seconds
    :param mean_probs: base probability sequence for each annotaator group
    :param gaussian_std: standard deviation of the Gaussian distribution
    :return: binary annotation and probabilities
    """
    gaussian_samples = np.random.normal(loc=mean_probs, scale=gaussian_std, size=(1, n_points)) # Generated means for that group, and std used same for all of them
    binary_signal = (gaussian_samples >= 0.5).astype(int)  # Thresholding
    return binary_signal, gaussian_samples

def process_baby(annotations, consensus):
    """
    Process annotations for a single baby.

    :param annotations: Raw annotations for a single baby.
    :param consensus: Type of consensus.
    :return: Processed annotations, and a mask indicating which annotations were used.
    """
    a_consensus = annotations.mean(axis=0)

    if consensus == 'unanimous':
        unanimity_mask = (a_consensus == 1) | (a_consensus == 0)
        unanimous_annotations = a_consensus[unanimity_mask]
        return unanimous_annotations, unanimity_mask

    elif consensus == 'majority':
        majority_annotations = a_consensus.round().astype(int)
        return majority_annotations, np.ones_like(majority_annotations), a_consensus

    elif consensus == 'all':
        # Use all annotations
        return annotations, np.ones_like(a_consensus).astype(bool)
    else:
        raise ValueError(f"Invalid consensus type: {consensus}")


def process_annotations(annotations, consensus):
    """
    Process annotations based on consensus type.

    :param annotations: Raw annotations from the dataset.
    :param consensus: Type of consensus.
    :return: Processed annotations and a mask indicating which annotations were used.
    """

    processed_annotations = []
    annotation_masks = []
    consensus_percents = []
    seizure_consensus = []

    for i in range(0, len(annotations)):
        a, m, a_consensus = process_baby(annotations[i].copy(), consensus)

        full_consensus = np.sum((a_consensus == 0) | (a_consensus == 1))
        consensus_percents.append(full_consensus / len(annotations[i][0]))
        seizure_consensus.append(np.sum(a_consensus == 1) / np.sum(a_consensus > 0.5))

        processed_annotations.append(a)
        annotation_masks.append(m)
    return processed_annotations, consensus_percents, seizure_consensus, annotation_masks


def compute_IRA_from_df(df, metric):
    cac = CAC(df, digits=15)
    if metric == 'fleiss':
        return cac.fleiss()["est"]["coefficient_value"]
    elif metric == 'gwet':
        return cac.gwet()["est"]["coefficient_value"]
    elif metric == 'krippendorf':
        return cac.krippendorf()["est"]["coefficient_value"]
    else:
        raise ValueError('Unknown metric')

def _aggregate_raters_binary(data, n_cat=2):
    '''Optimized aggregation for binary data.

    :param data : array_like, 2-Dim
        Binary data containing category assignment with subjects in rows and
        raters in columns. Values must be 0 or 1.
    :param n_cat : int, optional
        Number of categories. Default is 2, which is the only valid value
        for binary data. Included for interface consistency.
    :param return : ndarray, (n_rows, n_cat)
        Contains counts of raters that assigned each category level to individuals.
        Subjects are in rows, category levels (0 and 1) in columns.
    '''
    assert n_cat == 2, 'Binary data must have 2 categories.'
    data = np.asarray(data, dtype=int)  # Ensure data is an integer array

    # Since data is binary, simply sum the ones for each row and subtract from
    # the total number of raters to get counts for zeros.
    sum_ones = data.sum(axis=1).reshape(-1, 1)
    sum_zeros = data.shape[1] - sum_ones

    # Stack the counts for zeros and ones horizontally
    counts = np.hstack((sum_zeros, sum_ones))

    cat_uni = np.array([0, 1])  # Categories are known to be 0 and 1

    return counts, cat_uni


def _format(metric):
    """
    Format a metric as a float with 3 decimal places.
    :param metric: Metric to format.
    :return: Formatted metric.
    """
    rounded = round(metric, 3)
    if isinstance(rounded, (np.float64, np.int64)):
        return rounded.item()
    else:
        return rounded

def _verify_mulitple_annotators(annotations):
    """
    Verify that there are multiple annotators for each baby.
    :param annotations: Annotations for each baby.
    """
    if len(annotations[0].shape) != 2:
        raise ValueError("Annotations should be a list of 2D arrays, where each array is the annotations for a single baby from each annotator.")

def _concatenate_preds_and_annos(annotations, predictions):
    """
    Concatenate all annotations and predictions.
    :param annotations: Annotations for each baby.
    :param predictions: Predictions for each baby.
    :return: Concatenated annotations and predictions.
    """
    all_annotations = np.concatenate(annotations)
    all_preds = np.concatenate([pred['mask'] for pred in predictions])
    all_probs = np.concatenate([pred['probs'] for pred in predictions])
    return all_annotations, all_preds, all_probs

def mcc(annotations, predictions):
    """
    Calculate the MCC across all babies, by concatenating all annotations and predictions.
    :param annotations: Annotations for each baby.
    :param predictions: Predictions for each baby.
    :return: MCC
    """

    all_annotations, all_preds, _ = _concatenate_preds_and_annos(annotations, predictions)

    return _format(matthews_corrcoef(all_annotations, all_preds))

def auc_cc(annotations, predictions):
    """
    Calculate the AUC across all babies, by concatenating all annotations and predictions.
    :param annotations: Annotations for each baby.
    :param predictions: Predictions for each baby.
    :return: AUC
    """
    all_annotations, _, all_probs = _concatenate_preds_and_annos(annotations, predictions)

    return _format(roc_auc_score(all_annotations, all_probs))


def cohens_kappa(annotations, predictions):
    """
    Calculate Cohen's kappa concatenated predictions and consensus annotations.
    :param annotations: Annotations for each baby.
    :param predictions: Predictions for each baby.
    :return: Cohen's kappa
    """
    if isinstance(predictions[0], dict):
        all_preds = np.concatenate([pred['mask'] for pred in predictions])
    else:
        all_preds = np.concatenate(predictions)
    all_annotations = np.concatenate(annotations)
    #print(all_annotations)
    #print(all_preds)

    return _format(cohen_kappa_score(all_annotations, all_preds))

