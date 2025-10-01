from utils import *
from utils import _format
from math import comb
import tqdm

def metric_record(annotations, predictions, which_metric):
    """
    Calculate metric values.
    :param annotations: Annotations made by human raters.
    :param predictions: Predictions made by AI.
    :param which_metric: Which metric to calculate.
    :return: Value of metric for given annotations and predictions.
    """
    metric = []
    for a, p in zip(annotations, predictions):
        if np.sum(a) == 0:
            metric.append(np.nan)
        else:
            if which_metric == 'mcc':
                metric.append(mcc([a], [p]))
            elif which_metric == 'auc':
                metric.append(auc_cc([a], [p]))
            elif which_metric == 'cohen':
                metric.append(cohens_kappa([a], [p]))

    return _format(np.nanmean(metric))

def pairwise_bootstrap(annotations, predictions, metric_to_use, N=1000):
    """
    Generalized bootstrap method for pairwise comparisons between annotators and AI predictions.
    :param annotations: list
        List of annotations made by raters.
    :param predictions: list
        List of predictions.
    :param metric_to_use: string
        Which metric to use in pairwise test: 'mcc', 'auc', 'cohen'.
    :param N: int
        Number of bootstrap iterations.
    :return: dict
        Boostrap results containing metric values, information if AI is noninferior, etc.
    """
    num_annotators = annotations[0].shape[0]  # Number of annotators
    num_pairs = comb(num_annotators - 1, 2) * num_annotators * 2

    metrics = {
        'metric_human': np.zeros((num_pairs, N)),
        'metric_ai': np.zeros((num_annotators * (num_annotators-1), N)),
    }
    data = {
        'human_metric': [], 'human_metric_err': [],
        'ai_metric': [], 'ai_metric_err': [],
    }
    ai_lower_bounds = []
    for n in tqdm(range(N)):
        indices = np.random.choice(len(annotations), len(annotations), replace=True)
        sampled_annotations = [annotations[index] for index in indices]
        sampled_predictions = [predictions[index] for index in indices]

        pair_idx = 0
        for ref_idx in range(num_annotators): # each time different rater is reference
            reference = [a[ref_idx, :] for a in sampled_annotations] # reference rater

            metric_list = []
            for other_idx in range(num_annotators):
                if ref_idx == other_idx:
                    continue

                other = [{'mask': a, 'probs': a} for a in [a[other_idx, :] for a in sampled_annotations]]
                metric_list.append(metric_record(reference, other, which_metric=metric_to_use))

            # Compare reference annotator with AI predictions
            metric_ai = metric_record(reference, sampled_predictions, which_metric=metric_to_use)

            for i in range(len(metric_list)): # firsti against all others
                for j in range(i + 1, len(metric_list)):
                    metrics['sens_human'][pair_idx:pair_idx + 2, n] = [metric_list[i] - metric_list[j], metric_list[j] - metric_list[i]]
                    pair_idx += 2

                metrics['metric_ai'][ref_idx * (num_annotators - 1) + i, n] = metric_ai - metric_list[i]

    # Non-inferiority testing for AI
    for i in range(num_pairs):
        metric_type = 'metric'
        compare_type = 'human'
        # Get values from metrics dictionary
        values = metrics[f'{metric_type}_{compare_type}'][i]
        mean = np.mean(values)
        std = np.std(values)
        ci = 1.96 * std

        # Store mean in appropriate list
        data[f'{compare_type}_metric'].append(mean)
        data[f'{compare_type}_metric_err'].append([ci, ci])

    for i in range(num_annotators * (num_annotators-1)):
        metric_type = 'metric'
        compare_type = 'ai'
        # Get values from metrics dictionary
        values = metrics[f'{metric_type}_{compare_type}'][i]
        mean = np.mean(values)
        std = np.std(values)
        ci = 1.96 * std

        # Store mean in appropriate list
        data[f'{compare_type}_metric'].append(mean)
        data[f'{compare_type}_metric_err'].append([ci, ci])

    non_inferiority_margin = min(s - e[0] for s, e in zip(data['human_metric'], data['human_metric_err']))

    lower_bounds_ai = [s - e[0] for s, e in zip(data['ai_metric'], data['ai_metric_err'])]

    ai_non_inferior = all(lb >= non_inferiority_margin for lb in lower_bounds_ai)
    lower_mean_ai = min(lower_bounds_ai)
    results = {
        "metrics": metrics,
        "ai_non_inferior": ai_non_inferior,
        "human_data": np.mean(metrics["metric_human"], axis=1),
        "ai_data": np.mean(metrics["metric_ai"], axis=1),
        "non_inferiority_margin": non_inferiority_margin,
        "ai_worst": lower_mean_ai,
    }

    if lower_mean_ai > non_inferiority_margin:
        print(f"AI is noninferior.")
    else:
        print(f"AI is NOT noninferior.")

    return results

def pairwise_test(annotations, predictions, N, metric_to_use):
    """
    Function that prepares data format and calls pairwise_boostrap() to run the test for given annotations and predictions.
    :param annotations: ndarray, (n_signals, signal_length, n_raters)
        Numpy array containing annotations for each signal (from multiple raters).
    :param predictions: ndarray, (n_signals, signal_length)
        Predictions for each signal.
    :param N: int
        Number of bootstrap iterations.
    :param metric_to_use: string
        Which metric to use in pairwise test: 'mcc', 'auc', 'cohen'.
    :return: dict
        Boostrap results containing metric values, information if AI is noninferior, etc.
    """
    annotations1 = []
    signals_to_stack = []
    num_experts = annotations.shape[2]

    for j in range(annotations.shape[0]):
        for i in range(int(num_experts)):
            signals_to_stack.append((annotations[j, :, i].copy() > 0.5).astype(int))

        annotations1.append(np.vstack(signals_to_stack))
        signals_to_stack = []

    predictions1 = []
    for j in range(predictions.shape[0]):
        predictions1.append({'mask': (predictions[j, :] > 0.5).astype(int),
                             'probs': predictions[j, :]})

    bootstrap_results = pairwise_bootstrap(annotations1, predictions1, metric_to_use=metric_to_use, N=N)


    return bootstrap_results
