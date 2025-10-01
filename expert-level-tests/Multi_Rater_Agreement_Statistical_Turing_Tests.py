from utils import _aggregate_raters_binary, _verify_mulitple_annotators, _format
from utils import *
from scipy.stats import  norm
from statsmodels.stats.inter_rater import fleiss_kappa as _fleiss_kappa
from tqdm import tqdm

def fleiss_kappa_delta(annotations, predictions):
    """
    Calculate the difference between the Fleiss' kappa for all annotators vs the Fleiss' kappa when the AI replaces an annotator.
    :param annotations: Annotations for each baby from each annotator.
    :param annotations: Annotations for each baby from each annotator.
    :param predictions: Predictions for each baby.
    :return: Fleiss' kappa Delta
    """
    _verify_mulitple_annotators(annotations)
    ai = np.concatenate([pred['mask'] for pred in predictions])
    humans = np.concatenate(annotations, axis=1) #
    human_ratings = np.stack(humans).T
    freq_human, _ = _aggregate_raters_binary(human_ratings)
    kappa_human = _fleiss_kappa(freq_human) # Kappa for raters

    # Replace each rater with AI and save Kappa value
    kappa_ai = np.zeros(annotations[0].shape[0])
    for i in range(annotations[0].shape[0]):
        annots = humans.copy()
        annots[i] = ai
        ai_and_human_ratings = np.stack(annots).T
        freq_ai, _ = _aggregate_raters_binary(ai_and_human_ratings)
        kappa_ai_val = _fleiss_kappa(freq_ai)
        kappa_ai[i] = kappa_ai_val

    delta_mean = np.mean(kappa_ai) - kappa_human
    delta_all = [kappa - kappa_human for kappa in kappa_ai]
    return _format(delta_mean), delta_all

def fleiss_kappa_delta_bootstrap(annotations, predictions, i_anot,  N=1000, per_annotator=False):
    """
    Assess non-inferiority of AI using the statistical significance of Fleiss' kappa delta.
    A single bootstrap sample is drawn by resampling the annotations per baby with replacement.
    Method taken from
    "Time-Varying EEG Correlations Improve Automated Neonatal Seizure Detection" by Tapani et al.
    https://doi.org/10.1142/S0129065718500302

    :param annotations: Annotations for each baby from each annotator.
    :param annotations: Annotations for each baby from each annotator.
    :param predictions: Predictions for each baby.
    :param N: Number of bootstrap samples to draw.
    :return: p-value for AI inferiority
    """
    _verify_mulitple_annotators(annotations)
    if per_annotator:
        deltas = np.zeros((annotations[0].shape[0], N)) # koliko anotatora pa za svakog N vrednosti
        deltas2 = np.zeros(N)
    else:
        deltas = np.zeros(N) # inace samo N vrednosti
    for i in tqdm(range(N), desc='Running boostrap for non-inferiority test'):
        indices = np.random.choice(len(annotations), len(annotations), replace=True) # true znaci da isti broj moze da se pojavi vise puta
        sampled_annotations = [annotations[index] for index in indices] # ovde je semplovanje takvo da jedna beba moze vise puta da se pojavi
        sampled_predictions = [predictions[index] for index in indices]

        mean_delta, all_delta = fleiss_kappa_delta(sampled_annotations.copy(), sampled_predictions.copy())

        if per_annotator:
            deltas[:, i] = all_delta
            deltas2[i] = mean_delta
        else:
            deltas[i] = mean_delta

    # assume deltas follow a normal distribution
    # find p-value for null hypothesis that AI is non-inferior
    if per_annotator:
        std = np.std(np.mean(deltas, axis=0))
        mean = np.mean(deltas)
        #p_value_mean = 2 * min(norm.cdf(0, mean, std), 1 - norm.cdf(0, mean, std))
        p_value_mean = 1 - norm.cdf(0, mean, std)
        print(f"{mean:63f} ({mean - 1.96 * std:.6f}, {mean + 1.96 * std:.6f}) 95% CI, p={p_value_mean:.6f}")
        print(f'Annotator: {i_anot}')
    else:
        #p_value_mean = 2 * min(norm.cdf(0, np.mean(deltas), np.std(deltas)), 1 - norm.cdf(0, np.mean(deltas), np.std(deltas)))
        p_value_mean = 1 - norm.cdf(0, np.mean(deltas), np.std(deltas))

        print(f"{np.mean(deltas):.6f} ({np.mean(deltas) - 1.96 * np.std(deltas):.6f}, {np.mean(deltas) + 1.96 * np.std(deltas):.6f}) 95% CI, p={p_value_mean:.6f}")
        print(f'Annotator: {i_anot}')
    if per_annotator:
        stds = np.std(deltas, axis=1)
        means = np.mean(deltas, axis=1)
        #p_values = [_format(2 * min(norm.cdf(0, means[i], stds[i]), 1 - norm.cdf(0, means[i], stds[i]))) for i in range(annotations[0].shape[0])]
        p_values = [_format(1 - norm.cdf(0, means[i], stds[i])) for i in range(annotations[0].shape[0])]

        print('Separate: ')
        for i in range(annotations[0].shape[0]):
            print(f"{means[i]:.6f} ({means[i] - 1.96 * stds[i]:.6f}, {means[i] + 1.96 * stds[i]:.6f}) 95% CI, p={p_values[i]:.6f}")
        p = (p_value_mean, p_values)
    else:
        p = p_value_mean

    return p, deltas, np.mean(deltas), np.mean(deltas) - 1.96 * np.std(deltas), np.mean(deltas) + 1.96 * np.std(deltas)
