from utils import _aggregate_raters_binary, _verify_mulitple_annotators, _format
from utils import *
from scipy.stats import  norm
from statsmodels.stats.inter_rater import fleiss_kappa as _fleiss_kappa
from tqdm import tqdm

def fleiss_kappa_expert_distribution(annotations, N=1000):
    """
    Bootstrap distribution of Fleiss' kappa using experts only.
    :param annotations: Annotations for each baby from each annotator.
    :return:
        mean_kappa
        lower_2_5_percentile
        margin (mean - 2.5th percentile) 
        full_distribution
    """
    _verify_mulitple_annotators(annotations)

    kappas = np.zeros(N)

    for i in tqdm(range(N), desc="Bootstrapping expert kappa"):
        indices = np.random.choice(len(annotations), len(annotations), replace=True)
        sampled_annotations = [annotations[idx] for idx in indices]

        humans = np.concatenate(sampled_annotations, axis=1)
        human_ratings = np.stack(humans).T
        freq_human, _ = _aggregate_raters_binary(human_ratings)
        kappas[i] = _fleiss_kappa(freq_human)

    mean_kappa = np.mean(kappas)
    lower_2_5 = np.percentile(kappas, 2.5)
    margin = mean_kappa - lower_2_5

    #print(f"Expert κ mean: {mean_kappa:.6f}")
    #print(f"2.5th percentile: {lower_2_5:.6f}")
    #print(f"Non-inferiority margin: {margin:.6f}")

    return mean_kappa, lower_2_5, margin, kappas

def fleiss_kappa_delta(annotations, predictions):
    """
    Calculate the difference between the Fleiss' kappa for all annotators vs the Fleiss' kappa when the AI replaces an annotator.
    :param annotations: Annotations for each baby from each annotator.
    :param predictions: Predictions for each baby.
    :return: Fleiss' kappa Delta
    """
    _verify_mulitple_annotators(annotations)
    ai = np.concatenate([pred['mask'] for pred in predictions])
    humans = np.concatenate(annotations, axis=1) 
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
    https://doi.org/10.1142/S0129065718500302 with modification.

    Steps:
    1) Compute expert-only bootstrap distribution, and define margin as mean - 2.5th percentile
    2) Compute delta-kappa bootstrap distribution
    3) Check if 5th percentile of delta-kappa distribution > -margin

    :param annotations: Annotations for each baby from each annotator.
    :param predictions: Predictions for each baby.
    :param N: Number of bootstrap samples to draw.
    :return: p-value for AI inferiority
    """
    _verify_mulitple_annotators(annotations)

    # Step 1 - Computing distribution of Fleiss Kappa for only experts
    mean_exp, lower_exp, margin, expert_dist = fleiss_kappa_expert_distribution(annotations, N=N) 
    
    if per_annotator:
        deltas = np.zeros((annotations[0].shape[0], N)) 
        deltas2 = np.zeros(N) 
    else:
        deltas = np.zeros(N) 

    # Bootstrap for one AI
    for i in tqdm(range(N), desc='Running boostrap for non-inferiority test'):
        indices = np.random.choice(len(annotations), len(annotations), replace=True) 
        sampled_annotations = [annotations[index] for index in indices] 
        sampled_predictions = [predictions[index] for index in indices]

        mean_delta, all_delta = fleiss_kappa_delta(sampled_annotations.copy(), sampled_predictions.copy())

        if per_annotator:
            deltas[:, i] = all_delta
            deltas2[i] = mean_delta
        else:
            deltas[i] = mean_delta

    # assume deltas follow a normal distribution
    # find p-value for null hypothesis that 5th percentile <= margin
    if per_annotator:
        n_annotators = deltas.shape[0] 
        p_values = np.zeros(n_annotators)
        passes = np.zeros(n_annotators, dtype=bool) # for each annotator information if passed the test
        
        for i in range(n_annotators): # for each annotator separately, results when annotator n was substituted with AI
            delta_dist = deltas[i]
            # 5th percentile
            delta_p5 = np.percentile(delta_dist, 5)
    
            # empirical one-sided p-value
            p_val = np.mean(delta_dist <= margin)
    
            p_values[i] = p_val
            passes[i] = delta_p5 > margin
    
            print(f"Annotator {i}: "
                  f"mean={np.mean(delta_dist):.3f}, "
                  f"P5={delta_p5:.3f}, "
                  f"margin={-margin:.3f}, "
                  f"p={p_val:.3f}, "
                  f"PASS={passes[i]}")
        
    else:
        delta_p5 = np.percentile(deltas, 5)
        # one sided p value
        p_value_mean = np.mean(deltas <= -margin)
        passes = delta_p5 > -margin
        print(f"Mean delta={np.mean(deltas):.3f}")
        print(f"Delta P5={delta_p5:.3f}")
        print(f"Margin={-margin:.3f}")
        print(f"p-value={p_value_mean:.3f}")
        print(f"PASS={passes}")

    if per_annotator:
        pass_all = np.all(passes)
        pass_majority = np.sum(passes) > (n_annotators // 2)
        pass_any = np.any(passes)
    
        print("\nSummary:")
        print(f"Pass ALL raters: {pass_all}")
        print(f"Pass MAJORITY raters: {pass_majority}")
        print(f"Pass ANY rater: {pass_any}")
    
        p = {
            "p_values": p_values,
            "passes_individual": passes,
            "pass_all": pass_all,
            "pass_majority": pass_majority,
            "pass_any": pass_any
        }
    else:
        p = {
            "p_values": p_value_mean,
            "pass": passes
            }

    return p, deltas



