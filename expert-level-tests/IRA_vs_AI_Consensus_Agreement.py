import pandas as pd
from tqdm import tqdm
from utils import *

def IRA_vs_AI_Majority_Comparison(annotations, predictions, n_bootstrap, metric):
    """
    Implementation of approach for expert-level test from "Automated Interpretation of Clinical Electroencephalograms
    Using Artificial Intelligence". AC1 is calculated for between experts for each signal separately, then also
    for prediction and majority consensus, separately again. CI 95% is calculated using std from Gwet paper.
    :param annotations: Original annotations. Shape: number_of_annotations x annotation_length x number_of_raters
    :param predictions: Predictions of the model. {['prediction': [], 'probabilities': []], ...}
    :param majority: Majority consensus for annotations of experts.
    :param metric: Which IRA metric is used.
    :return: Returns ac1 metric for both cases, and 95% CI for both cases.
    """
    # Change to dataframe because CAC takes that as input
    IRA_list = []
    IRA_AI_list = []
    n = len(annotations)

    annotations_copy = annotations.copy()
    annotations2 = []
    for j in range(annotations_copy.shape[0]):
        stacked = [annotations_copy[j, :, k] for k in range(annotations_copy.shape[2])]
        annotations2.append(np.vstack(stacked))
    majority, _, _, _ = process_annotations(annotations2.copy(), 'majority')

    for _ in tqdm(range(n_bootstrap), desc="Bootstrapping AC1"):
        indices = np.random.choice(n, n, replace=True)

        stacked_annotations = np.concatenate([annotations2[i] for i in indices], axis=1) # shape: (n, A, T)
        annotator_matrix = np.stack(stacked_annotations, axis=0).T  # shape: (n*A, T)
        boot_df_human = pd.DataFrame(annotator_matrix)

        boot_df = pd.DataFrame({
            "AI": np.concatenate([predictions[i]['mask'] for i in indices]),
            "Majority": np.concatenate([majority[i] for i in indices])
        })

        IRA_human = compute_IRA_from_df(boot_df_human, metric)
        IRA_list.append(IRA_human)

        IRA_AI = compute_IRA_from_df(boot_df, metric)
        IRA_AI_list.append(IRA_AI)

    IRA = {'Human': round(np.mean(IRA_list), 10), 'AI': round(np.mean(IRA_AI_list), 10)}
    CI = {'Human': (round(np.mean(IRA_list) - 1.96 * np.std(IRA_list), 10), round(np.mean(IRA_list) + 1.96 * np.std(IRA_list), 10)), 'AI': (round(np.mean(IRA_AI_list) - 1.96 * np.std(IRA_AI_list), 10), round(np.mean(IRA_AI_list) + 1.96 * np.std(IRA_AI_list), 10))}
    return IRA, CI





