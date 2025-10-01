from utils import *

def generate_annotations_method_B(num_signals, n_points, error_rates, case):
    """
    This function generates annotations using method B.
    :param num_signals: number of annotations in dataset that needs to be created
    :param n_points: how long is each annotation (how many seconds)
    :param error_rates: percent of zeros/ones to be misclassified, relative to the total count of zeros/ones
    :param case: target imbalance in dataset
    :return: matrix of created annotations
    """
    alpha, beta = compute_beta_params(case) # create beta distribution parameters based on desired imbalance
    mean = np.random.beta(alpha, beta, n_points * num_signals) # sample ground truth probabilities from that beta distribution
    annotations = np.zeros((len(error_rates) + 1, num_signals * n_points)) # matrix for annotations

    signal, signal_probabilities = generate_signal(n_points * num_signals, mean, 0)
    signal = signal.flatten()
    signal_probabilities = signal_probabilities.flatten()

    annotations[0, :] = signal.copy()  # Expert annotation, or ground truth annotation

    zeros_indices = np.where(signal == 0)[0]  # Find zeros and ones in original signal
    ones_indices = np.where(signal == 1)[0]

    for i, percent in enumerate(error_rates):
        num_zeros_to_flip = int(np.ceil(len(zeros_indices) * percent)) # percent to number of zeros to flip relative to zero count in the signal
        num_ones_to_flip = int(np.ceil(len(ones_indices) * percent)) # percent to number of ones to flip relative to ones count in the signal
        zeros_to_flip = np.random.choice(zeros_indices, num_zeros_to_flip, replace=False) # randomly choose zeros to flip
        ones_to_flip = np.random.choice(ones_indices, num_ones_to_flip, replace=False) # randomly choose ones to flip

        # Flip zeros and ones (make FP and FN)
        signal_probs = signal_probabilities.copy()
        signal_probs[zeros_to_flip] = 1 - signal_probs[zeros_to_flip]
        signal_probs[ones_to_flip] = 1 - signal_probs[ones_to_flip]

        # Save that prediction
        annotations[i + 1, :] = signal_probs  # Prediction

        # Save annotations for imbalance case
        output_file_name = 'Annotations_' + str(case) + '.npy'
        #np.save(output_file_name, annotations)

    return annotations
