from utils import *

def generate_annotations_method_A(num_signals, n_points, num_annotators, prevalence,
                         group_shifts, group_variabilities, group_proportions):
    """ Generate group-specific base probability sequence for multiple groups with controlled parameters.
    :param num_signals: number of annotations that needs to be created
    :param n_points: duration of annotations in seconds
    :param num_annotators: number of annotators
    :param prevalence: prevalence of ground truth annotation
    :param group_shifts: group shifts of each group (overraters or underraters)
    :param group_variabilities: group variabilities of each group (used as std in Gaussian distribution)
    :param group_proportions: number of annotators in each group
    :return: created annotations
    """

    # Convert prevalence into Beta distribution parameters
    alpha, beta = compute_beta_params(prevalence)

    # Determine group sizes based on proportions
    group_sizes = group_proportions # Number of annotators in each group

    # Sample mean probabilities from Beta distribution for ground truth - Ground Truth Generation
    mean_probs = np.random.beta(alpha, beta, (num_signals, n_points))

    # Create probability shifts for each group
    group_means = [
        mean_probs.copy() + np.random.uniform(0, shift, (num_signals, n_points))
        for shift in group_shifts
    ]

    # Store annotations
    annotations_output = np.zeros((num_signals, n_points, num_annotators + 1), dtype=int) # init
    annotations_output[:, :, 0] = (mean_probs.reshape(num_signals, n_points) > 0.5) # Ground truth

    annotator_idx = 1
    # Generate specific annotators from each group
    for g, (group_mean, variability, group_size) in enumerate(zip(group_means, group_variabilities, group_sizes)):
        for group_number in range(group_size):
            signal, signal_probs = generate_signal(n_points * num_signals, group_mean.flatten(), variability)
            annotations_output[:, :, annotator_idx] = signal.reshape(num_signals, n_points)
            annotator_idx += 1

    return annotations_output, group_sizes
