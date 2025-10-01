from gen_annotations_method_A import *

# Set parameters
num_signals = 500
n_points = 3600 * 2 # 2 hours, per second annotations
num_annotators = 30
prevalence = 20 # Adjust prevalence for ground truth


# Define groups: shift from ground truth, variability within group, number of annotators in each group
group_shifts = [0, -0.16, 0.15]
group_variabilities = [0.1, 0.16, 0.15] # std values
group_proportions_list = [[i, int(np.floor((num_annotators - i)/2)), int(np.ceil((num_annotators - i)/2))] for i in range(1, num_annotators)]

for group_proportions in group_proportions_list:

    annotations, group_sizes = generate_annotations_method_A(num_signals, n_points, num_annotators,
                                                    prevalence, group_shifts, group_variabilities, group_proportions)
    # Compute Fleiss' kappa
    fleiss_all = compute_fleiss_kappa(annotations[:, :, 1:])
    fleiss_groups = compute_fleiss_by_group(annotations[:, :, 1:], group_sizes)
    fleiss_2 = compute_fleiss_kappa(annotations[:, :, group_proportions[0]+1:])

    # Calculate ratio of 1s to 0s for each annotator
    ratios = np.mean(annotations, axis=(0, 1))  # Mean of 1s across signals and time steps

    # Print the ratio for each annotator
    #for i, ratio in enumerate(ratios):
    #    print(f'{(1 - ratio) / ratio:.4f}')

    # Print Fleiss Kappa values for each group
    print(f"Fleiss' Kappa (All Annotators): {fleiss_all:.4f}")
    print(f"Fleiss' Kappa (2+3 group): {fleiss_2:.4f}")
    for i, fk in enumerate(fleiss_groups):
        print(f"Fleiss' Kappa (Group {i+1}): {fk:.4f}")