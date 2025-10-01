from gen_annotations_method_B import *

n_points = 3600 * 2  # 2h - each signal length
num_signals = 500 # number of signals in dataset
error_rates = [0.1, 0.2] # Desired error rates for each class separately
case_list = [50, 45, 40, 35, 30, 25, 20, 15, 10, 5, 1]

for case in case_list:
    annotations = generate_annotations_method_B(n_points, num_signals, error_rates, case)
    print(f"Case {case}:")
    print((num_signals * n_points - sum(annotations[0, :])) / sum(annotations[0, :]))