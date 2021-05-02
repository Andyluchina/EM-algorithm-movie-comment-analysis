import argparse
import numpy as np
import hmm
import os


def main():
    parser = argparse.ArgumentParser(description='Evaluate a model on a test file.')
    parser.add_argument('--modelpos', help='Path to the model file.', default='PositiveHMM')
    parser.add_argument('--modelneg', help='Path to the model file.', default='NegativeHMM')
    parser.add_argument('--test_path_pos', help='Path to the test file.', default='../imdbFor246/test/pos')
    parser.add_argument('--test_path_neg', help='Path to the test file.', default='../imdbFor246/test/neg')
    parser.add_argument('--test_data_size', type=int, default=2000,
                            help='Testing data size.')
    args = parser.parse_args()

    # Load the test data.

    # Load the mlp.
    hmm1 = hmm.HMM.load_mlp(args.modelpos)
    hmm2 = hmm.HMM.load_mlp(args.modelneg)

    print("=======================================================================================")
    print()
    print("Testing...")

    paths_test_pos = [args.test_path_pos]
    paths_test_neg = [args.test_path_neg]

    test_pos = []
    for path in paths_test_pos:
        for filename in os.listdir(path):
            with open(os.path.join(path, filename)) as fh:
                answer = convert_chars_to_ints(fh.read(), hmm1.train_vocab1)
                if answer is not None:
                    test_pos.append(answer)
                else:
                    print("Drop Positive Files.")
    test_neg = []
    for path in paths_test_neg:
        for filename in os.listdir(path):
            with open(os.path.join(path, filename)) as fh:
                answer = convert_chars_to_ints(fh.read(), hmm2.train_vocab1)
                if answer is not None:
                    test_neg.append(answer)
                else:
                    print("Drop Negative Files.")

    accurate_sample = 0
    total_sample = 0
    count = 0

    test_pos_num = len(test_pos)
    test_neg_num = len(test_neg)

    for sample in test_pos:
        count += 1
        if count > args.test_data_size:
            break
        log_prob1 = hmm1.loglikelihood_helper(sample)
        log_prob2 = hmm2.loglikelihood_helper(sample)

        if log_prob1 == -math.inf:
            print("log_prob1 goes to negative infinity")
            continue
        if log_prob2 == -math.inf:
            print("log_prob2 goes to negative infinity")
            continue
        if math.isnan(log_prob1):
            print("log_prob1 is nan")
            continue
        if math.isnan(log_prob2):
            print("log_prob2 is nan")
            continue

        total_sample += 1
        if log_prob1 > log_prob2:
            print("Positive Sample " + str(count) + "/" + str(test_pos_num) +
                  " CORRECT:  " + str(log_prob1) + " > " + str(log_prob2) + ".")
            accurate_sample += 1
        else:
            print("Positive Sample " + str(count) + "/" + str(test_pos_num) +
                  " WRONG:  " + str(log_prob1) + " <= " + str(log_prob2) + ".")

    print()
    count = 0

    for sample in test_neg:
        count += 1
        if count > args.test_data_size:
            break
        log_prob1 = hmm1.loglikelihood_helper(sample)
        log_prob2 = hmm2.loglikelihood_helper(sample)

        if log_prob1 == -math.inf:
            print("log_prob1 goes to negative infinity")
            continue
        if log_prob2 == -math.inf:
            print("log_prob2 goes to negative infinity")
            continue
        if math.isnan(log_prob1):
            print("log_prob1 is nan")
            continue
        if math.isnan(log_prob2):
            print("log_prob2 is nan")
            continue

        total_sample += 1
        if log_prob1 < log_prob2:
            print("Negative Sample " + str(count) + "/" + str(test_neg_num) +
                  " CORRECT:  " + str(log_prob1) + " < " + str(log_prob2) + ".")
            accurate_sample += 1
        else:
            print("Negative Sample " + str(count) + "/" + str(test_neg_num) +
                  " WRONG:  " + str(log_prob1) + " >= " + str(log_prob2) + ".")

    print()
    print("Total Accurate Sample: " + str(int(accurate_sample)))
    print("Total Tested Sample: " + str(int(total_sample)))
    print("Total Accuracy: " + str(accurate_sample / total_sample))
    print()
    print("Program Finishes.")

if __name__ == '__main__':
    main()
