
# CSC 246 Project 3
# Qingjie Lu, qlu7
# Haoqi Zhang, hzhang84


import argparse
import numpy as np
import os
import sys
import math

from hmm import HMM
from nlputil import *
import pickle


def main():
    parser = argparse.ArgumentParser(description='Evaluate a model on a test file.')
    parser.add_argument('--modelpos', help='Path to the model file.', default='PositiveHMM')
    parser.add_argument('--modelneg', help='Path to the model file.', default='NegativeHMM')
    parser.add_argument('--test_path_pos',
                        help='Path to the positive test file.',
                        default='/Users/zhanghaoqi/Desktop/csc246p3/csc246project3/imdbFor246/test/pos')
    parser.add_argument('--test_path_neg',
                        help='Path to the negative test file.',
                        default='/Users/zhanghaoqi/Desktop/csc246p3/csc246project3/imdbFor246/test/neg')
    parser.add_argument('--test_data_size', type=int, default=5000,
                            help='Testing data size.')
    args = parser.parse_args()


    # Load the test data.
    hmm1 = HMM.load_hmm(args.modelpos)
    hmm2 = HMM.load_hmm(args.modelneg)

    print("Testing...")

    paths_test_pos = [args.test_path_pos]
    paths_test_neg = [args.test_path_neg]

    test_pos = []
    for path in paths_test_pos:
        for filename in os.listdir(path):
            with open(os.path.join(path, filename)) as fh:
                answer = convert_chars_to_ints(fh.read(), hmm1.train_vocab)
                if answer is not None:
                    test_pos.append(answer)
                else:
                    print("Drop Positive Files.")

    print("=======================================================================================")

    test_neg = []
    for path in paths_test_neg:
        for filename in os.listdir(path):
            with open(os.path.join(path, filename)) as fh:
                answer = convert_chars_to_ints(fh.read(), hmm2.train_vocab)
                if answer is not None:
                    test_neg.append(answer)
                else:
                    print("Drop Negative Files.")

    print("=======================================================================================")

    accurate_sample = 0
    total_sample = 0
    count = 0

    test_pos_num = len(test_pos)
    test_neg_num = len(test_neg)

    sample_count = [int(0)]
    accuracy_list = [0.0]

    for sample in test_pos:
        count += 1
        if count > args.test_data_size:
            break
        log_prob1 = hmm1.loglikelihood_helper(sample, False)
        log_prob2 = hmm2.loglikelihood_helper(sample, True)

        if math.isnan(log_prob1) is True and math.isnan(log_prob2) is True:
            print("File Dropped.")
            continue
        if log_prob1 == log_prob2:
            print("File Dropped.")
            continue

        total_sample += 1

        if math.isnan(log_prob2) is True and math.isnan(log_prob1) is not True:
            print("Positive Sample " + str(count) + "/" + str(test_pos_num) +
                  " CORRECT: NEGATIVE UNDERFLOW.")
            accurate_sample += 1
        elif math.isnan(log_prob1) is True and math.isnan(log_prob2) is not True:
            print("Positive Sample " + str(count) + "/" + str(test_pos_num) +
                  " WRONG: POSITIVE UNDERFLOW.")
        elif log_prob1 >= log_prob2:
            print("Positive Sample " + str(count) + "/" + str(test_pos_num) +
                  " CORRECT:  " + str(log_prob1) + " > " + str(log_prob2) + ".")
            accurate_sample += 1
        else:
            print("Positive Sample " + str(count) + "/" + str(test_pos_num) +
                  " WRONG:  " + str(log_prob1) + " < " + str(log_prob2) + ".")

        sample_count.append(int(total_sample))
        accuracy_list.append(accurate_sample / total_sample)

    print()
    print("=======================================================================================")
    print()

    count = 0

    for sample in test_neg:
        count += 1
        if count > args.test_data_size:
            break
        log_prob1 = hmm1.loglikelihood_helper(sample, True)
        log_prob2 = hmm2.loglikelihood_helper(sample, False)

        if math.isnan(log_prob1) is True and math.isnan(log_prob2) is True:
            print("File Dropped.")
            continue
        if log_prob1 == log_prob2:
            print("File Dropped.")
            continue

        total_sample += 1

        if math.isnan(log_prob1) is True and math.isnan(log_prob2) is not True:
            print("Positive Sample " + str(count) + "/" + str(test_pos_num) +
                  " CORRECT: POSITIVE UNDERFLOW.")
            accurate_sample += 1
        elif math.isnan(log_prob2) is True and math.isnan(log_prob1) is not True:
            print("Positive Sample " + str(count) + "/" + str(test_pos_num) +
                  " WRONG: NEGATIVE UNDERFLOW.")
        elif log_prob2 >= log_prob1:
            print("Negative Sample " + str(count) + "/" + str(test_neg_num) +
                  " CORRECT:  " + str(log_prob1) + " < " + str(log_prob2) + ".")
            accurate_sample += 1
        else:
            print("Negative Sample " + str(count) + "/" + str(test_neg_num) +
                  " WRONG:  " + str(log_prob1) + " > " + str(log_prob2) + ".")

        sample_count.append(int(total_sample))
        accuracy_list.append(accurate_sample / total_sample)

    print()
    print("Total Accurate Sample: " + str(int(accurate_sample)))
    print("Total Effective Sample: " + str(int(total_sample)))
    print("Total Accuracy: " + str(accurate_sample / total_sample))

    plt.plot(sample_count, accuracy_list, color='green', linewidth=2)
    plt.title('Testing Accuracy Over Time', size=14)
    plt.xlabel('Effective Sample Count (Positive Labeled First)', size=12)
    plt.ylabel('Testing Accuracy', size=12)
    plt.show()

    print()
    print("Program Finishes.")


if __name__ == '__main__':
    main()
