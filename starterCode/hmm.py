
# CSC 246 Project 3
# Qingjie Lu, qlu7
# Haoqi Zhang, hzhang84


import argparse
import math
import random
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import pickle

from nlputil import *
from time import time


# A utility class for bundling together relevant parameters - you may modify if you like.
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# num_states -- this should be an integer recording the number of hidden states
#
# pi -- this should be the distribution over the first hidden state of a sequence
#
# transitions -- this should be a num_states x num_states matrix of transition probabilities
#
# emissions -- this should be a num_states x vocab_size matrix of emission probabilities
#              (i.e., the probability of generating token X when operating in state K)
#
# vocab_size -- this should be an integer recording the vocabulary size
#
# Note: You may want to add fields for expectations.


class HMM:
    __slots__ = ('train_vocab', 'pi', 'transitions', 'emissions', 'num_states', 'vocab_size', 'train_data_size')

    # The constructor should initalize all the model parameters.
    def __init__(self, num_states, vocab_size, train_vocab, train_data_size):

        pi = np.random.rand(num_states)  # 1 * N

        # Normalization for pi
        factor = pi.sum()
        pi /= factor

        transitions = []  # N * N

        # Normalization for transitions
        for i in range(num_states):
            transitions.append([])
            factor = 0.0
            for _ in range(num_states):
                seed = 1.0 - random.uniform(0.0, 1.0)
                factor += seed
                transitions[i].append(seed)
            for j in range(num_states):
                transitions[i][j] /= factor
        transitions = np.asarray(transitions)

        emissions = []  # N * vocab_size

        # Normalization for emissions
        for i in range(num_states):
            emissions.append([])
            factor = 0.0
            for _ in range(vocab_size):
                seed = 1.0 - random.uniform(0.0, 1.0)
                factor += seed
                emissions[i].append(seed)
            for j in range(0, vocab_size):
                emissions[i][j] /= factor
        emissions = np.asarray(emissions)

        # Initialize HMM
        self.pi = pi
        self.transitions = transitions
        self.emissions = emissions
        self.num_states = num_states
        self.vocab_size = vocab_size
        self.train_vocab = train_vocab
        self.train_data_size = math.log10(train_data_size)

    def save(self, filename):
        with open(filename, 'wb') as fh:
            pickle.dump(self, fh)

    def load_hmm(filename):
        with open(filename, 'rb') as fh:
            return pickle.load(fh)

    # return the loglikelihood for a complete dataset (train OR test) (list of matrices)
    def loglikelihood(self, dataset):

        mean_log = 0.0
        dropped_file = 0.0
        print()
        print("Calculating Log Likelihood ...")
        begin = time()

        for sample in dataset:
            sample_log = self.loglikelihood_helper(sample, False)
            if sample_log == -math.inf:
                print("File Dropped")
                dropped_file += 1.0
            else:
                mean_log += sample_log
        mean_log /= (len(dataset) - dropped_file)

        end = time()
        print("Current Log Likelihood: " + str(mean_log) + ".")
        print('Done in', end - begin, 'seconds.')
        print()
        return mean_log

    # return the loglikelihood for a single sequence (numpy matrix)
    def loglikelihood_helper(self, sample, scale):
        pi = self.pi  # 1 * N
        transitions = self.transitions  # N * N
        emissions = self.emissions  # N * vocab_size
        num_states = self.num_states  # N

        T = len(sample)
        if T == 0:
            return -math.inf

        c = np.zeros(T, dtype=np.longdouble)  # 1 * T
        alpha = np.zeros((T, num_states), dtype=np.longdouble)  # T * N

        alpha[0] = np.multiply(pi, emissions[:, int(sample[0] - 1)].transpose())
        c[0] = 1.0 / alpha[0].sum()
        alpha[0] *= c[0]

        for t in range(1, T):  # Normalize Alpha
            alpha[t] = alpha[t - 1].dot(transitions)
            alpha[t] = np.multiply(alpha[t], emissions[:, int(sample[t] - 1)].transpose())
            factor = alpha[t].sum()
            if factor == 0.0:
                return -math.inf
            c[t] = 1.0 / factor
            alpha[t] *= c[t]

        logProb = -np.log10(c).sum()  # Calculate and Scale Log Likelihood
        if scale is True:
            logProb -= math.log10(self.train_data_size)

        return logProb

    # apply a single step of the em algorithm to the model on all the training data,
    # which is most likely a python list of numpy matrices (one per sample).
    # Note: you may find it helpful to write helper methods for the e-step and m-step,
    def em_step(self, dataset, old_log_prob):

        begin = time()
        big_file = []

        for sample in dataset:
            for number in sample:
                big_file.append(number)

        T = len(big_file)

        pi = self.pi
        transitions = self.transitions
        emissions = self.emissions
        num_states = self.num_states

        # E-STEP, FORWARD, ALPHA
        print("Running Alpha...")
        c = np.zeros(T, dtype=np.longdouble)
        alpha = np.zeros((T, num_states), dtype=np.longdouble)
        alpha[0] = np.multiply(pi, emissions[:, int(big_file[0] - 1)].transpose())
        c[0] = 1.0 / alpha[0].sum()
        alpha[0] *= c[0]

        for t in range(1, T):
            alpha[t] = alpha[t - 1].dot(transitions)
            alpha[t] = np.multiply(alpha[t], emissions[:, int(big_file[t] - 1)].transpose())
            c[t] = 1.0 / alpha[t].sum()
            alpha[t] *= c[t]

        # E-STEP, BACKWARD, BETA
        print("Running Beta...")
        beta = np.zeros((T, num_states), dtype=np.longdouble)
        gamma1 = np.zeros((T, num_states), dtype=np.longdouble)
        gamma2 = np.zeros((T, num_states, num_states), dtype=np.longdouble)
        beta[(T - 1), :] = c[T - 1]

        for t in range(T - 2, -1, -1):
            beta[t] = transitions.dot(np.multiply(beta[t + 1].transpose(),
                                                  emissions[:, int(big_file[t + 1] - 1)])).transpose()
            beta[t] *= c[t]

        # E-STEP, VITERBI, GAMMA
        print("Running Gamma...")
        for t in range(T - 1):
            gamma2[t] = np.multiply(transitions, alpha[t].transpose()
                                    .dot(np.multiply(beta[t + 1],
                                                     emissions[:, int(big_file[t + 1] - 1)].transpose())))
            gamma1[t] = np.sum(gamma2[t], axis=0)
        gamma1[T - 1] = np.copy(alpha[T - 1])

        # M-STEP, PARAMETER RE-ESTIMATION
        print("Parameter Re-Estimation...")
        pi = np.copy(gamma1[0])

        denom1 = gamma1.sum(axis=0)  # 1 * N
        numer1 = gamma2.sum(axis=0)  # N * N
        transitions = numer1 / denom1[:, None]

        denom = gamma1.sum(axis=0)  # 1 * N
        numer = np.zeros((num_states, self.vocab_size))

        for t in range(T):
            numer[:, int(big_file[t] - 1)] += gamma1[t].transpose()
        emissions = numer / denom[:, None]

        self.pi = pi / pi.sum()  # Update pi

        sum_of_factors = transitions.sum(axis=1)
        self.transitions = transitions / sum_of_factors[:, None]  # Update transitions

        sum_of_factors = emissions.sum(axis=1)
        self.emissions = emissions / sum_of_factors[:, None]  # Update emissions

        end = time()
        print('EM Finished. Done in', end - begin, 'seconds.')

        # FINAL LOG LIKELIHOOD
        logProb = self.loglikelihood(dataset)
        if logProb > old_log_prob:
            return False, logProb  # EM NOT Converge
        return True, logProb  # EM Converge


################################ MAIN HERE ################################
################################ MAIN HERE ################################
################################ MAIN HERE ################################
################################ MAIN HERE ################################
################################ MAIN HERE ################################

def main():
    parser = argparse.ArgumentParser(description='Program to build and train a neural network.')

    parser.add_argument('--train_path_pos',
                        default='/Users/zhanghaoqi/Desktop/csc246p3/csc246project3/imdbFor246/train/pos',
                        help='Path to the training data directory.')
    parser.add_argument('--train_path_neg',
                        default='/Users/zhanghaoqi/Desktop/csc246p3/csc246project3/imdbFor246/train/neg',
                        help='Path to the training data directory.')
    parser.add_argument('--test_path_pos',
                        default='/Users/zhanghaoqi/Desktop/csc246p3/csc246project3/imdbFor246/test/pos',
                        help='Path to the testing data directory.')
    parser.add_argument('--test_path_neg',
                        default='/Users/zhanghaoqi/Desktop/csc246p3/csc246project3/imdbFor246/test/neg',
                        help='Path to the testing data directory.')

    parser.add_argument('--max_iters', type=int, default=100,
                        help='The maximum number of EM iterations.')
    parser.add_argument('--hidden_states', type=int, default=3,
                        help='The number of hidden states to use.')
    parser.add_argument('--train_data_size', type=int, default=200,
                        help='Training data size.')
    parser.add_argument('--test_data_size', type=int, default=100000,
                        help='Testing data size.')

    args = parser.parse_args()

    print()
    train_data1, train_vocab1 = parse_data(args.train_path_pos, args.train_data_size)
    print()
    train_data2, train_vocab1 = parse_data_with_existing_vocab(args.train_path_neg,
                                                               train_vocab1, args.train_data_size)

    print("=======================================================================================")

    print(train_vocab1)

    hmm1 = HMM(args.hidden_states, len(train_vocab1), train_vocab1, len(train_data1))  # Positive HMM
    new_pos_pi = hmm1.pi
    new_pos_transitions = hmm1.transitions
    new_pos_emissions = hmm1.emissions

    hmm2 = HMM(args.hidden_states, len(train_vocab1), train_vocab1, len(train_data2))  # Negative HMM
    new_neg_pi = hmm2.pi
    new_neg_transitions = hmm2.transitions
    new_neg_emissions = hmm2.emissions

    # Positive EM
    logProb = hmm1.loglikelihood(train_data1)
    logProb_pos = [round(logProb, 2)]
    x_pos = [int(0)]
    converge = False
    for i in range(1, args.max_iters + 1):
        print("POSITIVE EM algorithm working ... Iteration(s) #" + str(i) + " ...")
        converge, logProb = hmm1.em_step(train_data1, logProb)
        logProb_pos.append(round(logProb, 2))  # Drawing, y-axis
        x_pos.append(i)  # Drawing, x-axis
        new_pos_pi = hmm1.pi
        new_pos_transitions = hmm1.transitions
        new_pos_emissions = hmm1.emissions
        if converge is True:
            print("Congratulations. EM algorithm converges after " + str(i) + " iteration(s).")
            break
    if converge is False:
        print("POSITIVE EM algorithm is not converging.")

    print("=======================================================================================")

    # Negative EM
    logProb = hmm2.loglikelihood(train_data2)
    logProb_neg = [round(logProb, 2)]
    x_neg = [int(0)]
    converge = False
    for i in range(1, args.max_iters + 1):
        print("NEGATIVE EM algorithm working ... Iteration(s) #" + str(i) + " ...")
        converge, logProb = hmm2.em_step(train_data2, logProb)
        logProb_neg.append(round(logProb, 2))
        x_neg.append(i)
        new_neg_pi = hmm2.pi
        new_neg_transitions = hmm2.transitions
        new_neg_emissions = hmm2.emissions
        if converge is True:
            print("Congratulations. EM algorithm converges after " + str(i) + " iteration(s).")
            break
    if converge is False:
        print("NEGATIVE EM algorithm is not converging.")

    # Update Positive HMM
    hmm1.pi = new_pos_pi
    hmm1.transitions = new_pos_transitions
    hmm1.emissions = new_pos_emissions

    # Update Negative HMM
    hmm2.pi = new_neg_pi
    hmm2.transitions = new_neg_transitions
    hmm2.emissions = new_neg_emissions

    # Save Two Models
    hmm1.save("PositiveHMM_"+str(args.hidden_states)+"_"+str(args.train_data_size))
    hmm2.save("NegativeHMM_"+str(args.hidden_states)+"_"+str(args.train_data_size))

    # Plotting Log Likelihood
    plt.plot(x_pos, logProb_pos, marker='o', color='blue', linewidth=3)
    plt.title('MEAN Log Likelihood for Training: POSITIVE', size=14)
    plt.xlabel('Iterations before Converge', size=12)
    plt.ylabel('Log Likelihood', size=12)
    plt.savefig('loglikelihood_plot_positive_'+str(args.hidden_states)+"_"+str(args.train_data_size))
    plt.clf()
    plt.plot(x_neg, logProb_neg, marker='o', color='blue', linewidth=3)
    plt.title('MEAN Log Likelihood for Training: NEGATIVE', size=14)
    plt.xlabel('Iterations before Converge', size=12)
    plt.ylabel('Log Likelihood', size=12)
    plt.savefig('loglikelihood_plot_negative_'+str(args.hidden_states)+"_"+str(args.train_data_size))
    plt.clf()
    print("=======================================================================================")
    print()
    print("Testing...")

    paths_test_pos = [args.test_path_pos]
    paths_test_neg = [args.test_path_neg]

    test_pos = []
    for path in paths_test_pos:
        for filename in os.listdir(path):
            with open(os.path.join(path, filename)) as fh:
                answer = convert_chars_to_ints(fh.read(), train_vocab1)
                if answer is not None:
                    test_pos.append(answer)
                else:
                    print("Drop Positive Files.")

    print("=======================================================================================")

    test_neg = []
    for path in paths_test_neg:
        for filename in os.listdir(path):
            with open(os.path.join(path, filename)) as fh:
                answer = convert_chars_to_ints(fh.read(), train_vocab1)
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
    plt.savefig('accuracy_plot_'+str(args.hidden_states)+"_"+str(args.train_data_size))

    print()
    print("Program Finishes.")


if __name__ == '__main__':
    main()


# end of hmm.py
