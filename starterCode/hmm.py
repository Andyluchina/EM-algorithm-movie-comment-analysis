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


def testing_likelihood(sample, pi, transitions, emissions, num_states):
    T = len(sample)
    if T == 0:
        return -math.inf

    c = [0.0] * T

    alpha = [[0] * num_states]
    for i in range(num_states):
        alpha[0][i] = pi[i] * emissions[i][int(sample[0] - 1)]
        c[0] += alpha[0][i]
    if c[0] == 0:
        return -math.inf
    c[0] = 1.0 / c[0]
    for i in range(num_states):
        alpha[0][i] = c[0] * alpha[0][i]

    for t in range(1, T):
        alpha.append([0] * num_states)
        for i in range(num_states):
            for j in range(num_states):
                alpha[t][i] += alpha[t - 1][j] * transitions[j][i]
            alpha[t][i] *= emissions[i][int(sample[t] - 1)]
            c[t] += alpha[t][i]
        c[t] = 1.0 / c[t]
        for i in range(num_states):
            alpha[t][i] *= c[t]

    logProb = 0.0
    # print(c)
    for i in range(T):
        logProb += math.log10(c[i])
    logProb = -logProb
    return logProb


class HMM:
    __slots__ = ('pi', 'transitions', 'emissions', 'num_states', 'vocab_size')

    def init_N_by_N_matrix(N):  # N = number of hidden states
        return np.zeros((N,N))

    # The constructor should initalize all the model parameters.
    def __init__(self, num_states, vocab_size):

        factor = 0.0  # normalization

        pi = np.random.rand(num_states)  # 1 * N
        factor = pi.sum()
        pi = pi/factor


        transitions = []  # N * N
        for i in range(num_states):
            transitions.append([])
            factor = 0.0
            for _ in range(num_states):
                seed = 1.0 - random.uniform(0.0, 1.0)
                factor += seed
                transitions[i].append(seed)
            for j in range(num_states):
                transitions[i][j] /= factor

        emissions = []  # N * vocab_size
        for i in range(num_states):
            emissions.append([])
            factor = 0.0
            for _ in range(vocab_size):
                seed = 1.0 - random.uniform(0.0, 1.0)
                factor += seed
                emissions[i].append(seed)
            for j in range(0, vocab_size):
                emissions[i][j] /= factor

        transitions = np.asarray(transitions)
        emissions = np.asarray(emissions)

        # initialize HMM
        self.pi = pi
        self.transitions = transitions
        self.emissions = emissions
        self.num_states = num_states
        self.vocab_size = vocab_size

    # return the loglikelihood for a complete dataset (train OR test) (list of matrices)
    def loglikelihood(self, dataset, data_size):

        mean_log = 0.0
        dropped_file = 0.0
        print()
        print("Calculating Log Likelihood ...")
        begin = time()

        for i in range(data_size):
            sample_log = HMM.loglikelihood_helper(self, dataset[i])
            # drop file
            if sample_log == -math.inf or sample_log == math.inf:
                print("dropped_file")
                dropped_file += 1.0
            else:
                mean_log += sample_log
        mean_log /= (data_size - dropped_file)

        end = time()
        print("Current Log Likelihood: " + str(mean_log) + ".")
        print('Done in', end - begin, 'seconds.')
        print()
        return mean_log

    # return the loglikelihood for a single sequence (numpy matrix)
    def loglikelihood_helper(self, sample):
        pi = self.pi # 1 * N
        transitions = self.transitions # N * N
        emissions = self.emissions # N * vocab_size
        num_states = self.num_states # N
        # print(sample) # 1* T
        T = len(sample)
        if T == 0:
            return -math.inf

        c = np.zeros(T) #1*T
        alpha = np.zeros((T,num_states)) #T * N
        # for i in range(num_states):
        #     alpha[0][i] = pi[i] * emissions[i][int(sample[0] - 1)]
        #     c[0] += alpha[0][i]
        alpha[0] = np.multiply(pi,  emissions[:, int(sample[0] - 1)].transpose())
        c[0] = alpha[0].sum()
        c[0] = 1.0 / c[0]
        alpha[0] = alpha[0] * c[0]
        # for i in range(num_states):
        #     alpha[0][i] = c[0] * alpha[0][i]
        for t in range(1, T):
            alpha[t] = alpha[t - 1].dot(transitions)
            alpha[t] = np.multiply(alpha[t],  emissions[:, int(sample[t] - 1)].transpose())
            c[t] = alpha[t].sum()
            c[t] = 1.0 / c[t]
            alpha[t] = alpha[t] * c[t]

        # for t in range(1, T):
        #     for i in range(num_states):
        #         for j in range(num_states):
        #             alpha[t][i] += (alpha[t - 1][j] * transitions[j][i])
        #             # print(transitions[j][i])
        #         alpha[t][i] *= emissions[i][int(sample[t] - 1)]
        #         c[t] += alpha[t][i]
        #     # print(alpha[t - 1].dot(transitions))
        #     c[t] = 1.0 / c[t]
        #     for i in range(num_states):
        #         alpha[t][i] *= c[t]
        logProb = np.log10(c).sum()
        print(c)
        # for i in range(T):
        #     logProb += math.log10(c[i])
        #     print(math.log10(c[i]))
        logProb = -logProb

        return logProb

    # apply a single step of the em algorithm to the model on all the training data,
    # which is most likely a python list of numpy matrices (one per sample).
    # Note: you may find it helpful to write helper methods for the e-step and m-step,
    def em_step(self, dataset, old_log_prob, data_size):

        count = 0
        begin = time()

        for index in range(data_size):

            sample = dataset[index]

            char_set = set()
            for num in sample:
                char_set.add(int(num - 1))

            count += 1
            print("EM sample " + str(count) + "/" + str(data_size))

            T = len(sample)
            if T == 0:
                continue

            pi = self.pi
            transitions = self.transitions
            emissions = self.emissions
            num_states = self.num_states

            # print()
            # print("***** START DEBUG HERE !!!!! (DO NOT DELETE THIS) *****")
            # print()
            # print(pi)
            # print()
            # print(transitions)
            # print()
            # print(emissions)
            # print()
            # print("***** END DEBUG HERE !!!!!  (DO NOT DELETE THIS) *****")
            # print()

            # E-STEP, FORWARD, ALPHA
            c = np.zeros(T)

            alpha = np.zeros((T,num_states))

            alpha[0] = np.multiply(pi,  emissions[:, int(sample[0] - 1)].transpose())
            c[0] = alpha[0].sum()
            c[0] = 1.0 / c[0]
            alpha[0] = alpha[0] * c[0]


            for t in range(1, T):
                alpha[t] = alpha[t - 1].dot(transitions)
                alpha[t] = np.multiply(alpha[t],  emissions[:, int(sample[t] - 1)].transpose())
                c[t] = alpha[t].sum()
                c[t] = 1.0 / c[t]
                alpha[t] = alpha[t] * c[t]

            # E-STEP, BACKWARD, BETA
            beta = []
            gamma1 = []
            gamma2 = []

            beta = np.zeros((T, num_states))
            gamma1 = np.zeros((T, num_states))
            gamma2 = np.zeros((T, num_states, num_states))
            beta[(T - 1), :] = c[T - 1]
            # for i in range(T):
            #     beta.append([0] * self.num_states)
            #     gamma1.append([0] * self.num_states)
            #     gamma2.append(HMM.init_N_by_N_matrix(self.num_states))

            # for i in range(num_states):
            #     beta[T - 1][i] = c[T - 1]
            # pi = self.pi # 1 * N
            # transitions = self.transitions # N * N
            # emissions = self.emissions # N * vocab_size
            # num_states = self.num_states # N
            for t in range(T - 2, -1, -1):
                beta[t] = transitions.dot(np.multiply(beta[t + 1].transpose(), emissions[:, int(sample[t + 1] - 1)])).transpose()
                beta[t] *= c[t]
            # for t in range(T - 2, -1, -1):
            #     for i in range(num_states):
            #         for j in range(num_states):
            #             beta[t][i] += (transitions[i][j] * emissions[j][int(sample[t + 1] - 1)] * beta[t + 1][j])
            #         beta[t][i] *= c[t]
            # alpha T * N
            # E-STEP, VITERBI, GAMMA
            for t in range(T - 1):
                gamma2[t] = np.multiply(transitions, alpha[t].transpose().dot(np.multiply(beta[t + 1], emissions[:,int(sample[t + 1] - 1)].transpose())))
                gamma1[t] = np.sum(gamma2[t], axis=0)
            # for t in range(T - 1):
            #     for i in range(num_states):
            #         for j in range(num_states):
            #             gamma2[t][i][j] = alpha[t][i] * transitions[i][j] * \
            #                                   emissions[j][int(sample[t + 1] - 1)] * beta[t + 1][j]
            #             gamma1[t][i] += gamma2[t][i][j]
            gamma1[T - 1] = np.copy(alpha[T - 1])
            # for i in range(num_states):
            #     gamma1[T - 1][i] = alpha[T - 1][i]

            # M-STEP, PARAMETER RE-ESTIMATION
            pi = np.copy(gamma1[0])
            # for i in range(num_states):
            #     pi[i] = gamma1[0][i]

            for i in range(num_states):
                denom = 0
                for t in range(T - 1):
                    denom += gamma1[t][i]
                for j in range(num_states):
                    numer = 0
                    for t in range(T - 1):
                        numer += gamma2[t][i][j]
                    transitions[i][j] = numer / denom

            for i in range(num_states):
                denom = 0.0
                for t in range(T):
                    denom += gamma1[t][i]
                for j in char_set:
                    numer = 0.0
                    for t in range(T):
                        if sample[t] - 1 == j:
                            numer += gamma1[t][i]
                    emissions[i][j] = numer / denom


            pi = pi / pi.sum()
            # factor = 0.0
            #
            # for i in range(num_states):
            #     factor += pi[i]
            # for i in range(num_states):
            #     pi[i] /= factor
            sum_of_factors = transitions.sum(axis=1)
            transitions = transitions / sum_of_factors[:,None]
            # for i in range(num_states):
            #     factor = 0.0
            #     for j in range(num_states):
            #         factor += transitions[i][j]
            #     for j in range(num_states):
            #         transitions[i][j] /= factor
            sum_of_factors = emissions.sum(axis=1)
            emissions = emissions / sum_of_factors[:,None]
            # for i in range(num_states):
            #     factor = 0.0
            #     for j in range(len(emissions[i])):
            #         factor += emissions[i][j]
            #     for j in range(len(emissions[i])):
            #         emissions[i][j] /= factor

            self.pi = pi
            self.transitions = transitions
            self.emissions = emissions

        end = time()
        print('EM Finished. Done in', end - begin, 'seconds.')

        # FINAL LOG LIKELIHOOD
        logProb = HMM.loglikelihood(self, dataset, data_size)
        if logProb > old_log_prob:
            return False, logProb
        return True, logProb

    # Return a "completed" sample by adding additional steps based on model probability.
    def complete_sequence(self, sample, steps):
        pass


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
    parser.add_argument('--max_iters', type=int, default=10,
                        help='The maximum number of EM iterations.')
    parser.add_argument('--hidden_states', type=int, default=5,
                        help='The number of hidden states to use.')
    parser.add_argument('--train_data_size', type=int, default=200,
                        help='Training data size.')
    parser.add_argument('--test_data_size', type=int, default=1000,
                        help='Testing data size.')

    args = parser.parse_args()

    print()
    train_data1, train_vocab1 = parse_data(args.train_path_pos)
    print()
    train_data2 = parse_data_with_existing_vocab(args.train_path_neg, train_vocab1)
    print("=======================================================================================")

    print(train_vocab1)
    hmm1 = HMM(args.hidden_states, len(train_vocab1))
    hmm2 = HMM(args.hidden_states, len(train_vocab1))

    new_pos_pi = hmm1.pi
    new_pos_transitions = hmm1.transitions
    new_pos_emissions = hmm1.emissions

    new_neg_pi = hmm2.pi
    new_neg_transitions = hmm2.transitions
    new_neg_emissions = hmm2.emissions

    logProb = hmm1.loglikelihood(train_data1, args.train_data_size)
    logProb_pos = [round(logProb, 2)]  # Drawing, y-axis
    x_pos = [int(0)]  # Drawing, x-axis

    converge = False
    for i in range(1, args.max_iters + 1):
        print("POSITIVE EM algorithm working ... Iteration(s) #" + str(i) + " ...")
        converge, logProb = hmm1.em_step(train_data1, logProb, args.train_data_size)
        logProb_pos.append(round(logProb, 2))  # Drawing, y-axis
        x_pos.append(i)  # Drawing, x-axis
        if converge is True:
            print("Congratulations. EM algorithm converges after " + str(i) + " iteration(s).")
            break
        else:
            new_pos_pi = hmm1.pi
            new_pos_transitions = hmm1.transitions
            new_pos_emissions = hmm1.emissions
    if converge is False:
        print("POSITIVE EM algorithm is not converging.")

    print("=======================================================================================")

    logProb = hmm2.loglikelihood(train_data2, args.train_data_size)
    logProb_neg = [round(logProb, 2)]
    x_neg = [int(0)]
    converge = False
    for i in range(1, args.max_iters + 1):
        print("NEGATIVE EM algorithm working ... Iteration(s) #" + str(i) + " ...")
        converge, logProb = hmm2.em_step(train_data2, logProb, args.train_data_size)
        logProb_neg.append(round(logProb, 2))
        x_neg.append(i)
        if converge is True:
            print("Congratulations. EM algorithm converges after " + str(i) + " iteration(s).")
            break
        else:
            new_neg_pi = hmm2.pi
            new_neg_transitions = hmm2.transitions
            new_neg_emissions = hmm2.emissions
    if converge is False:
        print("NEGATIVE EM algorithm is not converging.")

    # print()
    # print(new_pos_pi)
    # print()
    # print(new_pos_transitions)
    # print()
    # print(new_pos_emissions)
    # print()
    # print(new_neg_pi)
    # print()
    # print(new_neg_transitions)
    # print()
    # print(new_neg_emissions)
    # print()

    # plt.plot(x_pos, logProb_pos, marker='o', color='blue', linewidth=3)
    # plt.title('MEAN Log Likelihood for Training: POSITIVE', size=14)
    # plt.xlabel('Iterations before Converge', size=12)
    # plt.ylabel('Log Likelihood', size=12)
    # plt.show()
    #
    # plt.plot(x_neg, logProb_neg, marker='o', color='blue', linewidth=3)
    # plt.title('MEAN Log Likelihood for Training: NEGATIVE', size=14)
    # plt.xlabel('Iterations before Converge', size=12)
    # plt.ylabel('Log Likelihood', size=12)
    # plt.show()

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
    test_neg = []
    for path in paths_test_neg:
        for filename in os.listdir(path):
            with open(os.path.join(path, filename)) as fh:
                answer = convert_chars_to_ints(fh.read(), train_vocab1)
                if answer is not None:
                    test_neg.append(answer)

    accurate_sample = 0
    total_sample = 0

    count = 0

    test_pos_num = len(test_pos)
    test_neg_num = len(test_neg)

    for sample in test_pos:
        count += 1
        if count > args.test_data_size:
            break
        log_prob1 = testing_likelihood(sample, new_pos_pi, new_pos_transitions,
                                       new_pos_emissions, args.hidden_states)
        log_prob2 = testing_likelihood(sample, new_neg_pi, new_neg_transitions,
                                       new_neg_emissions, args.hidden_states)
        total_sample += 1
        if log_prob1 > log_prob2:
            print("Positive Sample " + str(count) + "/" + str(test_neg_num) +
                  " CORRECT:  " + str(log_prob1) + " > " + str(log_prob2) + ".")
            accurate_sample += 1
        else:
            print("Positive Sample " + str(count) + "/" + str(test_pos_num) + " WRONG!")

    print()
    count = 0

    for sample in test_neg:
        count += 1
        if count > args.test_data_size:
            break
        log_prob1 = testing_likelihood(sample, new_pos_pi, new_pos_transitions,
                                       new_pos_emissions, args.hidden_states)
        log_prob2 = testing_likelihood(sample, new_neg_pi, new_neg_transitions,
                                       new_neg_emissions, args.hidden_states)
        if log_prob1 == log_prob2 or log_prob1 == -math.inf or log_prob2 == -math.inf:
            continue
        total_sample += 1
        if log_prob1 < log_prob2:
            print("Negative Sample " + str(count) + "/" + str(test_neg_num) +
                  " CORRECT:  " + str(log_prob1) + " < " + str(log_prob2) + ".")
            accurate_sample += 1
        else:
            print("Negative Sample " + str(count) + "/" + str(test_neg_num) + " WRONG!")

    print()
    print("Total Accurate Sample: " + str(int(accurate_sample)))
    print("Total Tested Sample: " + str(int(total_sample)))
    print("Total Accuracy: " + str(accurate_sample / total_sample))
    print()
    print("Program Finishes.")


if __name__ == '__main__':
    main()
