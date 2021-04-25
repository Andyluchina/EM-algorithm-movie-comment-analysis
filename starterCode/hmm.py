
# CSC 246 Project 3
# Qingjie Lu, qlu7
# Haoqi Zhang, hzhang84


import argparse
import math
from nlputil import *  # utility methods for working with text


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
    __slots__ = ('pi', 'transitions', 'emissions', 'num_states', 'vocab_size')

    def init_N_by_N_matrix(self, N):
        matrix = []
        for i in range(0, N):
            matrix.append([])
            for j in range(0, N):
                matrix[i].append(0)
        return matrix

    # The constructor should initalize all the model parameters.
    def __init__(self, num_states, vocab_size):
        pi = [1.0 / num_states] * num_states
        transitions = []
        for _ in range(0, num_states):
            transitions.append([1.0 / num_states] * num_states)
        emissions = []
        for _ in range(0, num_states):
            emissions.append([1.0 / vocab_size] * vocab_size)
        HMM.pi = pi
        HMM.transitions = transitions
        HMM.emissions = emissions
        HMM.num_states = num_states
        HMM.vocab_size = vocab_size

    # return the loglikelihood for a complete dataset (train OR test) (list of matrices)
    def loglikelihood(self, dataset):
        return -math.inf

    # return the loglikelihood for a single sequence (numpy matrix)
    def loglikelihood_helper(self, sample):
        pass

    # apply a single step of the em algorithm to the model on all the training data,
    # which is most likely a python list of numpy matrices (one per sample).
    # Note: you may find it helpful to write helper methods for the e-step and m-step,
    def em_step(self, dataset, old_log_prob):
        pi = HMM.pi
        transitions = HMM.transitions
        emissions = HMM.emissions

        print()
        print("***** START DEBUG HERE !!!!! (DO NOT DELETE THIS) *****")
        print()
        print(dataset)
        print()
        print(pi)
        print()
        print(transitions)
        print()
        print(emissions)
        print()
        print("***** END DEBUG HERE !!!!!  (DO NOT DELETE THIS) *****")
        print()

        T = len(dataset)

        # E-STEP, FORWARD, ALPHA
        c = [0] * len(dataset)

        alpha = [[0] * HMM.num_states]
        for i in range(0, HMM.num_states):
            alpha[0][i] = pi[i] * emissions[i][dataset[0]]
            c[0] += alpha[0][i]

        c[0] = 1.0 / c[0]
        for i in range(0, HMM.num_states):
            alpha[0][i] = c[0] * alpha[0][i]

        for t in range(1, T):
            alpha.append([0] * HMM.num_states)
            for i in range(0, HMM.num_states):
                for j in range(0, HMM.num_states):
                    alpha[t][i] += alpha[t - 1][j] * transitions[j][i]
                alpha[t][i] *= emissions[i][dataset[t]]
                c[t] += alpha[t][i]
            c[t] = 1.0 / c[t]
            for i in range(0, HMM.num_states):
                alpha[t][i] *= c[t]

        # E-STEP, BACKWARD, BETA
        beta = []
        gamma1 = []
        gamma2 = []

        for i in range(0, T):
            beta.append([0] * HMM.num_states)
            gamma1.append([0] * HMM.num_states)
            gamma2.append(HMM.init_N_by_N_matrix(self, HMM.num_states))

        for i in range(0, HMM.num_states):
            beta[T - 1][i] = c[T - 1]

        for t in range(T - 2, -1, -1):
            for i in range(0, HMM.num_states):
                for j in range(0, HMM.num_states):
                    beta[t][i] += (transitions[i][j] * emissions[j][dataset[t + 1]] * beta[t + 1][j])
                beta[t][i] *= c[t]

        # E-STEP, VITERBI, GAMMA
        for t in range(0, T - 1):
            for i in range(0, HMM.num_states):
                for j in range(0, HMM.num_states):
                    gamma2[t][i][j] = alpha[t][i] * transitions[i][j] * \
                                      emissions[j][dataset[t + 1]] * beta[t + 1][j]
                    gamma1[t][i] += gamma2[t][i][j]

        for i in range(0, HMM.num_states):
            gamma1[T - 1][i] = alpha[T - 1][i]

        # M-STEP, PARAMETER RE-ESTIMATION
        for i in range(0, HMM.num_states):
            pi[i] = gamma1[0][i]

        for i in range(0, HMM.num_states):
            denom = 0
            for t in range(0, T - 1):
                denom += gamma1[t][i]
            for j in range(0, HMM.num_states):
                numer = 0
                for t in range(0, T - 1):
                    numer += gamma2[t][i][j]
                transitions[i][j] = numer / denom

        for i in range(0, HMM.num_states):
            denom = 0
            for t in range(0, T):
                denom += gamma1[t][i]
            for j in range(0, HMM.vocab_size):
                numer = 0
                for t in range(0, T):
                    if dataset[t] == j:
                        numer += gamma1[t][i]
                emissions[i][j] = numer / denom

        HMM.pi = pi
        HMM.transitions = transitions
        HMM.emissions = emissions

        # LOG LIKELIHOOD
        logProb = 0
        for i in range(0, T):
            logProb += math.log10(c[i])
        logProb = -logProb

        if logProb > old_log_prob:
            return False, logProb
        return True, logProb

    # Return a "completed" sample by additing additional steps based on model probability.
    def complete_sequence(self, sample, steps):
        pass


def main():
    parser = argparse.ArgumentParser(description='Program to build and train a neural network.')
    parser.add_argument('--train_path', default=None,
                        help='Path to the training data directory.')
    parser.add_argument('--dev_path', default=None,
                        help='Path to the development data directory.')
    parser.add_argument('--max_iters', type=int, default=30,
                        help='The maximum number of EM iterations (default 30)')
    parser.add_argument('--hidden_states', type=int, default=5,
                        help='The number of hidden states to use. (default 10)')

    args = parser.parse_args()

    # 1. load training and testing data into memory
    # 2. build vocabulary using training data ONLY
    train_data, train_vocab = parse_data()

    # 3. instantiate an HMM with given number of states -- (1) initial parameters can
    #    be random or uniform for transitions and (2) inital state distributions,
    #    (3) initial emission parameters could be uniform OR based on vocabulary
    #    frequency (you'll have to count the words/characters as they occur in
    #    the training data.)
    hmm = HMM(args.hidden_states, len(train_vocab))

    # 4. output initial loglikelihood on training data and on testing data
    logProb = hmm.loglikelihood(train_data)

    # 5+. use EM to train the HMM on the training data,
    #     output loglikelihood on train and test after each iteration
    #     if it converges early, stop the loop and print a message
    converge = False

    for i in range(1, args.max_iters + 1):
        for dataset in train_data:
            converge, logProb = hmm.em_step(list(dataset), logProb)
            if converge is True:
                print("The EM algorithm converges after " + str(i) + " iteration(s).")
                break
        if converge is True:
            break

    # 6. Calculate the Accuracy
    ##############################

    print("Program Finishes.")


if __name__ == '__main__':
    main()
