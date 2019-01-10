from os import cpu_count

import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm


def apply_parallel_groups(df_groupby, f, n_jobs=cpu_count(), use_tqdm=False):
    """
    Apply function to each group in parallel, and return combined DataFrame.
    :param df_groupby: Pandas DataFrame GroupBy object
    :param f: function f(group_name:str, group_df:DataFrame) -> DataFrame
    :param n_jobs: run n jobs in parallel
    :param use_tqdm: show progress bar
    :return: concatenated DataFrame
    """

    df_partitions = [(name, group) for name, group in df_groupby]
    df_partitions = tqdm(df_partitions) if use_tqdm else df_partitions

    df_results = joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(f)(df[0], df[1]) for df in df_partitions)
    ret = pd.concat(df_results)
    return ret


class Viterbi:
    """
    Adaptation of Viterbi algorithm to find the most likely sequence of states, as in
    https://en.wikipedia.org/wiki/Viterbi_algorithm#Pseudocode
    T1 and T2 have a similar definition. i,j are swapped: j is the position index and
    i is the state at that position. The number of potential states with probability higher
    than zero varies at each position and the distribution is represented as a sparse matrix
    using a dictionary.
    """

    def __init__(self, A, sequence):
        """
        Initialize Viterbi data structures
        :param A: sparse transition matrix as dict where key is (prev_state,state) and value is transition probability
        :param sequence: sequence of states
        """

        self.sequence = sequence
        self.T = len(sequence)
        self.A = A
        self.T1 = None

    def states_at(self, j):
        """
        Return states whose probab at position j is higher than zero
        :param j: position, 0 <= j <= T-1
        :return: list of states
        """
        return list(self.sequence.iloc[j].keys())

    def max_state_and_prob_at(self, j):
        """
        Return state,prob pair with highest prob among states at position j
        :param j: position, 0 <= j <= T-1
        :return: pair (state,prob)
        """
        return max(self.sequence.iloc[j].items(), key=lambda kv: kv[1])

    def get_state_prob_at(self, j, i):
        """
        Return probab for state i at position j
        :param i: state. e.g., "Zone 101"
        :param j: position, 0 <= j <= T-1
        :return: probability value
        """
        return self.sequence.iloc[j].get(i, 0)

    def get_trans_prob(self, k, i, j):
        """
        Return transition probab from state $k$ to state $i$ from position $j-1$ to position $j$
        :param prev_i: previous state (source)
        :param i: state (destination)
        :return: probability value
        """

        # We are ignoring j. However, it can be used to extend the model to consider an additional
        # discretised temporal dimension.

        return self.A.get((k, i), 0)

    def lg(self, x):
        """
        Returns log of probability. Adding 1E-20 to avoid -inf (log(0)) in presence of zero probabilities,
        that might be returned by get_trans_prob, get_state_prob_at functions. In these cases, we can assume that
        the probability is never exactly zero. This allows us to preserve a ranking between different candidates,
        that otherwise would lead to degenerated log-probability estimates (-inf).
        :param x: probability value
        :return: log of probability
        """
        return np.log(x + 1E-20)

    def f(self, k, i, j):
        # evaluate logp to transition from state k at position j-1 to state i at position j
        # evaluate $T1[k,j-1]*A_{ki}*B_{ij}$
        return self.T1[j - 1][k] + self.lg(self.get_trans_prob(k, i, j)) + self.lg(self.get_state_prob_at(j, i))

    def eval_T1(self):
        """
        Estimate T1 matrix
        :return:
        """

        # initialize T1 matrix: rows are positions in the sequence ranging from 0...T
        # and sparse columns are states whose probability is higher than zero
        self.T1 = [None] * self.T

        # T1[j][i] stores probability of most likely sub-sequence sequence[0...j]
        # with state $i$ at its last position $j$. consequently, T1[T][i] stores the
        # probability that the most likely sequence ends with state $i$.

        # initialize first position
        self.T1[0] = {}
        for i in self.states_at(0):
            self.T1[0][i] = self.lg(self.get_state_prob_at(0, i))

        # now, for each other position in ascending order, initialize T1
        for j in range(1, self.T):
            self.T1[j] = {}
            # for j ranging from 1 to T-1
            for i in self.states_at(j):
                # for all states i at position j whose probability is higher than zero
                self.T1[j][i] = max(
                    [(k, self.f(k, i, j)) for k in self.states_at(j - 1)], key=lambda kv: kv[1])[1]
                # we assign the maximum probability for state $i$ at position $j$ considering all states $k$ at position $j-1$

    def T2(self, j, i):
        """
        Given state $i$ at position $j$, returns most likely state $k$ for position $j-1$

        :param j: state
        :param i: position
        :return: most likely state
        """

        # print("getting T2 of j=", j, "i=", i)
        return max([(k, self.f(k, i, j)) for k in self.states_at(j - 1)], key=lambda kv: kv[1])[0]

    def find_best_last_state_and_prob(self):
        """
        Return state $k$ at last position T-1 and its probability $p$, where $p$ highest at position T-1
        """
        return max([(k, self.T1[self.T - 1][k]) for k in self.states_at(self.T - 1)], key=lambda kv: kv[1])

    def get_best_sequence(self):
        """
        Return list containing most likely sequence of states
        """

        # evaluate T1 matrix
        self.eval_T1()

        # print("T1: ", self.T1)

        # X at position j stores best state at position j
        X = [None] * self.T

        # get best state at position T-1 (last in the sequence, whose range is 0...T-1)
        k, p = self.find_best_last_state_and_prob()
        # print("Best end state is {} with logp {}".format(k, p))

        X[self.T - 1] = k

        for j in reversed(range(1, self.T)):
            # print("estimating x[j-1], j=", j, "X[j]=", X[j])
            # range from j= T-1 to 1
            # T2 at position $j$ stores best state at position $j-1$ given that at position $j$ the state is $X[j]$
            X[j - 1] = self.T2(j, X[j])

        return X

    def get_clean_series(self):
        # print("Initial states = {}".format(self.states_at(0)))
        # print("Final states = {}".format(self.states_at(self.T - 1)))
        # print("Sequence length T = {}".format(self.T))
        return pd.Series(self.get_best_sequence())


class HMMFilter():
    """
    Find most likely sequence of states for a dataset of sequences
    """

    def __init__(self):
        """
        Initialise internal state
        """

        self.A = None

    def fit(self, dataset, session_column, prediction_column, n_jobs=cpu_count(), use_tqdm=False):
        """
        Estimate the transition matrix given a dataset with predicted classes using their normalised
        tranistion frequencies. Sets self.A to the estimated transition Matrix.

        :param dataset: DataFrame with columns [session_column, prediction_column]
        :param session_column: column name of dataset containing the session identifier
        :param prediction_column: column name of dataset containing the predicted class
        :param n_jobs: run n jobs in parallel
        :param use_tqdm: show progress bar
        :return: None
        """

        # make sure that columns exist
        assert prediction_column in dataset.columns and prediction_column in dataset.columns

        # extract list of transition pairs
        def extract_transitions_list(name, df):
            """
            Inner function that extracts sequences of pairs of consecutive predicted states
            :param name: name of the group from GroupBy, that corresponds to the value of session_column for this group
            :param df: rows matching the value of this groups' session ID
            :return: DataFrame containing the pairs of consecutive predicted states
            """
            df['prev_prediction'] = df[prediction_column].shift()
            df.rename(columns={'prev_prediction': 'src', prediction_column: 'dst'}, inplace=True)
            # first row has no prev! remove it
            df = df.iloc[1:]
            return df[['src', 'dst']]

        # extract all pairs of consecutive predicted states in parallel for each session ID
        transition = apply_parallel_groups(dataset.groupby(session_column), extract_transitions_list, n_jobs, use_tqdm)

        def extract_probabs(name, df):
            """
            Extract transition probabilities for a source state
            :param name: source state
            :param df: rows whose source state matches this groups' source state (value of name parameter)
            :return: DataFrame containings triplets source state, destination state, and probability estimate
            """
            df = df.groupby('dst').size().rename('count').reset_index()
            df['src'] = name
            df['probability'] = df['count'] / df['count'].sum()
            return df[['src', 'dst', 'probability']]

        # estimate transition probabilities counting matching pairs
        transition_probabs = apply_parallel_groups(transition.groupby(['src']), extract_probabs, n_jobs, use_tqdm)

        # assign transition matrix as sparse matrix
        self.A = transition_probabs.set_index(['src', 'dst'])["probability"].to_dict()

    def predict(self, dataset, session_column, probabs_column, prediction_column, k=20, n_jobs=cpu_count(),
                use_tqdm=False):
        """
        Predict state at each timestamp for all sessions in dataset, extracting most likely state considering
        state distributions and transition matrix.
        :param dataset: DataFrame of sessions. Rows are sorted by ascending timestamp.
        :param session_column: column name of dataset containing session name
        :param probabs_column: column name of dataset containing state probability distribution as dictionary in the form { state1: probability1, state2: probability2, ...} where all probabilities are higher than sero
        :param prediction_column: column name of dataset where prediction is saved
        :param k: pruning parameter, only top-k probabilities are retained for each state
        :param n_jobs: run n jobs in parallel
        :param use_tqdm: show progress bar
        :return: DataFrame dataset with mutated row order and new column prediction_column
        """

        # Limit each sample's class distribution to its top-k.
        # This to limit the number of possible worlds, resulting in a faster, approximated evaluation.
        # In most cases, this results in the very same output of the complete evaluation.
        def limit_topk(d, k):
            if len(d) <= k:
                return d
            l_topk = sorted(d.items(), key=lambda kv: kv[1], reverse=True)[:k]
            norm_factor = 1 / sum(map(lambda kv: kv[1], l_topk))
            return dict(list(map(lambda kv: (kv[0], kv[1] * norm_factor), l_topk)))

        dataset[probabs_column] = dataset[probabs_column].apply(lambda d: limit_topk(d, k))

        def predict_sequence(name, df):
            """
            Predict the most likely seuence for one session
            :param name: session ID
            :param df: rows matching session ID
            :return: input DataFrame with additional column, prediction_column, containing most likely state for each row
            """
            df.reset_index(drop=True, inplace=True)
            df[prediction_column] = Viterbi(self.A, df[probabs_column]).get_clean_series()
            return df

        # add column containing prediction according to the markov model prediction in parallel
        return apply_parallel_groups(dataset.groupby(session_column), predict_sequence, n_jobs, use_tqdm=use_tqdm)
