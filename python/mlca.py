#
# (c) 2022 Naoki Masuyama
#
# MLCA is proposed in:
# N. Masuyama, Y. Nojima, C. K. Loo, and H. Ishibuchi,
# "Multi-label classification via adaptive resonance theory-based clustering,"
# IEEE Transactions on Pattern Analysis and Machine Intelligence, 2022."
#
# Please contact "masuyama@omu.ac.jp" if you have any problems.
#

import networkx as nx
import numpy as np
import scipy.sparse as sparse
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.estimator_checks import check_estimator
from collections.abc import Iterable, Iterator


class MLCA(BaseEstimator, ClassifierMixin):
    """ Multi-Label CIM-based Adaptive Resonance Theory (MLCA)"""

    def __init__(self, plambda=50, v_thres=0.25, ref_node=10, s=1.0, G_=nx.Graph(), dim_=None, init_label_list_=None, num_signal_=0, init_count_label_=0, sigma_=None, labels_=None, likelihood_false_=None, likelihood_true_=None, ci_false_=None, ci_true_=None, prior_prob_false_=None, prior_prob_true_=None):
        """
        :param plambda:
            A period deleting nodes. The nodes that doesn't satisfy some condition are deleted every this period.
        :param v_thres:
            A similarity threshold for node learning
        """

        self.plambda = plambda  # An interval for adapting \sigma
        self.v_thres = v_thres  # A similarity threshold
        self.ref_node = ref_node  # Number of neighbours
        self.s = s  # A smoothing parameter

        self.G_ = G_  # network
        self.dim_ = dim_  # Number of variables in an instance
        self.init_label_list_ = init_label_list_  # A list for initial label counts
        self.num_signal_ = num_signal_  # Counter for training instances
        self.init_count_label_ = init_count_label_  # Number of labels of signal
        self.sigma_ = sigma_  # An estimated sigma for CIM
        self.labels_ = labels_  # Label information for a testing instance

        self.prior_prob_false_ = prior_prob_false_  # Prior probability
        self.prior_prob_true_ = prior_prob_true_
        self.ci_false_ = ci_false_  # Label count for prior probability
        self.ci_true_ = ci_true_
        self.likelihood_false_ = likelihood_false_  # Likelihood
        self.likelihood_true_ = likelihood_true_

    def fit(self, x: np.ndarray, y: np.ndarray):
        """
        train data in batch manner
        :param y: array-like or ndarray
        :param x: array-like or ndarray
        """

        # avoiding IndexError caused by a complex-valued label
        # Comment-in when check_estimator() is used
        # y = y.real.astype(int)

        self.initialization(x, y)

        for signal in x:
            self.input_signal(signal, x, y)  # training network

        return self

    def predict(self, x: np.ndarray):
        """
        predict label for each sample.
        :param x: array-like or ndarray
        :rtype list:
            label for each sample.
        """
        y_pred, _ = self.__labeling_sample_for_multi_label_classification(x)

        return y_pred

    def predict_proba(self, x: np.ndarray):
        """
        predict label probability for each sample.
        :param x: array-like or ndarray
        :rtype list:
            label probability for each sample.
        """
        _, y_prob = self.__labeling_sample_for_multi_label_classification(x)

        return y_prob

    def initialization(self, x: np.ndarray, y: np.ndarray):
        """Initialize parameters
        :param y: array-like or ndarray
        :param x: array-like or ndarray
        """
        # set graph
        if len(list(self.G_.nodes)) == 0:
            self.G_ = nx.Graph()

        # set dimension of x
        if self.dim_ is None:
            self.dim_ = x.shape[1]

        # set an initial label count for a node
        if self.init_count_label_ is None:
            self.init_count_label_ = sparse.lil_array(np.zeros(y.shape[1]))

        # set a label count for a prior probability
        if self.ci_false_ is None:
            self.ci_false_ = sparse.lil_matrix((y.shape[1], self.ref_node + 1), dtype='i8')
            self.ci_true_ = sparse.lil_matrix((y.shape[1], self.ref_node + 1), dtype='i8')

        # set likelihood
        if self.likelihood_false_ is None:
            self.likelihood_false_ = sparse.lil_matrix((y.shape[1], self.ref_node + 1), dtype='float')
            self.likelihood_true_ = sparse.lil_matrix((y.shape[1], self.ref_node + 1), dtype='float')

    def input_signal(self, signal: np.ndarray, x: np.ndarray, y: sparse):
        """
        Input a new signal one by one, which means training in online manner.
        fit() calls __init__() before training, which means resetting the state. So the function does batch training.
        :param signal: A new input signal
        :param y: array-like or ndarray
            label
        :param x: array-like or ndarray
            data
        """
        if self.num_signal_ == x.shape[0]:
            self.num_signal_ = 1
        else:
            self.num_signal_ += 1

        label = y[self.num_signal_ - 1, :]  # extract a label of signal

        if self.G_.number_of_nodes() < 1:
            self.__estimate_sigma(x)
            self.__add_node(signal, label)
        else:

            node_list, cim = self.__calculate_cim(signal)
            s1_idx, s1_cim, s2_idx, s2_cim = self.__find_nearest_node(node_list, cim)

            if self.v_thres < s1_cim:
                self.__estimate_sigma(x)
                self.__add_node(signal, label)

                new_node_list = node_list + [list(self.G_.nodes())[-1]]  # add a new node
                new_cim = np.append(cim, 0)  # add a cim value of a new node
                self.__update_likelihood(label, new_node_list, new_cim)
            else:
                self.__update_s1_node(s1_idx, signal, label)
                if self.v_thres >= s2_cim:
                    self.__update_s2_node(s2_idx, signal)

                self.__update_likelihood(label, node_list, cim)

        self.__compute_prior()

    def __estimate_sigma(self, X: np.ndarray) -> None:
        """Estimate a sigma for CIM

        Setup
        >>> MLCAnet = MLCA()
        >>> MLCAnet.G_ = nx.Graph()
        >>> MLCAnet.dim_ = 2
        >>> x = np.array([[1,2],[3,4],[5,6],[7,8],[9,10]])

        Depending on num_signal, pLamba, and the number of datapoints x,
        a value of an estimated sigma will be changed.

        if x.shape[0] < self.pLambda:
        >>> MLCAnet.num_signal_ = 4
        >>> MLCAnet.plambda = 10
        >>> MLCAnet._MLCA__estimate_sigma(x)
        >>> MLCAnet.sigma_
        2.4182711751219577

        elif (self.num_signal - self.pLambda) <= 0:
        >>> MLCAnet.num_signal_ = 2
        >>> MLCAnet.plambda = 4
        >>> MLCAnet._MLCA__estimate_sigma(x)
        >>> MLCAnet.sigma_
        2.0493259460083237

        elif (self.num_signal - self.pLambda) > 0:
        >>> MLCAnet.num_signal_ = 4
        >>> MLCAnet.plambda = 2
        >>> MLCAnet._MLCA__estimate_sigma(x)
        >>> MLCAnet.sigma_
        1.2599210498948732
        """
        # if self.num_signal == 1:
        #     selected_signals = X[0:2]
        # elif X.shape[0] < self.pLambda:
        if X.shape[0] < self.plambda:
            selected_signals = X
        elif (self.num_signal_ - self.plambda) <= 0:
            selected_signals = X[0:self.plambda]
        elif (self.num_signal_ - self.plambda) > 0:
            selected_signals = X[(self.num_signal_ - self.plambda):self.num_signal_]

        std_signals = np.std(selected_signals, axis=0, ddof=1)
        np.putmask(std_signals, std_signals == 0.0, 1.0e-6)  # If value=0, add a small value for avoiding an error.

        # Silverman's Rule
        s = np.power(4 / (2 + self.dim_), 1 / (4 + self.dim_)) * std_signals * np.power(selected_signals.shape[0], -1 / (4 + self.dim_))
        self.sigma_ = np.median(s)

    def __calculate_cim(self, signal: np.ndarray):
        """Calculate CIM between signal and nodes

        Setup
        >>> MLCAnet = MLCA()
        >>> MLCAnet.G_ = nx.Graph()
        >>> MLCAnet.G_.add_node(0, weight=[0.1, 0.5], winning_counts=1, sigma=1.0)
        >>> signal = np.array([0, 0])

        Return an index and a value of the cim between a node and a signal
        >>> MLCAnet._MLCA__calculate_cim(signal)
        ([0], array([0.2474779]))

        If there are multiple nodes, return multiple indexes and values of the cim
        >>> MLCAnet.G_.add_node(1, weight=[0.9, 0.6], winning_counts=1, sigma=1.0)
        >>> MLCAnet._MLCA__calculate_cim(np.array([0, 0]))
        ([0, 1], array([0.2474779 , 0.49887522]))
        """
        node_list = list(self.G_.nodes)
        weights = list(self.__get_node_attributes_from('weight', node_list))
        sigma = list(self.__get_node_attributes_from('sigma', node_list))

        c = np.exp(-(signal - np.array(weights)) ** 2 / (2 * np.mean(np.array(sigma)) ** 2))
        return node_list, np.sqrt(1 - np.mean(c, axis=1))

    def __add_node(self, signal: np.ndarray, label: sparse = None) -> None:
        """Add a new node to G with winning count and sigma

        Setup
        >>> MLCAnet = MLCA()
        >>> MLCAnet.sigma_ = 0.5
        >>> MLCAnet.G_ = nx.Graph()

        Add the 1st node to G
        >>> signal = np.array([1,2])
        >>> label = sparse.csr_array([0, 0, 1])
        >>> MLCAnet._MLCA__add_node(signal, label)
        >>> list(MLCAnet.G_.nodes.data())
        [(0, {'weight': array([1, 2]), 'winning_counts': 1, 'sigma': 0.5, 'count_label': <1x3 sparse array of type '<class 'numpy.int64'>'
        with 1 stored elements in Compressed Sparse Row format>})]

        Add the 2nd node to G
        >>> signal = np.array([3,4])
        >>> label = sparse.csr_array([1, 0, 1])
        >>> MLCAnet._MLCA__add_node(signal, label)
        >>> list(MLCAnet.G_.nodes.data())
        [(0, {'weight': array([1, 2]), 'winning_counts': 1, 'sigma': 0.5, 'count_label': <1x3 sparse array of type '<class 'numpy.int64'>'
    	    with 1 stored elements in Compressed Sparse Row format>}),
    	(1, {'weight': array([3, 4]), 'winning_counts': 1, 'sigma': 0.5, 'count_label': <1x3 sparse array of type '<class 'numpy.int64'>'
    	    with 2 stored elements in Compressed Sparse Row format>})]
        """
        if len(self.G_.nodes) == 0:  # for the first node
            new_node_idx = 0
        else:
            new_node_idx = max(self.G_.nodes) + 1

        # assign initial label
        self.G_.add_node(new_node_idx, weight=signal, winning_counts=1, sigma=self.sigma_, count_label=label)

    def __find_nearest_node(self, node_list: list, cim: np.ndarray):
        """Return indexes and weights of the 1st and 2nd nearest nodes from signal

        Setup
        >>> MLCAnet = MLCA()
        >>> MLCAnet.G_ = nx.Graph()
        >>> MLCAnet.G_.add_node(0, weight=[0.1, 0.5], winning_counts=1, sigma=1.0)

        If there is only one node, return an index and the cim value of the 1st nearest node.
        In this case, for the 2nd nearest node, an index is the same as the 1st nearest node and its value is inf.
        >>> node_list = [0]
        >>> cim = np.array([0.5])
        >>> MLCAnet._MLCA__find_nearest_node(node_list, cim)
        (0, 0.5, 0, inf)

        If there are two nodes, return an index and the cim value of the 1st and 2nd nearest nodes.
        >>> MLCAnet.G_.add_node(1, weight=[0.9, 0.6], winning_counts=1, sigma=1.0)
        >>> node_list = [0, 1]
        >>> cim = np.array([0.5, 0.9])
        >>> MLCAnet._MLCA__find_nearest_node(np.array([0, 1]), cim)
        (0, 0.5, 1, 0.9)
        """

        if len(node_list) == 1:
            node_list = node_list + node_list
            cim = np.array(list(cim) + [np.inf])

        idx = np.argsort(cim)
        return node_list[idx[0]], cim[idx[0]], node_list[idx[1]], cim[idx[1]]

    def __update_s1_node(self, idx: int, signal: np.ndarray, label: sparse):
        """Update weight for s1 node

        Setup
        >>> MLCAnet = MLCA()
        >>> MLCAnet.G_ = nx.Graph()
        >>> MLCAnet.sigma_ = 1.0
        >>> label = sparse.csr_array([0, 0, 1])
        >>> MLCAnet.G_.add_node(0, weight=[0.1, 0.5], winning_counts=1, sigma=MLCAnet.sigma_, count_label=label)
        >>> MLCAnet.G_.nodes[0]
        {'weight': [0.1, 0.5], 'winning_counts': 1, 'sigma': 1.0, 'count_label': <1x3 sparse array of type '<class 'numpy.int64'>'
    	    with 1 stored elements in Compressed Sparse Row format>}
    	>>> MLCAnet.G_.nodes[0]['count_label'].toarray()
    	array([[0, 0, 1]])

        Update weight, winning_counts, and count_label of s1 node
        >>> s1_idx = 0
        >>> signal = np.array([0,0])
        >>> label = sparse.csr_array([1, 0, 0])
        >>> MLCAnet._MLCA__update_s1_node(s1_idx, signal, label)
        >>> MLCAnet.G_.nodes[s1_idx]
        {'weight': array([0.05, 0.25]), 'winning_counts': 2, 'sigma': 1.0, 'count_label': <1x3 sparse array of type '<class 'numpy.int64'>'
            with 2 stored elements in Compressed Sparse Row format>}
    	>>> MLCAnet.G_.nodes[0]['count_label'].toarray()
        array([[1, 0, 1]])
        """
        # weight and winning count
        weight = self.G_.nodes[idx].get('weight')
        new_winning_count = self.G_.nodes[idx].get('winning_counts') + 1
        new_weight = weight + (signal - weight) / new_winning_count

        # label count
        count_label = self.G_.nodes[idx].get('count_label')
        net_count_label = count_label + label

        # update attributes
        nx.set_node_attributes(self.G_, {idx: {'weight': new_weight, 'winning_counts': new_winning_count, 'count_label': net_count_label}})

    def __update_s2_node(self, idx, signal):
        """Update weight for s2 node

        Setup
        >>> MLCAnet = MLCA()
        >>> MLCAnet.G_ = nx.Graph()
        >>> MLCAnet.sigma_ = 1.0
        >>> label = sparse.csr_array([0, 0, 1])
        >>> MLCAnet.G_.add_node(0, weight=[0.1, 0.5], winning_counts=1, sigma=MLCAnet.sigma_, count_label=label)
        >>> MLCAnet.G_.nodes[0]
        {'weight': [0.1, 0.5], 'winning_counts': 1, 'sigma': 1.0, 'count_label': <1x3 sparse array of type '<class 'numpy.int64'>'
    	    with 1 stored elements in Compressed Sparse Row format>}

        Update weight of s2 node
        Because a learning coefficient is different from __update_s1_node(), a value of weight is different.
        In addition, winning_counts of s2 node is not updated.
        >>> s2_idx = 0
        >>> signal = np.array([0,0])
        >>> MLCAnet._MLCA__update_s2_node(s2_idx, signal)
        >>> MLCAnet.G_.nodes[s2_idx]
        {'weight': array([0.09, 0.45]), 'winning_counts': 1, 'sigma': 1.0, 'count_label': <1x3 sparse array of type '<class 'numpy.int64'>'
    	    with 1 stored elements in Compressed Sparse Row format>}
        """
        weight = self.G_.nodes[idx].get('weight')
        winning_counts = self.G_.nodes[idx].get('winning_counts')
        new_weight = weight + (signal - weight) / (10 * winning_counts)
        nx.set_node_attributes(self.G_, {idx: {'weight': new_weight}})

    def __get_node_attributes_from(self, attr: str, node_list: Iterable[int]) -> Iterator:
        """Get an attribute of nodes in G

        Setup
        >>> MLCAnet = MLCA()
        >>> MLCAnet.G_ = nx.Graph()
        >>> MLCAnet.G_.add_node(0, weight=[0.1, 0.5], winning_counts=1, sigma=1.0)  # node 0
        >>> MLCAnet.G_.add_node(1, weight=[0.9, 0.6], winning_counts=2, sigma=2.0)  # node 1
        >>> MLCAnet.G_.add_node(2, weight=[1.0, 0.9], winning_counts=3, sigma=3.0)  # node 2
        >>> node_list = list(MLCAnet.G_.nodes)
        >>> node_list
        [0, 1, 2]

        Get weight of node.
        >>> list(MLCAnet._MLCA__get_node_attributes_from('weight', node_list))
        [[0.1, 0.5], [0.9, 0.6], [1.0, 0.9]]

        Get winning_counts of node.
        >>> list(MLCAnet._MLCA__get_node_attributes_from('winning_counts', node_list))
        [1, 2, 3]

        Get sigma of node.
        >>> list(MLCAnet._MLCA__get_node_attributes_from('sigma', node_list))
        [1.0, 2.0, 3.0]
        """
        att_dict = nx.get_node_attributes(self.G_, attr)
        return map(att_dict.get, node_list)

    def __compute_prior(self) -> None:
        """Helper function to compute for the prior probabilities of each label

        Setup
        >>> MLCAnet = MLCA()
        >>> MLCAnet.s = 1.0
        >>> MLCAnet.num_signal_ = 3
        >>> MLCAnet.G_ = nx.Graph()
        >>> MLCAnet.G_.add_node(0, weight=[0.1, 0.5], winning_counts=1, sigma=1.0, count_label=[0, 0, 1])  # node 0
        >>> MLCAnet.G_.add_node(1, weight=[0.9, 0.6], winning_counts=2, sigma=2.0, count_label=[0, 1, 1])  # node 1
        >>> MLCAnet.G_.add_node(2, weight=[1.0, 0.9], winning_counts=3, sigma=3.0, count_label=[0, 0, 1])  # node 2

        Get prior probabilities
        >>> MLCAnet._MLCA__compute_prior()
        >>> MLCAnet.prior_prob_true
        array([0.2, 0.4, 0.8])
        >>> MLCAnet.prior_prob_false
        array([0.8, 0.6, 0.2])

        Sum of prior_prob_true and prior_prob_false must be shown 1.0
        >>> MLCAnet.prior_prob_true + MLCAnet.prior_prob_false
        array([1., 1., 1.])
        """

        count_label = nx.get_node_attributes(self.G_, 'count_label')
        sum_labels = np.sum(list(count_label.values()), 0)

        self.prior_prob_true = np.array((self.s + sum_labels) / (self.s * 2 + self.num_signal_))
        self.prior_prob_false = 1 - self.prior_prob_true

    def __update_likelihood(self, label: sparse, node_list: list, cim: np.ndarray) -> None:
        """Helper function to update for the likelihood of each label in nodes

        Setup
        >>> MLCAnet = MLCA()
        >>> MLCAnet.G_ = nx.Graph()
        >>> MLCAnet.ref_node = 10
        >>> init_label = [0, 0, 0, 0, 0]
        >>> MLCAnet.ci_false_ = sparse.lil_matrix((len(init_label), MLCAnet.ref_node + 1), dtype='i8')
        >>> MLCAnet.ci_true_ = sparse.lil_matrix((len(init_label), MLCAnet.ref_node + 1), dtype='i8')
        >>> MLCAnet.likelihood_false_ = sparse.lil_matrix((len(init_label), MLCAnet.ref_node + 1), dtype='float')
        >>> MLCAnet.likelihood_true_ = sparse.lil_matrix((len(init_label), MLCAnet.ref_node + 1), dtype='float')

        If a new node is created,
        >>> signal = [0.1, 0.5]
        >>> label = [0, 0, 1, 0, 1]
        >>> MLCAnet.G_.add_node(0, weight=signal, winning_counts=1, sigma=1.0, count_label=label)  # node 0
        >>> node_list, cim = MLCAnet._MLCA__calculate_cim(signal)
        >>> new_node_list = node_list + [list(MLCAnet.G_.nodes())[-1]]  # add a new node
        >>> new_cim = np.append(cim, 0)  # add a cim value of a new node
        >>> MLCAnet._MLCA__update_likelihood(label, new_node_list, new_cim)
        >>> MLCAnet.ci_true_.toarray()
        array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])
        >>> MLCAnet.likelihood_true_.toarray()
        array([[0.5       , 0.5       , 0.        , 0.        , 0.        ,
                0.        , 0.        , 0.        , 0.        , 0.        ,
                0.        ],
               [0.5       , 0.5       , 0.        , 0.        , 0.        ,
                0.        , 0.        , 0.        , 0.        , 0.        ,
                0.        ],
               [0.33333333, 0.33333333, 0.        , 0.        , 0.        ,
                0.        , 0.        , 0.        , 0.        , 0.        ,
                0.        ],
               [0.5       , 0.5       , 0.        , 0.        , 0.        ,
                0.        , 0.        , 0.        , 0.        , 0.        ,
                0.        ],
               [0.33333333, 0.33333333, 0.        , 0.        , 0.        ,
                0.        , 0.        , 0.        , 0.        , 0.        ,
                0.        ]])

        If an existing node is updated,
        >>> signal = [0.2, 0.3]
        >>> label = np.array([1, 0, 0, 0, 1])
        >>> node_list, cim = MLCAnet._MLCA__calculate_cim(signal)
        >>> MLCAnet._MLCA__update_likelihood(label, node_list, cim)
        >>> MLCAnet.ci_true_.toarray()
        array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2]])
        >>> MLCAnet.likelihood_true_.toarray()
        array([[0.33333333, 0.33333333, 0.        , 0.        , 0.        ,
                0.        , 0.        , 0.        , 0.        , 0.        ,
                0.        ],
               [0.5       , 0.5       , 0.        , 0.        , 0.        ,
                0.        , 0.        , 0.        , 0.        , 0.        ,
                0.        ],
               [0.33333333, 0.33333333, 0.        , 0.        , 0.        ,
                0.        , 0.        , 0.        , 0.        , 0.        ,
                0.        ],
               [0.5       , 0.5       , 0.        , 0.        , 0.        ,
                0.        , 0.        , 0.        , 0.        , 0.        ,
                0.        ],
               [0.25      , 0.25      , 0.        , 0.        , 0.        ,
                0.        , 0.        , 0.        , 0.        , 0.        ,
                0.        ]])

        """

        # set the number of neighbors
        neighbors = self.ref_node
        if self.G_.number_of_nodes() < neighbors:
            neighbors = self.G_.number_of_nodes()

        # get indexes of neighbors
        idx = np.argsort(cim)
        idx_neighbors = idx[0:neighbors]

        # accumulate count_label of neighbors
        count_label = nx.get_node_attributes(self.G_, "count_label")
        neighbors_count_label = [count_label.get(idx_neighbors[n]) for n in range(idx_neighbors.shape[0])]
        adjusted_neighbors_count_label = np.sum(neighbors_count_label, 0) / np.max(np.sum(neighbors_count_label, 0)) * neighbors
        deltas = self.my_round(adjusted_neighbors_count_label)

        if neighbors > 1:
            indexes = [deltas[k] for k in range(len(label))]
        else:
            indexes = np.ones(len(label)) * self.ref_node

        for k, idx in enumerate(indexes):
            if label[k] == 1:
                self.ci_true_[k, int(idx)] += 1
            else:
                self.ci_false_[k, int(idx)] += 1

        # compute likelihood
        ci_true_sum = self.ci_true_.sum(axis=1)
        ci_false_sum = self.ci_false_.sum(axis=1)
        for k in range(len(label)):
            for n in range(neighbors + 1):
                self.likelihood_true_[k, n] = (self.s + self.ci_true_[k, n]) / (self.s * (neighbors + 1) + ci_true_sum[k, 0])
                self.likelihood_false_[k, n] = (self.s + self.ci_false_[k, n]) / (self.s * (neighbors + 1) + ci_false_sum[k, 0])

    def __labeling_sample_for_multi_label_classification(self, x: np.ndarray):
        """A label of testing sample is determined by using frequency of label_counts in each cluster.
        Labeled samples should be evaluated by using classification metrics.

        """

        # set the number of neighbors
        neighbors = self.ref_node
        if self.G_.number_of_nodes() < neighbors:
            neighbors = self.G_.number_of_nodes()

        # compute cim between x and nodes
        node_list = list(self.G_.nodes)
        weights = list(self.__get_node_attributes_from('weight', node_list))
        sigma = list(self.__get_node_attributes_from('sigma', node_list))
        c = [np.exp(-(x[k, :] - np.array(weights)) ** 2 / (2 * np.mean(np.array(sigma)) ** 2)) for k in range(len(x))]
        cim = [np.sqrt(1 - np.mean(c[k], axis=1)) for k in range(len(x))]

        # get indexes of neighbors
        idx = np.argsort(cim)
        idx_neighbors = idx[:, 0:neighbors]

        # accumulate labels of neighbors
        count_label = nx.get_node_attributes(self.G_, "count_label")
        neighbors_count_label = [[count_label.get(idx_neighbors[k, n]) for n in range(idx_neighbors.shape[1])] for k in range(len(x))]
        sum_neighbors_count_label = [np.sum(neighbors_count_label[k], 0) for k in range(len(x))]
        adjusted_neighbors_count_label = [(sum_neighbors_count_label[k] / np.max(sum_neighbors_count_label[k])) * neighbors for k in range(len(x))]
        deltas = [self.my_round(adjusted_neighbors_count_label[k]) for k in range(len(x))]

        # compute label probability
        p_true = [[self.likelihood_true_[m, int(deltas[k][m])] * self.prior_prob_true[m] for m in range(self.likelihood_true_.shape[0])] for k in range(len(x))]
        p_false = [[self.likelihood_false_[m, int(deltas[k][m])] * self.prior_prob_false[m] for m in range(self.likelihood_false_.shape[0])] for k in range(len(x))]

        y_prob = [[pt / (pt + pf) for pt, pf in zip(p_true[k], p_false[k])] for k in range(len(x))]  # predicted labels
        y_pred = [[int(y_prob[k][m] >= 0.5) for m in range(self.likelihood_true_.shape[0])] for k in range(len(x))]  # probability of labels

        return y_pred, y_prob

    @staticmethod
    def my_round(num: np.ndarray, digit: int = 0):
        """Rounding 0.5 to 1.0

        Parameters
        ----------
        num : np. arrays
        digit :

        Returns
        -------
        r : np. arrays
        """
        p = 10 ** digit
        r = (num * p * 2 + 1) // 2 / p
        return r

if __name__ == '__main__':
    # https://docs.python.org/3.10/library/doctest.html
    import doctest

    doctest.testmod()

    check_estimator(MLCA())
