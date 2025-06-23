"""
University of Bari Aldo Moro

@author: Ivan Diliso, Nicola Fanizzi
"""

import numpy as np
from owlready2 import *
from scipy.sparse import isspmatrix
from scipy.special import gammaln, logsumexp, psi
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.linear_model import LogisticRegression
from sklearn.utils import check_array
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.utils.validation import check_is_fitted, check_X_y

TRS = 0.5

# ============================= Helpers =============================================#


class StudentMultivariate(object):
    """
    Multivariate Student Distribution
    """

    def __init__(self, mean, precision, df, d):
        self.mu = mean
        self.L = precision
        self.df = df
        self.d = d

    def logpdf(self, x):
        """
        Calculates value of logpdf at point x
        """
        xdiff = x - self.mu
        quad_form = np.sum(np.dot(xdiff, self.L) * xdiff, axis=1)

        return (
            gammaln(0.5 * (self.df + self.d))
            - gammaln(0.5 * self.df)
            + 0.5 * np.linalg.slogdet(self.L)[1]
            - 0.5 * self.d * np.log(self.df * np.pi)
            - 0.5 * (self.df + self.d) * np.log(1 + quad_form / self.df)
        )

    def pdf(self, x):
        """
        Calculates value of pdf at point x
        """
        return np.exp(self.logpdf(x))


def _e_log_dirichlet(alpha0, alphaK):
    """Calculates expectation of log pdf of dirichlet distributed parameter"""
    log_C = gammaln(np.sum(alpha0)) - np.sum(gammaln(alpha0))
    e_log_x = np.dot(alpha0 - 1, psi(alphaK) - psi(np.sum(alphaK)))
    return np.sum(log_C + e_log_x)


def _e_log_beta(c0, d0, c, d):
    """Calculates expectation of log pdf of beta distributed parameter"""
    log_C = gammaln(c0 + d0) - gammaln(c0) - gammaln(d0)
    psi_cd = psi(c + d)
    log_mu = (c0 - 1) * (psi(c) - psi_cd)
    log_i_mu = (d0 - 1) * (psi(d) - psi_cd)
    return np.sum(log_C + log_mu + log_i_mu)


def _get_classes(X):
    """Finds number of unique elements in matrix"""
    if isspmatrix(X):
        v = X.data
        if len(v) < X.shape[0] * X.shape[1]:
            v = np.hstack((v, np.zeros(1)))
            V = np.unique(v)
    else:
        V = np.unique(X)
    return V


# ==================================================================================#


class GeneralMixtureModelExponential(BaseEstimator):
    """
    Superclass for Mixture Models
    """

    def __init__(
        self,
        n_components=2,
        n_iter=100,
        tol=1e-3,
        alpha0=10,
        n_init=3,
        init_params=None,
        compute_score=False,
        verbose=False,
    ):
        self.n_iter = n_iter
        self.n_init = n_init
        self.n_components = n_components
        self.tol = tol
        self.alpha0 = alpha0
        self.compute_score = compute_score
        self.init_params = init_params
        self.verbose = verbose

    def _update_resps(self, X, alphaK, *args):
        """
        Updates distribution of latent variable with Dirichlet prior
        """
        e_log_weights = psi(alphaK) - psi(np.sum(alphaK))
        return self._update_resps_parametric(X, e_log_weights, self.n_components, *args)

    def _update_resps_parametric(self, X, log_weights, clusters, *args):
        """Updates distribution of latent variable with parametric weights"""
        """
        HERE K IS THE SPECIFIC CLUSTER (SELECTED COMPONENT)
        """
        log_resps = np.asarray(
            [
                self._update_logresp_cluster(X, k, log_weights, *args)
                for k in range(clusters)
            ]
        ).T
        log_like = np.copy(log_resps)
        log_resps -= logsumexp(log_resps, axis=1, keepdims=True)
        resps = np.exp(log_resps)
        delta_log_like = np.sum(resps * log_like) - np.sum(resps * log_resps)
        return resps, delta_log_like

    def _update_dirichlet_prior(self, alpha_init, Nk):
        """
        For all models defined in this module prior for cluster distribution
        is Dirichlet, so all models will need to update parameters
        """
        return alpha_init + Nk

    def _check_X(self, X):
        """
        Checks validity of input for all mixture models
        """
        X = check_array(X, accept_sparse=["csr"])
        # check that number of components is smaller or equal to number of samples
        if X.shape[0] < self.n_components:
            raise ValueError(
                ("Number of components should not be larger than " "number of samples")
            )

        return X

    def _check_convergence(self, metric_diff, n_params):
        """Checks convergence of mixture model"""
        convergence = metric_diff / n_params < self.tol
        if self.verbose and convergence:
            print("Algorithm converged")
        return convergence

    def predict(self, X):
        """
        Predict cluster for test data

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
           Data Matrix

        Returns
        -------
        : array, shape = (n_samples,) component memberships
           Cluster index
        """
        return np.argmax(self.predict_proba(X), 1)

    def score(self, X):
        """
        Computes the log probability under the model

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            List of n_features-dimensional data points.  Each row
            corresponds to a single data point

        Returns
        -------
        logprob: array with shape [n_samples,]
            Log probabilities of each data point in X
        """
        probs = self.predict_proba(X)
        return np.log(np.dot(probs, self.weights_))


# Bernoulli Mixture With Variational Bayes
################################################################################


class BernoulliMixtureVB(GeneralMixtureModelExponential):
    """
    Variational Bayesian Bernoulli Mixture Model

    Parameters
    ----------
    n_components : int, optional (DEFAULT = 2)
        Number of mixture components

    n_init :  int, optional (DEFAULT = 5)
        Number of restarts of algorithm

    n_iter : int, optional (DEFAULT = 100)
        Number of iterations of Mean Field Approximation Algorithm

    tol : float, optional (DEFAULT = 1e-3)
        Convergence threshold

    alpha0 :float, optional (DEFAULT = 1)
        Concentration parameter for Dirichlet prior on weights

    c : float , optional (DEFAULT = 1)
        Shape parameter for beta distribution

    d: float , optional (DEFAULT = 1)
        Shape parameter for beta distribution

    compute_score: bool, optional (DEFAULT = True)
        If True computes logarithm of lower bound at each iteration

    verbose : bool, optional (DEFAULT = False)
        Enable verbose output


    Attributes
    ----------
    weights_ : numpy array of size (n_components,)
        Mixing probabilities for each cluster

    means_ : numpy array of size (n_features, n_components)
        Mean success probabilities for each cluster

    scores_: list of unknown size (depends on number of iterations)
        Log of lower bound

    """

    def __init__(
        self,
        n_components=2,
        n_init=10,
        n_iter=100,
        tol=1e-3,
        alpha0=1,
        c=1e-2,
        d=1e-2,
        init_params=None,
        compute_score=False,
        verbose=False,
    ):
        super(BernoulliMixtureVB, self).__init__(
            n_components,
            n_iter,
            tol,
            alpha0,
            n_init,
            init_params,
            compute_score,
            verbose,
        )
        self.c = c
        self.d = d

    def _check_X_train(self, X):
        """Preprocesses & check validity of training data"""
        X = super(BernoulliMixtureVB, self)._check_X(X)
        self.classes_ = _get_classes(X)
        n = len(self.classes_)
        # check that there are only two categories in data
        if n != 2:
            raise ValueError(
                (
                    "There are {0} categorical values in data, "
                    "model accepts data with only 2".format(n)
                )
            )
        return 1 * (X == self.classes_[1])

    def _check_X_test(self, X):
        """Preprocesses & check validity of test data"""

        classes_ = [0, 1]
        X = check_array(X, accept_sparse=["csr"])
        classes_ = _get_classes(X)
        n = len(classes_)
        # check number of classes
        if n != 2:
            raise ValueError(
                (
                    "There are {0} categorical values in data, "
                    "model accepts data with only 2".format(n)
                )
            )
            # check whether these are the same classes as in training
        if classes_[0] == self.classes_[0] and classes_[1] == self.classes_[1]:
            return 1 * (X == self.classes_[1])
        else:
            raise ValueError(
                (
                    "Classes in training and test set are different, "
                    "{0} in training, {1} in test".format(self.classes_, classes_)
                )
            )

    def _fit(self, X):
        """
        Performs single run of VBBMM
        """
        _, n_features = X.shape

        # Fattori di mixing e probabilità per ogni feature e per ogni component
        n_params = n_features * self.n_components + self.n_components
        scores = []

        # These are the parameters of the beta distribution for each feature and for each cluster
        # aka components
        # use initial values of hyperparameter as starting point
        c = self.c * np.random.random([n_features, self.n_components])
        d = self.d * np.random.random([n_features, self.n_components])

        c_old, d_old = c, d
        c_prev, d_prev = c, d

        # These are the parameters of the dirichlet distribution
        # we need to break symmetry for mixture weights
        alphaK = self.alpha0 * np.random.random(self.n_components)
        alphaK_old = alphaK
        alphaK_prev = alphaK

        for i in range(self.n_iter):

            # ---- update approximating distribution of latent variable ----- #

            resps, delta_log_like = self._update_resps(X, alphaK, c, d)

            # reuse responsibilities in computing lower bound
            if self.compute_score:
                scores.append(
                    self._compute_score(
                        delta_log_like, alphaK_old, alphaK, c_old, d_old, c, d
                    )
                )

            # ---- update approximating distribution of parameters ---------- #

            Nk = sum(resps, 0)

            # update parameters of Dirichlet Prior
            alphaK = self._update_dirichlet_prior(alphaK_old, Nk)

            # update parameters of Beta distributed success probabilities
            c, d = self._update_params(X, Nk, resps)
            diff = np.sum(abs(c - c_prev) + abs(d - d_prev) + abs(alphaK - alphaK_prev))

            if self.verbose:
                if self.compute_score:
                    print(
                        "Iteration {0}, value of lower bound is {1}".format(
                            i, scores[-1]
                        )
                    )
                else:
                    print(
                        (
                            "Iteration {0}, normalised delta of parameters " "is {1}"
                        ).format(i, diff)
                    )

            if self._check_convergence(diff, n_params):
                break
            c_prev, d_prev = c, d
            alphaK_prev = alphaK

        # compute log of lower bound to compare best model
        resps, delta_log_like = self._update_resps(X, alphaK, c, d)
        scores.append(
            self._compute_score(delta_log_like, alphaK_old, alphaK, c_old, d_old, c, d)
        )
        return alphaK, c, d, scores

    def get_probs(self):
        return self.weights_, self.means_

    # SCORE COMPUTATION FOR ELBO

    def _compute_score(self, delta_log_like, alpha_init, alphaK, c_old, d_old, c, d):
        """
        Computes lower bound
        """
        log_weights_prior = _e_log_dirichlet(alpha_init, alphaK)
        log_success_prior = _e_log_beta(c_old, d_old, c, d)
        log_weights_approx = -_e_log_dirichlet(alphaK, alphaK)
        log_success_approx = -_e_log_beta(c, d, c, d)
        lower_bound = log_weights_prior
        lower_bound += log_success_prior + log_weights_approx
        lower_bound += log_success_approx + delta_log_like
        return lower_bound

    # E STEP --> START

    def _update_resps(self, X, alphaK, *args):
        """
        Updates distribution of latent variable with Dirichlet prior
        """
        e_log_weights = psi(alphaK) - psi(np.sum(alphaK))
        return self._update_resps_parametric(X, e_log_weights, self.n_components, *args)

    def _update_resps_parametric(self, X, log_weights, clusters, *args):
        """Updates distribution of latent variable with parametric weights"""
        """
        HERE K IS THE SPECIFIC CLUSTER (SELECTED COMPONENT)
        """
        log_resps = np.asarray(
            [
                self._update_logresp_cluster(X, k, log_weights, *args)
                for k in range(clusters)
            ]
        ).T
        log_like = np.copy(log_resps)

        # NORMALIZZO SU SCALA LOGARITMICA
        log_resps -= logsumexp(log_resps, axis=1, keepdims=True)
        # TORNO A EXPONENTE (ABBIAMO PROBABILITA ORA)
        resps = np.exp(log_resps)

        # UTILE ALLA ELBO
        delta_log_like = np.sum(resps * log_like) - np.sum(resps * log_resps)
        return resps, delta_log_like

    def _update_logresp_cluster(self, X, k, e_log_weights, *args):
        """
        Unnormalised responsibilities for single cluster
        """
        c, d = args
        ck, dk = c[:, k], d[:, k]

        # Psi CK - Psi DK va a calcolare le expected log odds,
        # Successivamente moltiplico per la matrice dei soggetti
        # Credo un vettore di shape numero elementi training
        # Aka la expected log odds che date tutte le risposte si, che appartenga
        # al cluster K
        xcd = safe_sparse_dot(X, (psi(ck) - psi(dk)))

        # Prima ho tenuto conto della risposta 1, ora tengo conto del valore 0
        # prende la tendenza generale del cluster K alla risposta 0
        # infinte sommo il match score "average log commoness"
        log_resp = xcd + np.sum(psi(dk) - psi(ck + dk)) + e_log_weights[k]
        return log_resp

    # E STEP --> END

    def _update_params(self, X, Nk, resps):
        """
        Update parameters of prior distribution for Bernoulli Succes Probabilities
        """
        XR = safe_sparse_dot(X.T, resps)
        c = self.c + XR
        d = self.d + (Nk - XR)
        return c, d

    def fit(self, X):
        """
        Fits Variational Bayesian Bernoulli Mixture Model

        Parameters
        ----------
        X: array-like or sparse csr_matrix of size [n_samples, n_features]
           Data Matrix

        Returns
        -------
        self: object
           self

        Practical Advice
        ----------------
        Significant speedup can be achieved by using sparse matrices
        (see scipy.sparse.csr_matrix)

        """

        # preprocess data
        X = self._check_X_train(X)

        # refit & choose best model (log of lower bound is used)
        score_old = [-np.inf]
        alpha_, c_, d_ = 0, 0, 0

        for j in range(self.n_init):
            if self.verbose:
                print("New Initialisation, restart number {0} \n".format(j))

            alphaK, c, d, score = self._fit(X)
            if score[-1] > score_old[-1]:
                alpha_, c_, d_ = alphaK, c, d
                score_old = score

        # save parameters corresponding to best model
        self.alpha_ = alpha_
        self.means_ = c_ / (c_ + d_)
        self.c_, self.d_ = c_, d_
        self.weights_ = alpha_ / np.sum(alpha_)
        self.scores_ = score_old
        return self

    def predict_proba(self, X):
        """
        Predict probability of cluster for test data

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Data Matrix for test data

        Returns
        -------
        probs : array, shape = (n_samples,n_components)
            Probabilities of components membership
        """
        check_is_fitted(self, "scores_")
        X = self._check_X_test(X)
        probs = self._update_resps(X, self.alpha_, self.c_, self.d_)[0]
        return probs

    def cluster_prototype(self, threshold):
        """
        Computes most likely prototype for each cluster, i.e. vector that has
        highest probability of being observed under learned distribution
        parameters.

        Returns
        -------
        protoypes: numpy array of size n_COMPONTENS X N_FEATURES
           Cluster prototype
        """

        print(self.means_.round(2))

        prototypes = np.asarray(
            [
                self.classes_[1 * (self.means_[:, i] >= threshold)]
                for i in range(self.n_components)
            ]
        )
        return prototypes

    def transform(self, X):

        pred = (self.predict_proba(X) > TRS).astype(int)

        return pred

    def fit_transform(self, X, y):  # need this y to implement pipeline
        self.fit(X)

        return self.transform(X)


# BERNOULLI MIXTURE MODEL WITH EM
################################################################################


class BernoulliMixtureEM(BaseEstimator, ClusterMixin):
    def __init__(
        self,
        n_components: int = 2,
        max_iter: int = 100,
        tol: float = 1e-4,
        random_state: int = None,
    ):
        """Initialize the BernoulliMixure

        Args:
            n_components (int, optional): Number of clusters (factors) in the mixture. Defaults to 2.
            max_iter (int, optional): Maximum number of iterations. Defaults to 100.
            tol (float, optional): Tolerance in score in iterations. Defaults to 1e-4.
            random_state (int, optional): Random seed. Defaults to None.
        """

        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def _initialize_parameters(self, X: np.array):
        """Initialize mixing factors and feature probabilities

        Args:
            X (np.array): Input data
        """

        n_samples, n_features = X.shape
        rng = np.random.RandomState(self.random_state)

        # Initialize mixing coefficients uniformly
        self.pi_ = np.ones(self.n_components) / self.n_components

        # Initialize Bernoulli parameters randomly in (0.25, 0.75) to avoid extremes
        self.theta_ = rng.uniform(0.25, 0.75, size=(self.n_components, n_features))

    def get_probs(self):
        """Get the model mixing factors and feature probabilities.
        """
        return self.pi_, self.theta_.T

    def _e_step(self, X):

        # Compute log probability for each component and sample

        n_samples, n_features = X.shape

        log_prob = np.zeros((n_samples, self.n_components))

        for k in range(self.n_components):

            # Bernoulli log likelihood for each feature

            log_bernoulli = X * np.log(self.theta_[k] + 1e-10) + (1 - X) * np.log(
                1 - self.theta_[k] + 1e-10
            )
            log_prob[:, k] = np.sum(log_bernoulli, axis=1) + np.log(self.pi_[k] + 1e-10)

        # Normalize to get responsibilities

        log_prob_norm = log_prob - log_prob.max(
            axis=1, keepdims=True
        )  # for numerical stability

        prob = np.exp(log_prob_norm)

        prob_sum = prob.sum(axis=1, keepdims=True)

        responsibilities = prob / prob_sum

        return responsibilities

    def _m_step(self, X, responsibilities):

        # Update mixing coefficients

        Nk = responsibilities.sum(axis=0)

        self.pi_ = Nk / X.shape[0]

        # Update Bernoulli parameters

        self.theta_ = (responsibilities.T @ X) / Nk[:, np.newaxis]

        # Clip to avoid exactly 0 or 1 probabilities
        self.theta_ = np.clip(self.theta_, 1e-6, 1 - 1e-6)

    def cluster_prototype(self, threshold):
        """
        Returns the binary prototype (most likely vector) for each cluster.

        Output shape: (n_features, n_components), matching VB version.
        """

        return (self.theta_ >= threshold).astype(int)

    def fit(self, X, y=None):

        X = self._check_X_train(X)

        self._initialize_parameters(X)

        lower_bound = -np.inf

        for i in range(self.max_iter):

            responsibilities = self._e_step(X)

            self._m_step(X, responsibilities)

            # Compute log likelihood to check convergence

            log_prob = np.zeros((X.shape[0], self.n_components))

            for k in range(self.n_components):

                log_bernoulli = X * np.log(self.theta_[k]) + (1 - X) * np.log(
                    1 - self.theta_[k]
                )

                log_prob[:, k] = np.sum(log_bernoulli, axis=1) + np.log(self.pi_[k])

                log_likelihood = np.sum(np.log(np.exp(log_prob).sum(axis=1)))

            if abs(log_likelihood - lower_bound) < self.tol:
                break

            lower_bound = log_likelihood

        return self

    def predict_proba(self, X):

        responsibilities = self._e_step(X)

        return responsibilities

    def predict(self, X):

        X = self._check_X_test(X)

        responsibilities = self.predict_proba(X)

        return responsibilities.argmax(axis=1)

    def _check_X(self, X):
        """
        Checks validity of input for all mixture models
        """
        X = check_array(X, accept_sparse=["csr"])
        # check that number of components is smaller or equal to number of samples
        if X.shape[0] < self.n_components:
            raise ValueError(
                ("Number of components should not be larger than " "number of samples")
            )

        return X

    def _check_X_train(self, X):
        """Preprocesses & check validity of training data"""

        X = self._check_X(X)
        self.classes_ = _get_classes(X)
        n = len(self.classes_)
        # check that there are only two categories in data
        if n != 2:
            raise ValueError(
                (
                    "There are {0} categorical values in data, "
                    "model accepts data with only 2".format(n)
                )
            )
        return 1 * (X == self.classes_[1])

    def transform(self, X):
        pred = (self.predict_proba(X) > TRS).astype(int)

        return pred

    def fit_transform(self, X, y):  # need this y to implement pipeline
        self.fit(X)

        return self.transform(X)

    def _check_X_test(self, X):
        """Preprocesses & check validity of test data"""

        X = check_array(X, accept_sparse=["csr"])
        classes_ = _get_classes(X)
        n = len(classes_)
        # check number of classes
        if n != 2:
            raise ValueError(
                (
                    "There are {0} categorical values in data, "
                    "model accepts data with only 2".format(n)
                )
            )
            # check whether these are the same classes as in training
        if classes_[0] == self.classes_[0] and classes_[1] == self.classes_[1]:
            return 1 * (X == self.classes_[1])
        else:
            raise ValueError(
                (
                    "Classes in training and test set are different, "
                    "{0} in training, {1} in test".format(self.classes_, classes_)
                )
            )



# BERNOULLI MIXTURE WITH SDG
################################################################################


import numpy as np
import torch
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted


class BernoulliMixtureSGD(BaseEstimator, ClusterMixin):
    def __init__(
        self,
        n_components=2,
        max_iter=1000,
        tol=1e-4,
        learning_rate=0.1,
        batch_size=128,
        random_state=None,
        device="cpu",
    ):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.random_state = random_state
        self.device = device

    def _initialize_parameters(self, n_features):
        torch.manual_seed(self.random_state or 0)

        self.pi_ = torch.nn.Parameter(
            torch.zeros(self.n_components, device=self.device)
        )
        # Learn unconstrained logits instead of probabilities directly
        self.theta_logits_ = torch.nn.Parameter(
            torch.randn(self.n_components, n_features, device=self.device) * 0.1
        )

        self.optimizer = torch.optim.Adam(
            [self.pi_, self.theta_logits_], lr=self.learning_rate
        )

    def get_probs(self):
        theta_tensor = torch.sigmoid(self.theta_logits_)
        feature_probs = theta_tensor.T.detach().cpu().numpy()
        return torch.softmax(self.pi_, dim=0).detach().cpu().numpy(), feature_probs

    def _compute_responsibilities(self, X):
        # Get probabilities from logits
        theta = torch.sigmoid(self.theta_logits_)  # <<< APPLY SIGMOID

        log_pi = torch.log_softmax(self.pi_, dim=0)
        log_theta = torch.log(theta + 1e-10)  # Use the squashed theta
        log_1mtheta = torch.log(1 - theta + 1e-10)
        # (n_components, n_samples, n_features)
        log_probs = X * log_theta[:, None] + (1 - X) * log_1mtheta[:, None]

        # (n_components, n_samples)
        log_probs_sum = log_probs.sum(dim=2) + log_pi[:, None]

        # (n_samples, n_components)
        log_resp = log_probs_sum.T - torch.logsumexp(
            log_probs_sum.T, dim=1, keepdim=True
        )
        return torch.exp(log_resp)

    def fit(self, X, y=None):
        X = check_array(X, accept_sparse="csr")
        n_samples, n_features = X.shape

        # Convert to PyTorch tensor
        X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)

        self._initialize_parameters(n_features)
        prev_loss = np.inf

        for epoch in range(self.max_iter):
            # Mini-batch training
            indices = torch.randperm(n_samples, device=self.device)[: self.batch_size]
            batch = X_tensor[indices]

            self.optimizer.zero_grad()

            theta = torch.sigmoid(self.theta_logits_)  # <<< APPLY SIGMOID

            responsibilities = self._compute_responsibilities(batch)
            loss = -torch.sum(
                responsibilities
                * (
                    batch @ torch.log(theta + 1e-10).T
                    + (1 - batch) @ torch.log(1 - theta + 1e-10).T
                    + torch.log_softmax(self.pi_, dim=0)
                )
            )

            loss.backward()
            self.optimizer.step()

            # Check convergence
            if abs(loss.item() - prev_loss) < self.tol:
                break
            prev_loss = loss.item()

        return self

    def transform(self, X):
        pred = (self.predict_proba(X) > TRS).astype(int)

        return pred

    def fit_transform(self, X, y):  # need this y to implement pipeline
        self.fit(X)

        return self.transform(X)

    def predict_proba(self, X):
        check_is_fitted(self)
        X = check_array(X, accept_sparse="csr")
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
            return self._compute_responsibilities(X_tensor).cpu().numpy()

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)

    def _get_param_names(cls):
        return [
            "n_components",
            "max_iter",
            "tol",
            "learning_rate",
            "batch_size",
            "random_state",
            "device",
        ]

    def get_params(self, deep=True):
        return {param: getattr(self, param) for param in self._get_param_names()}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def cluster_prototype(self, threshold):
        """
        Returns the binary prototype (most likely vector) for each cluster.

        Output shape: (n_features, n_components), matching VB version.
        """
        return (self.theta_ >= threshold).int().cpu().numpy()


# Hierarchical Model
################################################################################


from scipy.special import logsumexp


class HierachicalBernoulliMixture(BaseEstimator, ClusterMixin):
    def __init__(self, bottom_tier_model, n_components, **kwargs):
        self.bottom_tier_model = bottom_tier_model
        self.bottom_tier_args = kwargs
        self.n_components = n_components

    def fit(self, X, y):
        X, y = check_X_y(X, y)

        # 1. Calculate class priors (pi and 1-pi from the paper)
        self.classes_, counts = np.unique(y, return_counts=True)
        # Assuming class 0 and 1
        self.p0_ = counts[0] / len(y)  # This is (1 - pi)
        self.p1_ = counts[1] / len(y)  # This is pi

        # 2. Split data by class
        X0 = X[y == self.classes_[0]]
        X1 = X[y == self.classes_[1]]

        # 3. Fit one VBBMM per class to learn the class-conditional parameters
        # This model learns the parameters for P(x | y=0)
        self.model_0_ = self.bottom_tier_model(
            n_components=self.n_components, **self.bottom_tier_args
        )
        self.model_0_.fit(X0)
        # self.model_0_.weights_ are the {mu_{k|y=0}}
        # self.model_0_.means_ are the {p_{k|y=0}}

        # This model learns the parameters for P(x | y=1)
        self.model_1_ = self.bottom_tier_model(
            n_components=self.n_components, **self.bottom_tier_args
        )
        self.model_1_.fit(X1)
        # self.model_1_.weights_ are the {mu_{k|y=1}}
        # self.model_1_.means_ are the {p_{k|y=1}}

        self.is_fitted_ = True
        return self

    def extract_rule(self, classes, theta_clusters=0.20, theta_classes=0.5):

        cls = np.array(classes)

        mixing_factors, feature_probs = self.model_1_.get_probs()

        cluster_mask = mixing_factors > theta_clusters
        feature_mask = feature_probs > theta_classes

        sel_clusters = feature_mask[:, cluster_mask].T

        dsj_terms = []

        for cluster in sel_clusters:
            dsj_terms.append(And(cls[cluster].tolist()))

        return Or(dsj_terms)

    def predict_proba(self, X):
        check_is_fitted(self, "is_fitted_")
        X = check_array(X, accept_sparse=True)  # Ensure X is in the right format

        # Calculate log P(x | y=1) using model_1_
        log_likelihood_1 = self._calculate_log_likelihood(X, self.model_1_)

        # Calculate log P(x | y=0) using model_0_
        log_likelihood_0 = self._calculate_log_likelihood(X, self.model_0_)

        # Now calculate the final posterior in log-space
        # log( P(y=1) * P(x|y=1) )
        log_posterior_1 = np.log(self.p1_) + log_likelihood_1

        # log( P(y=0) * P(x|y=0) )
        log_posterior_0 = np.log(self.p0_) + log_likelihood_0

        # Normalize to get the final probabilities
        # The denominator is log( P(y=1)P(x|y=1) + P(y=0)P(x|y=0) )
        log_denominator = np.logaddexp(log_posterior_1, log_posterior_0)

        log_prob_y1 = log_posterior_1 - log_denominator

        # Return probabilities for both classes [P(y=0), P(y=1)]
        # Note: exp(log_prob_y1) is P(y=1|x)
        # P(y=0|x) = 1 - P(y=1|x)
        prob_y1 = np.exp(log_prob_y1)
        prob_y0 = 1 - prob_y1

        return np.vstack((prob_y0, prob_y1)).T

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)

        res = self.predict_proba(X)

        return res.argmax(axis=1)

    def _calculate_log_likelihood(self, X, model):
        # Get the parameters from the fitted VBBMM
        # means_ shape is (n_features, n_components)
        # weights_ shape is (n_components,)

        weights, feature_probs = model.get_probs()

        log_weights = np.log(weights)

        # Clip probabilities to avoid log(0)
        epsilon = 1e-10
        feature_probs = np.clip(feature_probs, epsilon, 1 - epsilon)
        log_feature_probs = np.log(feature_probs)
        log_one_minus_feature_probs = np.log(1 - feature_probs)

        # For each component, calculate the log-likelihood of each sample
        # This results in a matrix of shape (n_samples, n_components)
        # log P(x | z_k) for each k
        log_likelihood_per_component = (
            X @ log_feature_probs + (1 - X) @ log_one_minus_feature_probs
        )

        # Add the log weights to get the log of the joint probability
        # log( P(x|z_k) * P(z_k) )
        log_joint_prob = log_likelihood_per_component + log_weights

        # Sum over the components in log-space to get the log marginal likelihood
        # This is log( sum_k( P(x|z_k) * P(z_k) ) ) = log P(x)
        # The result is an array of shape (n_samples,)
        log_likelihood_total = logsumexp(log_joint_prob, axis=1)

        return log_likelihood_total

    def __str__(self):
        return f"HierachicalBernoulliMixture({self.bottom_tier_model.__name__})"


class TwoTierMixture(BaseEstimator, ClusterMixin):
    def __init__(self, bottom_tier_model, n_components, **kwargs):
        self.bottom_tier_model = bottom_tier_model
        self.bottom_tier_args = kwargs
        self.n_components = n_components

    def fit(self, X, y):
        X, y = check_X_y(X, y)

        # 1. Calculate class priors (pi and 1-pi from the paper)
        self.classes_, counts = np.unique(y, return_counts=True)

        # This model learns the parameters for P(x | y=1)
        self.model_1_ = self.bottom_tier_model(
            n_components=self.n_components, **self.bottom_tier_args
        )
        self.model_1_.fit(X)

        self.clss_model = LogisticRegression(
            C=0.01, penalty="l1", multi_class="multinomial", solver="saga", max_iter=200
        )

        self.clss_model.fit(self.model_1_.predict_proba(X), y)

        self.is_fitted_ = True
        return self

    def predict_proba(self, X):
        return self.clss_model.predict_proba(self.model_1_.predict_proba(X))

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)

        res = self.predict_proba(X)

        return res.argmax(axis=1)
