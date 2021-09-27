import os, sys

PATH = os.path.abspath(__file__)
DIR_PATH = os.path.dirname(PATH)
sys.path.append(DIR_PATH)

# add when heterogeneous methods are included in rmf_tool repository
# from rmf_tool.src.rmf_tool import DDPP
# from rmf_tool.src.rmf_tool import DimensionError, NotImplemented, InitialConditionNotDefined, NegativeRate
# from rmf_tool.src.het_refinedRefined_transientRegime import drift_r_vector


from src.rmf_tool import DDPP
from src.rmf_tool import DimensionError, NotImplemented, InitialConditionNotDefined, NegativeRate
from src.het_refinedRefined_transientRegime import drift_r_vector

import numpy as np

import scipy
import scipy.integrate as integrate
from scipy.interpolate import interp1d
import random as rnd

import copy


class HetPP(DDPP):
    def __init__(self):
        # initialize the heterogeneous model
        super().__init__()
        self._model_dimension_N = None
        self._model_dimension_S = None

        self.ddF = None

    def add_rate_tensors(self, A, B):
        r"""
        Rate tensors must have the form:
        A[k, s, s_prime] specifying the unilateral transition of object k from s to s_prime
        B[k, k1, s, s1, s_prime, s1_prime] pairwise transition of object k from s to s_prime and k1 from s1 to s1_prime

        A has dimensions N x S x S
        B has dimensions N x N x S x S x S x S
        """
        # TODO - move into __init__ method?
        # add transitions based on the rate tensors for unilateral / binary transitions
        # derive transitions from nonzero entries of the vectors
        # fix vector structure to infer transition vectors

        # set model dimensions
        if self._model_dimension_N is not None and self._model_dimension_N != A.shape[0]:
            if self._model_dimension_S is not None and self._model_dimension_S != B.shape[1]:
                raise DimensionError

        # infer model dimensions
        self._model_dimension_N = A.shape[0]
        self._model_dimension_S = A.shape[1]

        self._model_dimension = (self._model_dimension_N, self._model_dimension_S)

        # save rate tensors in model
        if np.sum(A < -1e-14) >= 1:
            raise NegativeRate("Rate tensor A has negative rates.")
        if np.sum(B < -1e-14) >= 1:
            raise NegativeRate("Rate tensor B has negative rates")
        self.A = A
        self.B = B

        # TODO - diagonal entries should be zero ... implement test + warning

    def simulate(self, time, seed=None):
        # TODO add seeds
        if self._x0 is None:
            raise InitialConditionNotDefined

        trans_indices_A = np.nonzero(self.A)
        nr_trans_A = trans_indices_A[0].shape[0]

        trans_indices_B = np.nonzero(self.B)
        nr_trans_B = trans_indices_B[0].shape[0]

        nr_trans = nr_trans_A + nr_trans_B

        x = copy.copy(self._x0)
        t = 0
        T = [0]
        X = [copy.copy(x)]

        def transition_rates(x):

            _trans_rates_A = np.zeros(shape=(trans_indices_A[0].shape))

            for i in range(nr_trans_A):
                k, s, s_prime = trans_indices_A[0][i], trans_indices_A[1][i], trans_indices_A[2][i]
                _trans_rates_A[i] = self.A[k, s, s_prime] * x[k, s]

            _trans_rates_B = np.zeros(shape=(trans_indices_B[0].shape))

            for i in range(nr_trans_B):
                k, k1 = trans_indices_B[0][i], trans_indices_B[1][i]
                s, s1 = trans_indices_B[2][i], trans_indices_B[3][i]
                s_prime, s1_prime = trans_indices_B[4][i], trans_indices_B[5][i]
                _trans_rates_B[i] = self.B[k, k1, s, s1, s_prime, s1_prime] * x[k, s] * x[k1, s1]

            return _trans_rates_A, _trans_rates_B

        rnd.seed(seed)

        while t < time:
            # transition rates evaluated at x
            trans_rates_A, trans_rates_B = transition_rates(x)
            S = np.sum(trans_rates_A) + np.sum(trans_rates_B)

            if S <= 1e-12:
                print('System stalled (total rate = 0)')
                t = time
            else:
                a = rnd.random() * S
                # selected transition
                transition_selected = False
                for i in range(trans_rates_A.shape[0]):
                    if a > trans_rates_A[i]:
                        a -= trans_rates_A[i]
                    else:
                        transition_selected = True
                        k, s, s_prime = trans_indices_A[0][i], trans_indices_A[1][i], trans_indices_A[2][i]
                        # add unilateral transition to state vector
                        x += (- self.e(k, s) + self.e(k, s_prime))
                        break

                if transition_selected is False:
                    for i in range(trans_rates_B.shape[0]):
                        if a > trans_rates_B[i]:
                            a -= trans_rates_B[i]
                        else:
                            transition_selected = True
                            k, k1 = trans_indices_B[0][i], trans_indices_B[1][i]
                            s, s1 = trans_indices_B[2][i], trans_indices_B[3][i]
                            s_prime, s1_prime = trans_indices_B[4][i], trans_indices_B[5][i]
                            # add unilateral transition to state vector
                            x += (- self.e(k, s) + self.e(k, s_prime) - self.e(k1, s1) + self.e(k1, s1_prime))
                            break

                t += rnd.expovariate(S)

            T.append(t)
            X.append(copy.copy(x))

        X = np.array(X)
        return T, X

    def sampleMeanVariance(self, time=10, samples=1000, time_steps=2000):
        # TODO add seeds

        # initialize the mean values (simulation data come as array nr_items*nr_caches)
        sim_mean = np.zeros(shape=(time_steps, self._model_dimension_N, self._model_dimension_S))
        error_var = np.zeros(shape=(time_steps, self._model_dimension_N, self._model_dimension_S))

        # print("Calculating mean for " + str(nr_simulations) + " simulations.")
        interpolation_times = np.linspace(0, time, time_steps)

        print("Mean Calculation")
        for i in range(samples):
            if ((i + 1) % 10) == 0:
                print(i+1, ' ', end='')
            _T, _X = self.simulate(time=time, seed=i)
            # append new interpolation to list important that result from interpolations need to be transposed again
            # to obtain (nr_item, state) sized array
            interpolation = interp1d(_T, _X, axis=0)
            for k, time in enumerate(interpolation_times):
                sim_mean[k] += interpolation(time)
        sim_mean *= 1 / float(samples)

        print("\nVariance")
        for i in range(samples):
            if ((i + 1) % 10) == 0:
                print(i+1, ' ', end='')
            _T, _X = self.simulate(time=time, seed=i)
            # append new interpolation to list important that result from interpolations need to be transposed again
            # to obtain (nr_item, state) sized array
            interpolation = interp1d(_T, _X, axis=0)
            for k, time in enumerate(interpolation_times):
                error_var[k] += np.power((interpolation(time) - sim_mean[k]), 2)
        error_var *= 1 / float(samples - 1)

        return interpolation_times, sim_mean, error_var

    def defineDrift(self, evaluate_at=None):
        """
        Defines the drift vector given the transition tensors A and B.
        The drift is of the dimension N x S .

        returns numpy.ndarray
        """

        # raise error if evaluate_at is None
        if evaluate_at is None:
            raise NotImplemented("A lambdified version of the drift is currently only available for the homogeneous "
                                 "implementation.")

        # drift vector
        _F = np.zeros(shape=(self._model_dimension_N, self._model_dimension_S))

        x = evaluate_at

        # use non zero rate entries to filter actual transitions
        trans_indices_A = np.nonzero(self.A)
        nr_pair_trans_A = trans_indices_A[0].shape[0]

        for i in range(nr_pair_trans_A):
            k, s, s_prime = trans_indices_A[0][i], trans_indices_A[1][i], trans_indices_A[2][i]
            _F += self.A[k, s, s_prime] * x[k, s] * (- self.e(k, s) + self.e(k, s_prime))

        trans_indices_B = np.nonzero(self.B)
        nr_pair_trans_B = trans_indices_B[0].shape[0]

        for i in range(nr_pair_trans_B):
            k, k1 = trans_indices_B[0][i], trans_indices_B[1][i]
            s, s1 = trans_indices_B[2][i], trans_indices_B[3][i]
            s_prime, s1_prime = trans_indices_B[4][i], trans_indices_B[5][i]
            _F += self.B[k, k1, s, s1, s_prime, s1_prime] * x[k, s] * x[k1, s1] * \
                  (- self.e(k, s) + self.e(k, s_prime) - self.e(k1, s1) + self.e(k1, s1_prime))

        return _F

    def set_initial_state(self, x0):
        r"""
        Sets the initial state of the system to x0.
        """
        # make sure initial state dimension match with from tensor inferred dimensions
        if self._model_dimension_N is not None and self._model_dimension_S is not None:
            if self._model_dimension_N != x0.shape[0] or self._model_dimension_S != x0.shape[1]:
                raise DimensionError("The model dimensions do not match. Make sure the model is properly initialized.")
        if self._model_dimension_N is None and self._model_dimension_S is None:
            self._model_dimension_N, self._model_dimension_S = x0.shape[0], x0.shape[1]
        self._x0 = x0

    def defineDriftDerivativeQ_autograd(self, evaluate_at=None):
        """
        Autograd and Jax methods are not available for the heteroegeneous model.
        """
        raise NotImplementedError("The defineDriftDerivativeQ_autograd method is only available for "
                                  "the homogeneous model.")

    def defineDriftDerivativeQ(self, evaluate_at=None):
        """
        Defines the drift derivatives and Q matrix a given point. Used to calculate the refinement term.
        """

        if evaluate_at is None:
            raise NotImplemented("A lambdified version of the drift is currently only "
                                 "available for the homogeneous implementation.")

        # redefine evaluation point to define the derivatives and Q more comprehensively
        x = evaluate_at

        # use non zero rate entries to filter actual transitions
        trans_indices_A = np.nonzero(self.A)
        nr_unit_trans_A = trans_indices_A[0].shape[0]

        trans_indices_B = np.nonzero(self.B)
        nr_pair_trans_B = trans_indices_B[0].shape[0]

        # initialize Q matrix with dimension (N x S x N x S)
        Q = np.zeros(shape=(self._model_dimension_N, self._model_dimension_S,
                            self._model_dimension_N, self._model_dimension_S))

        # add transitions to the Q tensor
        # unilateral transitions
        for i in range(nr_unit_trans_A):
            # transition indices
            k, s, s_prime = trans_indices_A[0][i], trans_indices_A[1][i], trans_indices_A[2][i]
            # transition tensor
            transition_step = - self.e(k, s) + self.e(k, s_prime)
            transition_tensor = np.tensordot(transition_step, transition_step, axes=0)
            # add weighted transition to Q
            Q += self.A[k, s, s_prime] * x[k, s] * transition_tensor

        # pairwise transitions
        for i in range(nr_pair_trans_B):
            # transition indices
            k, k1 = trans_indices_B[0][i], trans_indices_B[1][i]
            k, k1 = trans_indices_B[0][i], trans_indices_B[1][i]
            s, s1 = trans_indices_B[2][i], trans_indices_B[3][i]
            s_prime, s1_prime = trans_indices_B[4][i], trans_indices_B[5][i]
            # transition tensor
            transition_step = (- self.e(k, s) + self.e(k, s_prime) - self.e(k1, s1) + self.e(k1, s1_prime))
            transition_tensor = np.tensordot(transition_step, transition_step, axes=0)
            # add weighted transition to Q
            Q += self.B[k, k1, s, s1, s_prime, s1_prime] * x[k, s] * x[k1, s1] * transition_tensor

        # define first derivative of drift
        # Jakobian has the dimensions (N x S) x (N x S)
        dF = np.zeros(shape=(self._model_dimension_N, self._model_dimension_S,
                             self._model_dimension_N, self._model_dimension_S))
        # unilateral transitions
        for i in range(nr_unit_trans_A):
            # transition indices
            k, s, s_prime = trans_indices_A[0][i], trans_indices_A[1][i], trans_indices_A[2][i]
            rate = self.A[k, s, s_prime]
            dF[k, s, k, s] -= self.A[k, s, s_prime]
            dF[k, s_prime, k, s] += self.A[k, s, s_prime]

        # pairwise transitions
        for i in range(nr_pair_trans_B):
            k, k1 = trans_indices_B[0][i], trans_indices_B[1][i]
            s, s1 = trans_indices_B[2][i], trans_indices_B[3][i]
            s_prime, s1_prime = trans_indices_B[4][i], trans_indices_B[5][i]
            # define derivatives
            dF[k, s, k, s] -= self.B[k, k1, s, s1, s_prime, s1_prime] * x[k1, s1]
            dF[k1, s1, k, s] -= self.B[k, k1, s, s1, s_prime, s1_prime] * x[k1, s1]
            dF[k, s, k1, s1] -= self.B[k, k1, s, s1, s_prime, s1_prime] * x[k, s]
            dF[k1, s1, k1, s1] -= self.B[k, k1, s, s1, s_prime, s1_prime] * x[k, s]

            dF[k, s_prime, k, s] += self.B[k, k1, s, s1, s_prime, s1_prime] * x[k1, s1]
            dF[k1, s1_prime, k, s] += self.B[k, k1, s, s1, s_prime, s1_prime] * x[k1, s1]
            dF[k, s_prime, k1, s1] += self.B[k, k1, s, s1, s_prime, s1_prime] * x[k, s]
            dF[k1, s1_prime, k1, s1] += self.B[k, k1, s, s1, s_prime, s1_prime] * x[k, s]

        # define first derivative of drift
        # Hessian has the dimensions (N x S) x (N x S) x (N x S)

        if self.ddF is None:

            ddF = np.zeros(shape=(self._model_dimension_N, self._model_dimension_S,
                                  self._model_dimension_N, self._model_dimension_S,
                                  self._model_dimension_N, self._model_dimension_S))
            # pairwise transitions ( no unilateral transitions)
            for i in range(nr_pair_trans_B):
                k, k1 = trans_indices_B[0][i], trans_indices_B[1][i]
                s, s1 = trans_indices_B[2][i], trans_indices_B[3][i]
                s_prime, s1_prime = trans_indices_B[4][i], trans_indices_B[5][i]
                # define derivatives
                # drift / dx[k,s] dx[k1,s1]
                ddF[k, s, k, s, k1, s1] -= self.B[k, k1, s, s1, s_prime, s1_prime]
                ddF[k, s_prime, k, s, k1, s1] += self.B[k, k1, s, s1, s_prime, s1_prime]
                ddF[k1, s1, k, s, k1, s1] -= self.B[k, k1, s, s1, s_prime, s1_prime]
                ddF[k1, s1_prime, k, s, k1, s1] += self.B[k, k1, s, s1, s_prime, s1_prime]
                # drift / dx[k1,s1] dx[k,s] - permutation of derivative direction
                ddF[k, s, k1, s1, k, s] -= self.B[k, k1, s, s1, s_prime, s1_prime]
                ddF[k, s_prime, k1, s1, k, s] += self.B[k, k1, s, s1, s_prime, s1_prime]
                ddF[k1, s1, k1, s1, k, s] -= self.B[k, k1, s, s1, s_prime, s1_prime]
                ddF[k1, s1_prime, k1, s1, k, s] += self.B[k, k1, s, s1, s_prime, s1_prime]

            self.ddF = ddF
        else:
            ddF = self.ddF

        return (dF, ddF, Q)

    def meanFieldExpansionTransient(self, order=1, time=10):
        """
        Computes the transient values of the mean field approximation or its O(1/N^{order})-expansions

        Args:
           - order : can be 0 (mean field approx.) or 1 (O(1/N)-expansion);


        Returns :
            for order 0 - (T, X) where T is a time interval, X is the solution to mean field approximation

            for order 1 - (T, X, V, XVW) where T is a time interval, X is the solution to the mean field approximation,
            V is the solution to the refinement term and XVW is a (2(N*S)+(N*S)^2)*number_of_steps matrix  where :
                * XVW[0:n,:]                 is the solution of the ODE (= mean field approximation, equal to X)
                * XVW[n:2*n,:]               is V(t) (= 1st order correction, equal to V)
                * XVW[2*n:2*n+n**2,:]        is W(t)
        """

        if (order >= 2):
            raise NotImplementedError("Second order methods can not be implemented for polynomial drift of order two.")

        # set model dimensions
        N = self._model_dimension_N
        S = self._model_dimension_S

        if order == 0:
            # return mean field model
            T, X = self.ode(time=time, number_of_steps=1000)
            return T, X

        if order == 1:
            # allocate N*S values for the mean field X, N*S values for the expansion term V
            # and (N*S)**2 values for the expansion term W
            XVW_0 = np.zeros(2 * N * S + (N * S) ** 2)
            XVW_0[0:N * S] = self._x0.flatten()

            # max time of the integration
            Tmax = time
            # specify time steps
            T = np.linspace(0, Tmax, 1000)

            # calculating the solution to the ivp
            numericalInteg = integrate.solve_ivp(lambda t, x:
                                                 drift_r_vector(
                                                     x, N, S, self.defineDrift, self.defineDriftDerivativeQ),
                                                 [0, Tmax], XVW_0, t_eval=T, rtol=1e-8)

            # separating and reshaping X and V values
            XVW = numericalInteg.y.transpose()

            X = XVW[:, 0:(N * S)]
            # reshape X values into (N x S) format
            X = np.array([X[i, :].reshape((N, S)) for i in range(X.shape[0])])

            V = XVW[:, (N * S):2 * (N * S)]
            # reshape V values into (N x S) format
            V = np.array([V[i, :].reshape((N, S)) for i in range(V.shape[0])])

            return numericalInteg.t, X, V, XVW

    def meanFieldExpansionSteadyState(self, order=1):
        """This code computes the O(1/N) and O(1/N^2) expansion of the mean field approximaiton
        (the term "V" is the "V" of Theorem~1 of https://hal.inria.fr/hal-01622054/document.

        Note : Probably less robust and slower that theoretical_V
        """
        pi = self.ode(time=10000)[1][-1]

        if order == 0:
            return pi
        if (order >= 1):  # We need 2 derivatives and Q to get the O(1/N)-term
            Fp, Fpp, Q = self.defineDriftDerivativeQ(pi)

            # reshaping tensors
            N = self._model_dimension_N
            S = self._model_dimension_S
            pi = pi.flatten()
            Q = Q.reshape((N * S, N * S))
            Fp = Fp.reshape((N * S, N * S))
            Fpp = Fpp.reshape((N * S, N * S, N * S))

            # reduce dimension for steady state calculation
            Fp, Fpp, Q, P, Pinv, rank = self.reduceDimensionFpFppQ(Fp, Fpp, Q)
            if order == 1:
                pi, V, (V, W) = self.computePiV(pi, Fp, Fpp, Q)
                V, W = self.expandDimensionVW(V, W, Pinv)

                # reshaping into original representation
                pi = pi.reshape((N, S))
                V = V.reshape((N, S))
                W = W.reshape((N, S, N, S))
                return pi, V, (V, W)

    def reduceDimensionFpFppQ(self, Fp, Fpp, Q):
        P, P_inv, rank = self.dimensionReduction(Fp)
        Fp = (P @ Fp @ P_inv)[0:rank, 0:rank]
        Fpp = np.tensordot(np.tensordot(np.tensordot(P, Fpp, 1), P_inv, 1), P_inv,
                           axes=[[1], [0]])[0:rank, 0:rank, 0:rank]
        Q = (P @ Q @ P.transpose())[0:rank, 0:rank]
        return Fp, Fpp, Q, P, P_inv, rank

    def dimensionReduction(self, A):
        n = len(A)

        # M = np.array([l for l in self._list_of_transitions])
        M = self.get_transition_matrix()
        rank_of_transitions = np.linalg.matrix_rank(M)
        # If rank_of_transisions < n, this means that the stochastic process
        # evolves on a linear subspace of R^n.
        eigenvaluesOfJacobian = scipy.linalg.eig(A, left=False, right=False)
        rank_of_jacobian = np.linalg.matrix_rank(A)
        if sum(np.real(eigenvaluesOfJacobian) < 1e-8) < rank_of_transitions:
            # This means that there are less than "rank_of_transisions"
            # eigenvalues with <0 real part
            print("The Jacobian seems to be not Hurwitz")
        if rank_of_jacobian == n:
            return np.eye(n), np.eye(n), n
        C = np.zeros((n, n))
        n = len(A)
        d = 0
        rank_of_previous_submatrix = 0
        for i in range(n):
            rank_of_next_submatrix = np.linalg.matrix_rank(A[0:i + 1, 0:i + 1])
            if rank_of_next_submatrix > rank_of_previous_submatrix:
                C[d, i] = 1
                d += 1
            rank_of_previous_submatrix = rank_of_next_submatrix
        U, s, V = scipy.linalg.svd(A)
        C[rank_of_jacobian:, :] = U.transpose()[rank_of_jacobian:, :]
        return C, scipy.linalg.inv(C), rank_of_jacobian

    def get_transition_matrix(self):

        M = []

        trans_indices_A = np.nonzero(self.A)
        nr_unit_trans_A = trans_indices_A[0].shape[0]

        trans_indices_B = np.nonzero(self.B)
        nr_pair_trans_B = trans_indices_B[0].shape[0]
        # arrivals
        for i in range(nr_unit_trans_A):
            # transition indices
            k, s, s_prime = trans_indices_A[0][i], trans_indices_A[1][i], trans_indices_A[2][i]
            # add transition to M
            M.append((- self.e(k, s) + self.e(k, s_prime)).flatten())

        # pairwise transitions
        for i in range(nr_pair_trans_B):
            # transition indices
            k, k1 = trans_indices_B[0][i], trans_indices_B[1][i]
            s, s1 = trans_indices_B[2][i], trans_indices_B[3][i]
            s_prime, s1_prime = trans_indices_B[4][i], trans_indices_B[5][i]
            # add transition to M
            M.append((- self.e(k, s) + self.e(k, s_prime) - self.e(k1, s1) + self.e(k1, s1_prime)).flatten())

        return np.array(M)

    def computePiV(self, pi, Fp, Fpp, Q):
        """
        Returns the constants V and W (1/N-term for the steady-state)

        This function assumes that Fp is invertible.
        """
        from scipy.linalg import solve_continuous_lyapunov, inv

        # W = computeW(Fp, Q)
        W = solve_continuous_lyapunov(Fp, -Q)
        # V = computeV(Fp, Fpp, W)
        V = -np.tensordot(inv(Fp),
                          np.tensordot(Fpp, W / 2, 2),
                          1)
        return pi, V, (V, W)

    def expandDimensionVW(self, V, W, P_inv):
        rank = len(V)
        return P_inv[:, 0:rank] @ V, P_inv[:, 0:rank] @ W @ P_inv.transpose()[0:rank, :]

    def ode(self, time, number_of_steps=1000):
        # TODO - naming -> meanFieldTransient?
        """
        Simulates the ODE (mean-field approximation) up to a given time.
        """
        if self._x0 is None:
            raise InitialConditionNotDefined

        T = np.linspace(0, time, number_of_steps)

        def vector_valued_drift(x):
            # reshape input vector
            x = x.reshape((self._model_dimension_N, self._model_dimension_S))
            # return drift at x
            return self.defineDrift(x).flatten()

        X = integrate.odeint(lambda x, t: vector_valued_drift(x), self._x0.flatten(), T)

        X = np.array([X[i, :].reshape((self._model_dimension_N, self._model_dimension_S))
                      for i in range(X.shape[0])])

        return T, X

    def fixed_point(self):
        """
        Computes the fixed of the ODE (if this ODE has a fixed point starting from x0).
        """
        if self._x0 is None:
            print(
                'No initial condition given. We assume that the initial condition is "x0=[[1,0,...],[1,0,...],...]"')
            self._x0 = np.zeros(self._model_dimension)
            self._x0[:, 0] = 1
        # TODO - test function
        return super().fixed_point()

    def e(self, k, s):
        """
        Defines a unit vector in the size of the model.
        """
        e_ = np.zeros(shape=(self._model_dimension_N, self._model_dimension_S))
        e_[k, s] = 1
        return e_
