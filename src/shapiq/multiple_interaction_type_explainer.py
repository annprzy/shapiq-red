"""
module explaining the interaction types
"""
import numpy as np
from shapiq import TabularExplainer
from scipy.stats import norm, rankdata
from scipy.stats import gaussian_kde


class TypeExplainer:
    """
    interaction type explainer class
    """
    def __init__(self, x, model, sample_data, index="RI", max_order=2, budget=256):
        self.x = x
        self.model = model
        self.data = sample_data
        self.index = index
        self.budget = budget
        self.max_order = max_order
    def prepare_data(self, data, model, x, trials):
        """
        prepare data for o-information computation
        """
        data_with_y = []
        ind = -1
        
        for i in range(len(data)):
            if ind == -1 and (x == data[i]).all():
                ind = i
            data_with_y.append(np.append(data[i], np.array(model(np.array([data[i]]))[0])))
        if ind == -1:
            ind = len(data_with_y) - 1
            data_with_y.append(np.append(x, np.array(model(np.array([x]))[0])))
        s=len(data_with_y)
        for i in range(s):
            for j in range(trials):
                data_with_y.append(data_with_y[i])
        data_with_y = np.array(data_with_y)

        return ind, data_with_y

    def o_information(self, x, model, data, coalition, trials=100, data_type="discrete"):
        """
        compute o-information of a given input
        """
        coalition = list(coalition)
        ind,prepared_data = self.prepare_data(data, model, x, trials)
        n=len(x)
        to_delete = []
        for i in range(n):
            if i not in coalition:
                to_delete.append(i-1)
        prepared_data = np.delete(prepared_data, to_delete, axis=1)
        x = np.delete(x, to_delete, axis=0)
        oinfo = self.continuous_local_oinfo(prepared_data, data_type=data_type)
        return oinfo[ind]
    def copula_transform(self,  X: np.ndarray) -> np.ndarray:
        """Transforms continuous/clustered feature columns into standard Gaussian distributions

        (N(0,1)) using empirical probability integral transform (Copula).
        This normalizes 1D, 3D, and 4D subspace scaling for KDE density estimation.
        """
        N, d = X.shape
        X_gauss = np.zeros_like(X)
        for col in range(d):
            ranks = (rankdata(X[:, col], method="average") - 0.5) / N
            X_gauss[:, col] = norm.ppf(ranks)
        return X_gauss
    def discrete_local_oinfo(self, X: np.ndarray) -> np.ndarray:
        data = np.asarray(X)
        n_samples, n_vars = data.shape
        
        if n_vars < 3:
            raise ValueError("O-information requires at least 3 variables.")

        def get_surprisals(subset):
            """Helper to calculate -log2(p) for every row in a subset."""
            _, inverse_idx, counts = np.unique(subset, axis=0, return_inverse=True, return_counts=True)
            
            probs = counts / n_samples
            return -np.log2(probs[inverse_idx])

        h_X = get_surprisals(data)

        sum_h_X_minus_i = np.zeros(n_samples)
        sum_h_X_i = np.zeros(n_samples)

        for i in range(n_vars):
            sum_h_X_i += get_surprisals(data[:, [i]])
            mask = [j for j in range(n_vars) if j != i]
            sum_h_X_minus_i += get_surprisals(data[:, mask])

        local_o_info = (n_vars - 2) * h_X + sum_h_X_minus_i - sum_h_X_i
        print(local_o_info)
        return local_o_info
    def continuous_local_oinfo(
        self, X: np.ndarray, bw_method: float = 0.4, data_type: str = "discrete"
    ) -> np.ndarray:
        """Computes Local O-information for continuous features using Gaussian Copula + KDE."""
        if data_type == "discrete":
            return self.discrete_local_oinfo(X=X)
        X_norm = self.copula_transform(X)

        N, n_features = X_norm.shape

        kde_full = gaussian_kde(X_norm.T, bw_method=bw_method)
        i_full = -kde_full.logpdf(X_norm.T)

        i_indiv = np.zeros(N)
        for j in range(n_features):
            kde_j = gaussian_kde(X_norm[:, j : j + 1].T, bw_method=bw_method)
            i_indiv += -kde_j.logpdf(X_norm[:, j : j + 1].T)

        i_leave_one_out = np.zeros(N)
        for j in range(n_features):
            subset = np.delete(X_norm, j, axis=1)
            kde_sub = gaussian_kde(subset.T, bw_method=bw_method)
            i_leave_one_out += -kde_sub.logpdf(subset.T)
        return (n_features - 2) * i_full + i_indiv - i_leave_one_out
    def copula_transform(self, X: np.ndarray) -> np.ndarray:
        """Transforms data to standard normal marginals via empirical copula

        to stabilize k-NN density estimation on clustered/jittered data.
        """
        N, d = X.shape
        X_norm = np.zeros_like(X)
        for col in range(d):
            ranks = rankdata(X[:, col], method="average") / (N + 1.0)
            X_norm[:, col] = norm.ppf(ranks)
        return X_norm
    def predict_type(self, coalition=None, trials=100, data_type="discrete"):
        """
        predict interaction type of a given input
        """
        if coalition is None:
            coalition = tuple(range(len(self.x)))
        x = self.x
        model = self.model
        data = self.data
        budget = self.budget
        oinfo = self.o_information(x, model, data, coalition, trials=trials, data_type=data_type)
        explainer = TabularExplainer(
            model=model,
            data=data,
            index="k-SII",
            max_order=len(x),
        ).explain(x, budget=budget)
        if explainer.dict_values[coalition]<0.00001 and explainer.dict_values[coalition]>-0.00001:
            return "independence"
        elif oinfo > 0:
            return "redundancy"
        else:
            for i in list(coalition):
                if self.sign(explainer[i+1]) != self.sign(explainer.dict_values[coalition]):
                    return "antagonism"
            return "synergy"

    

    def sign(self, a, margin=0.001):
        """
        check sign of a float
        """
        if a + margin > 0:
            return 1
        return -1
