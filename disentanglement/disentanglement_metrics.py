from sklearn.neural_network import MLPRegressor
from abc import ABC, abstractmethod
import numpy as np
from scipy.stats import entropy


class DisentanglementMetric(ABC):
    def __init__(self, model):
        self.model = model

    def compute_entropy(self, x, num_bins=30):
        counts, _ = np.histogram(x, bins=num_bins)
        return entropy(counts)

    def mutual_information(self, x, y, num_bins=30):
        counts_xy, _, _ = np.histogram2d(x, y, bins=num_bins)
        counts_x, _ = np.histogram(x, bins=num_bins)
        counts_y, _ = np.histogram(y, bins=num_bins)
        H_x = self.compute_entropy(counts_x)
        H_y = self.compute_entropy(counts_y)
        H_xy = self.compute_entropy(counts_xy.flatten())
        return H_x + H_y - H_xy

    @abstractmethod
    def compute(self, data):
        pass


class MIG(DisentanglementMetric):
    def compute(self, data):
        latent_vars = self.model.encode(data).detach().cpu().numpy()
        factors = data.detach().cpu().numpy()
        MIG_score = 0
        for i in range(latent_vars.shape[1]):
            MI = [self.mutual_information(
                latent_vars[:, i], factors[:, j]) for j in range(factors.shape[1])]
            MI.sort(reverse=True)
            MIG_score += MI[0] - MI[1]
        return MIG_score / latent_vars.shape[1]


class DCI(DisentanglementMetric):
    def compute(self, data):
        latent_vars = self.model.encode(data).detach().cpu().numpy()
        factors = data.detach().cpu().numpy()
        disentanglement_score = 0
        completeness_score = 0
        for j in range(factors.shape[1]):
            reg = MLPRegressor().fit(latent_vars, factors[:, j])
            preds = reg.predict(latent_vars)
            importance = np.abs(reg.coefs_)
            max_importance_idx = np.argmax(importance)
            disentanglement_score += (
                importance[max_importance_idx] / np.sum(importance))**2
            completeness_score += 2*np.corrcoef(factors[:, j], preds)[0, 1]**2
        return disentanglement_score / factors.shape[1], completeness_score / factors.shape[1]


class Modularity(DisentanglementMetric):
    def compute(self, data):
        latent_vars = self.model.encode(data).detach().cpu().numpy()
        factors = data.detach().cpu().numpy()
        modularity_score = 0
        for i in range(latent_vars.shape[1]):
            print(factors.shape, latent_vars.shape)
            reg = MLPRegressor().fit(factors, latent_vars[:, i])
            _ = reg.predict(factors)
            importance = np.abs(reg.coefs_)
            max_importance_idx = np.argmax(importance)
            modularity_score += (importance[max_importance_idx] /
                                 np.sum(importance))**2
        return modularity_score / latent_vars.shape[1]
