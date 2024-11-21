# This file is adapted from https://github.com/raymin0223/fast_robust_early_exit
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import torch
from transformers import AutoConfig


def weighted_mean(x, w):
    return np.sum(w * x) / np.sum(w)


def fit_beta_weighted(x, w):
    x_bar = weighted_mean(x, w)
    s2 = weighted_mean((x - x_bar) ** 2, w)
    alpha = x_bar * ((x_bar * (1 - x_bar)) / s2 - 1)
    beta = alpha * (1 - x_bar) / x_bar
    return alpha, beta


class BetaMixture1D(object):
    def __init__(
        self,
        max_iters=1,
        alphas_init=[1, 2],
        betas_init=[2, 1],
        weights_init=[0.5, 0.5],
    ):
        self.alphas = np.array(alphas_init, dtype=np.float64)
        self.betas = np.array(betas_init, dtype=np.float64)
        self.weight = np.array(weights_init, dtype=np.float64)
        self.max_iters = max_iters
        self.lookup_resolution = 10
        self.lam = 0.95
        self.lam_r = 0.8
        self.eps_nan = 1e-4
        self.zeta = 0.4

        self.init_alphas = np.array(alphas_init, dtype=np.float64)
        self.init_betas = np.array(betas_init, dtype=np.float64)
        self.init_weight = np.array(weights_init, dtype=np.float64)

    def likelihood(self, x, y):
        return stats.beta.pdf(x, self.alphas[y], self.betas[y])

    def weighted_likelihood(self, x, y):
        return self.weight[y] * self.likelihood(x, y)

    def probability(self, x):
        return sum(self.weighted_likelihood(x, y) for y in range(2))

    def posterior(self, x, y):
        return self.weighted_likelihood(x, y) / (self.probability(x) + self.eps_nan)

    def responsibilities(self, x, y=None):
        if y is None:
            r = np.array([self.weighted_likelihood(x, i) for i in range(2)])
        else:
            r = self.lam_r * np.array(
                [self.weighted_likelihood(x, i) for i in range(2)]
            )
            r = (1 - self.lam_r) * np.eye(2)[y].T

        # there are ~200 samples below that value
        r[r <= self.eps_nan] = self.eps_nan
        r /= r.sum(axis=0) + 1e-12
        return r

    def score_samples(self, x):
        return -np.log(self.probability(x))

    def fit(self, x, y=None):
        x = np.copy(x)

        # EM on beta distributions unable with x == 0 or 1
        eps = 1e-4
        x[x >= 1 - eps] = 1 - eps
        x[x <= eps] = eps

        if y is not None:
            return self.fit_supervised(x, y)
        else:
            return self.fit_unsupervised(x)

    def fit_supervised(self, x, y):
        for i in range(self.max_iters):
            # E-step
            r = self.responsibilities(x, y)

            # M-step
            self.alphas[0], self.betas[0] = fit_beta_weighted(x, r[0])
            self.alphas[1], self.betas[1] = fit_beta_weighted(x, r[1])

        return self

    def fit_unsupervised(self, x):
        for i in range(self.max_iters):
            # E-step
            r = self.responsibilities(x)

            # M-step
            for i in range(2):
                new_alpha, new_beta = fit_beta_weighted(x, r[i])
                self.alphas[i] = self.lam * self.alphas[i] + (1 - self.lam) * new_alpha
                self.betas[i] = self.lam * self.betas[i] + (1 - self.lam) * new_beta

            new_weight = r.sum(axis=1)
            new_weight /= new_weight.sum() + 1e-12
            self.weight = self.lam * self.weight + (1 - self.lam) * new_weight.reshape(
                -1
            )
        return self

    def predict(self, x):
        return self.posterior(x, 1) > 0.5

    def plot(self):
        x = np.linspace(0, 1, 100)
        plt.plot(x, self.weighted_likelihood(x, 0), label="negative")
        plt.plot(x, self.weighted_likelihood(x, 1), label="positive")

    def predict_proba(self, xmin=0.0, xmax=1.0):
        x = np.linspace(xmin, xmax, self.lookup_resolution)
        mean = self.alphas / (self.alphas + self.betas)
        i = mean.argmax()
        pred = (
            self.likelihood(x, i) / (self.likelihood(x, i) + self.likelihood(x, 1 - i))
            > self.zeta
        )
        idx = min(
            max(0, self.lookup_resolution - sum(pred)), self.lookup_resolution - 1
        )
        return x[idx]

    def reinit(self):
        self.alphas = np.copy(self.init_alphas)
        self.betas = np.copy(self.init_betas)
        self.weight = np.copy(self.init_weight)

    def __str__(self):
        return "BetaMixture1D(w={}, a={}, b={})".format(
            self.weight, self.alphas, self.betas
        )


def softmax_confidence(
    logits: torch.Tensor = None,
    hidden_states: torch.Tensor = None,
    classifier: torch.nn.Linear = None,
):
    assert logits is not None
    probs = torch.softmax(logits, dim=-1)
    top_2 = torch.topk(probs, dim=-1, k=2)[0]
    return (top_2[..., 0] - top_2[..., 1]).squeeze()


def meta_confidence(
    logits: torch.Tensor = None,
    hidden_states: torch.Tensor = None,
    classifier: torch.nn.Linear = None,
):
    assert hidden_states is not None
    assert classifier is not None

    preds = classifier(hidden_states)
    probs = torch.softmax(preds, dim=-1)
    return probs[..., 1].squeeze()


def get_confidence_class(key):

    _conf_class_map = {
        "softmax": softmax_confidence,
        "meta": meta_confidence,
    }

    if key in _conf_class_map:
        return _conf_class_map[key]
    else:
        raise ValueError("Invalid confidence measure: {}".format(key))


def get_skip_mask(
    logits: torch.Tensor = None,
    hidden_states: torch.Tensor = None,
    classifier: torch.nn.Linear = None,
    config: AutoConfig = None,
    pos_time: int = 1,
    adapt_threshold: float = None,
    return_conf=False,
):
    assert config.shallow2deep_conf_type is not None

    if config.shallow2deep_conf_type is not None:
        key = config.shallow2deep_conf_type
        threshold = (
            config.shallow2deep_conf_threshold
            if adapt_threshold is None
            else adapt_threshold
        )

    conf_measure = get_confidence_class(key=key)
    conf = conf_measure(
        logits=logits,
        hidden_states=hidden_states,
        classifier=classifier,
    )
    mask = torch.where(conf <= threshold, 0.0, 1.0).bool()

    if not return_conf:
        return mask.item() 
    else:
        return mask.item(), conf.item()
