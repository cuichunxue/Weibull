"""
Weibull and DS-Weibull lifetime analysis library.
Implements MLE, confidence intervals (Delta + Logit), and Weibull probability plots.
"""

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import norm, chi2
from scipy.special import gamma
import warnings

warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────────────────────
# Helper: Johnson adjusted ranks for Weibull probability plot
# ──────────────────────────────────────────────────────────────

def johnson_ranks(n, failures_sorted):
    """Median ranks via Benard's approximation for complete data."""
    ranks = np.arange(1, len(failures_sorted) + 1)
    median_ranks = (ranks - 0.3) / (n + 0.4)
    return median_ranks


def bernard_median_ranks(i, n):
    """Benard's approximation: (i - 0.3) / (n + 0.4)"""
    return (i - 0.3) / (n + 0.4)


# ──────────────────────────────────────────────────────────────
# 2-Parameter Weibull
# ──────────────────────────────────────────────────────────────

class WeibullAnalysis:
    """
    2-parameter Weibull distribution: F(t) = 1 - exp(-(t/alpha)^beta)
    Uses location-scale parameterization internally: mu=ln(alpha), sigma=1/beta
    """

    def __init__(self):
        self.alpha = None   # scale (eta)
        self.beta = None    # shape
        self.mu = None      # ln(alpha)
        self.sigma = None   # 1/beta
        self.cov = None     # 2x2 covariance matrix in (mu, sigma) space
        self.log_likelihood = None
        self.aic = None
        self.bic = None
        self.n_failures = None
        self.n_censored = None
        self.converged = False

    # ── Log-likelihood ──────────────────────────────────────────

    @staticmethod
    def _negloglik(params, t_fail, t_cens):
        """Negative log-likelihood in (mu, sigma) parameterization."""
        mu, log_sigma = params
        sigma = np.exp(log_sigma)
        if sigma <= 0:
            return 1e15

        ll = 0.0
        if len(t_fail) > 0:
            z = (np.log(t_fail) - mu) / sigma
            ll += np.sum(-np.log(sigma) - np.log(t_fail) + z - np.exp(z))

        if len(t_cens) > 0:
            z_c = (np.log(t_cens) - mu) / sigma
            ll += np.sum(-np.exp(z_c))

        return -ll

    # ── MLE fit ─────────────────────────────────────────────────

    def fit(self, times, censored=None):
        """
        Fit Weibull distribution by MLE.

        Parameters
        ----------
        times    : array-like, failure/suspension times (> 0)
        censored : array-like of bool, True = right-censored (suspension)
                   None → all failures
        """
        times = np.asarray(times, dtype=float)
        if censored is None:
            censored = np.zeros(len(times), dtype=bool)
        else:
            censored = np.asarray(censored, dtype=bool)

        if np.any(times <= 0):
            raise ValueError("All times must be > 0")

        t_fail = times[~censored]
        t_cens = times[censored]

        if len(t_fail) < 2:
            raise ValueError("At least 2 failures are required for Weibull MLE")

        self.n_failures = len(t_fail)
        self.n_censored = len(t_cens)

        # Multiple starting points
        mu0 = np.mean(np.log(t_fail))
        sigma0 = np.std(np.log(t_fail)) if np.std(np.log(t_fail)) > 0 else 1.0

        best_nll = np.inf
        best_res = None
        for mu_init in [mu0, mu0 * 0.5, mu0 * 1.5]:
            for sigma_init in [sigma0, 0.5, 1.0, 2.0]:
                try:
                    res = minimize(
                        self._negloglik,
                        x0=[mu_init, np.log(max(sigma_init, 1e-6))],
                        args=(t_fail, t_cens),
                        method="L-BFGS-B",
                        options={"maxiter": 5000, "ftol": 1e-12},
                    )
                    if res.success and res.fun < best_nll:
                        best_nll = res.fun
                        best_res = res
                except Exception:
                    pass

        if best_res is None:
            raise RuntimeError("Weibull MLE optimization failed")

        mu_hat, log_sigma_hat = best_res.x
        sigma_hat = np.exp(log_sigma_hat)

        self.mu = mu_hat
        self.sigma = sigma_hat
        self.alpha = np.exp(mu_hat)
        self.beta = 1.0 / sigma_hat
        self.log_likelihood = -best_nll
        self.converged = best_res.success

        n = self.n_failures + self.n_censored
        self.aic = 2 * 2 - 2 * self.log_likelihood
        self.bic = 2 * np.log(n) - 2 * self.log_likelihood

        # Covariance via numerical Hessian in (mu, log_sigma) space,
        # then delta-method to (mu, sigma)
        self.cov = self._compute_cov(best_res.x, t_fail, t_cens)
        self._converged = best_res.success

    def _compute_cov(self, params, t_fail, t_cens):
        """Numerical Hessian → inverse → covariance in (mu, sigma) space."""
        h = 1e-5
        n_params = len(params)
        H = np.zeros((n_params, n_params))
        f0 = self._negloglik(params, t_fail, t_cens)

        for i in range(n_params):
            for j in range(i, n_params):
                p_ij = params.copy(); p_ij[i] += h; p_ij[j] += h
                p_i  = params.copy(); p_i[i]  += h
                p_j  = params.copy(); p_j[j]  += h
                H[i, j] = (
                    self._negloglik(p_ij, t_fail, t_cens)
                    - self._negloglik(p_i, t_fail, t_cens)
                    - self._negloglik(p_j, t_fail, t_cens)
                    + f0
                ) / h**2
                H[j, i] = H[i, j]

        try:
            cov_log = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            return np.diag([np.nan, np.nan])

        # Delta method: d(sigma)/d(log_sigma) = sigma
        # So Var(sigma) = sigma^2 * Var(log_sigma)
        # Cov(mu, sigma) = sigma * Cov(mu, log_sigma)
        sig = np.exp(params[1])
        J = np.array([[1, 0], [0, sig]])  # Jacobian d(mu,sigma)/d(mu,log_sigma)
        cov_ms = J @ cov_log @ J.T
        return cov_ms

    # ── Distribution functions ───────────────────────────────────

    def cdf(self, t):
        t = np.asarray(t, dtype=float)
        return 1.0 - np.exp(-((t / self.alpha) ** self.beta))

    def sf(self, t):
        t = np.asarray(t, dtype=float)
        return np.exp(-((t / self.alpha) ** self.beta))

    def pdf(self, t):
        t = np.asarray(t, dtype=float)
        return (self.beta / self.alpha) * (t / self.alpha) ** (self.beta - 1) * self.sf(t)

    def hf(self, t):
        t = np.asarray(t, dtype=float)
        return self.pdf(t) / self.sf(t)

    def quantile(self, p):
        """t such that F(t) = p (B-life)."""
        return self.alpha * (-np.log(1 - p)) ** (1.0 / self.beta)

    def mean(self):
        return self.alpha * gamma(1 + 1.0 / self.beta)

    # ── Confidence intervals (Delta + Logit) ─────────────────────

    def cdf_ci(self, t, cl=0.9):
        """
        Two-sided confidence interval for F(t) using Delta method + Logit transform.
        Returns (lower, upper).
        """
        if self.cov is None or np.any(np.isnan(self.cov)):
            return np.full_like(t, np.nan), np.full_like(t, np.nan)
        t = np.asarray(t, dtype=float)
        z_ci = norm.ppf(0.5 + cl / 2)
        z = (np.log(t) - self.mu) / self.sigma

        # Gradient of logit(F) w.r.t. (mu, sigma)
        # F = 1 - exp(-exp(z)), logit(F) = log(F/(1-F)) = log(1 - exp(-exp(z))) - log(exp(-exp(z)))
        # But simpler: Var(log(F/(1-F))) via gradient of log(-log(S)) = z
        # Use: log(-log(S)) = z = (log(t)-mu)/sigma
        # Gradient: d/dmu = -1/sigma, d/dsigma = -z/sigma
        # Then Var(log(-log(S))) = (1/sigma^2)*[Var(mu) + z^2*Var(sigma) + 2z*Cov(mu,sigma)]
        # Then F_bounds via logit: using logit(F) = log(-log(1-F)) = log(-log(S)) ... same as log(-log(S))
        # Use Logit: logit(F) = log(F/(1-F))
        # Delta method on w = log(-log(1-F)) = log(exp(z)) = z
        # No wait... w = log(-log(SF)) and F = 1-exp(-exp(z))
        # Actually w = log(-log(SF)) = log(exp(z)) = z, but this uses the Gumbel transform
        # Better: use logit(F)

        F = self.cdf(t)
        F = np.clip(F, 1e-15, 1 - 1e-15)
        logit_F = np.log(F / (1 - F))

        # Gradient of logit(F) w.r.t. (mu, sigma)
        # F = 1 - exp(-exp(z)), z = (log(t)-mu)/sigma
        # dF/dz = exp(z-exp(z))
        # dz/dmu = -1/sigma, dz/dsigma = -z/sigma
        dF_dz = np.exp(z - np.exp(z))
        dlogitF_dF = 1.0 / (F * (1 - F))
        dlogitF_dmu    = dlogitF_dF * dF_dz * (-1.0 / self.sigma)
        dlogitF_dsigma = dlogitF_dF * dF_dz * (-z / self.sigma)

        grad = np.column_stack([dlogitF_dmu, dlogitF_dsigma])  # (n, 2)
        var_logit = np.einsum("ni,ij,nj->n", grad, self.cov, grad)
        se_logit = np.sqrt(np.maximum(var_logit, 0))

        lo_logit = logit_F - z_ci * se_logit
        hi_logit = logit_F + z_ci * se_logit

        lo = 1.0 / (1 + np.exp(-lo_logit))
        hi = 1.0 / (1 + np.exp(-hi_logit))
        return lo, hi

    def quantile_ci(self, p, cl=0.9):
        """Confidence interval for B-life t_p using Delta method + log transform."""
        if self.cov is None or np.any(np.isnan(self.cov)):
            return np.nan, np.nan
        p = float(p)
        z_ci = norm.ppf(0.5 + cl / 2)
        t_p = self.quantile(p)
        log_tp = np.log(t_p)

        # log(t_p) = mu + sigma * log(-log(1-p))
        c = np.log(-np.log(1 - p))
        # gradient of log(t_p) w.r.t. (mu, sigma)
        grad = np.array([1.0, c])
        var_log_tp = grad @ self.cov @ grad
        se_log_tp = np.sqrt(max(var_log_tp, 0))

        lo = np.exp(log_tp - z_ci * se_log_tp)
        hi = np.exp(log_tp + z_ci * se_log_tp)
        return lo, hi

    def parameter_ci(self, cl=0.9):
        """CI for alpha (scale) and beta (shape)."""
        if self.cov is None or np.any(np.isnan(self.cov)):
            return (np.nan, np.nan), (np.nan, np.nan)
        z = norm.ppf(0.5 + cl / 2)

        se_mu = np.sqrt(max(self.cov[0, 0], 0))
        alpha_lo = np.exp(self.mu - z * se_mu)
        alpha_hi = np.exp(self.mu + z * se_mu)

        # sigma = 1/beta  → log(sigma) ~ Normal
        se_sigma = np.sqrt(max(self.cov[1, 1], 0))
        sigma_lo = self.sigma * np.exp(-z * se_sigma / self.sigma)
        sigma_hi = self.sigma * np.exp(+z * se_sigma / self.sigma)
        beta_lo = 1.0 / sigma_hi
        beta_hi = 1.0 / sigma_lo
        return (alpha_lo, alpha_hi), (beta_lo, beta_hi)


# ──────────────────────────────────────────────────────────────
# DS-Weibull (Defective Subpopulation Weibull)
# ──────────────────────────────────────────────────────────────

class DSWeibullAnalysis:
    """
    DS-Weibull: F(t) = DS * (1 - exp(-(t/alpha)^beta))
    DS ∈ (0, 1]: fraction of defective/susceptible units.
    Uses parameterization: theta = (mu, log_sigma, logit_DS)
    mu = ln(alpha), sigma = 1/beta, DS = logistic(logit_DS)
    """

    def __init__(self):
        self.alpha = None
        self.beta = None
        self.DS = None
        self.mu = None
        self.sigma = None
        self.cov = None  # 3x3 in (mu, sigma, DS) space
        self.log_likelihood = None
        self.aic = None
        self.bic = None
        self.n_failures = None
        self.n_censored = None
        self.converged = False

    # ── Log-likelihood ──────────────────────────────────────────

    @staticmethod
    def _negloglik(params, t_fail, t_cens):
        mu, log_sigma, logit_DS = params
        sigma = np.exp(log_sigma)
        DS = 1.0 / (1 + np.exp(-logit_DS))  # sigmoid

        if sigma <= 0 or DS <= 0 or DS > 1:
            return 1e15

        ll = 0.0

        if len(t_fail) > 0:
            z = (np.log(t_fail) - mu) / sigma
            w = np.exp(z)             # exp(z)
            ew = np.exp(-w)           # exp(-exp(z)) = S_weibull
            # pdf_DS = DS * (1/sigma/t) * w * exp(-w)
            # but also need to handle DS < 1 (there's a mass at ∞ for non-defective)
            # log f(t) = log(DS) + log(w/sigma/t) + log(exp(-w))
            #          = log(DS) - log(sigma) - log(t) + z - w
            log_f = np.log(DS) - np.log(sigma) - np.log(t_fail) + z - w
            ll += np.sum(log_f)

        if len(t_cens) > 0:
            z_c = (np.log(t_cens) - mu) / sigma
            w_c = np.exp(z_c)
            # SF_DS(t) = 1 - F_DS(t) = 1 - DS*(1-exp(-w)) = (1-DS) + DS*exp(-w)
            sf_c = (1 - DS) + DS * np.exp(-w_c)
            sf_c = np.clip(sf_c, 1e-300, 1.0)
            ll += np.sum(np.log(sf_c))

        return -ll

    # ── MLE fit ─────────────────────────────────────────────────

    def fit(self, times, censored=None, DS_init=None):
        """
        Fit DS-Weibull by MLE.

        Parameters
        ----------
        times    : array-like
        censored : array-like of bool (True = right-censored)
        DS_init  : initial guess for DS fraction (default: fraction of failures)
        """
        times = np.asarray(times, dtype=float)
        if censored is None:
            censored = np.zeros(len(times), dtype=bool)
        else:
            censored = np.asarray(censored, dtype=bool)

        if np.any(times <= 0):
            raise ValueError("All times must be > 0")

        t_fail = times[~censored]
        t_cens = times[censored]

        if len(t_fail) < 2:
            raise ValueError("At least 2 failures are required for DS-Weibull MLE")

        self.n_failures = len(t_fail)
        self.n_censored = len(t_cens)

        n = len(times)
        mu0 = np.mean(np.log(t_fail))
        sigma0 = max(np.std(np.log(t_fail)), 0.5)

        if DS_init is None:
            DS_init = max(0.05, min(0.95, len(t_fail) / n))
        DS_init = float(DS_init)
        logit_DS0 = np.log(DS_init / (1 - DS_init))

        best_nll = np.inf
        best_res = None
        for mu_i in [mu0, mu0 * 0.7, mu0 * 1.3]:
            for s_i in [sigma0, 0.5, 1.0]:
                for ds_i in [logit_DS0, 0.0, 2.0, -1.0]:
                    try:
                        res = minimize(
                            self._negloglik,
                            x0=[mu_i, np.log(max(s_i, 1e-6)), ds_i],
                            args=(t_fail, t_cens),
                            method="L-BFGS-B",
                            options={"maxiter": 10000, "ftol": 1e-13},
                        )
                        if res.fun < best_nll:
                            best_nll = res.fun
                            best_res = res
                    except Exception:
                        pass

        if best_res is None:
            raise RuntimeError("DS-Weibull MLE optimization failed")

        mu_hat, log_sigma_hat, logit_DS_hat = best_res.x
        self.mu = mu_hat
        self.sigma = np.exp(log_sigma_hat)
        self.DS = 1.0 / (1 + np.exp(-logit_DS_hat))
        self.alpha = np.exp(mu_hat)
        self.beta = 1.0 / self.sigma
        self.log_likelihood = -best_nll
        self.converged = best_res.success

        n_total = self.n_failures + self.n_censored
        self.aic = 2 * 3 - 2 * self.log_likelihood
        self.bic = 3 * np.log(n_total) - 2 * self.log_likelihood

        self.cov = self._compute_cov(best_res.x, t_fail, t_cens)

    def _compute_cov(self, params, t_fail, t_cens):
        """Numerical Hessian → inverse → covariance in (mu, sigma, DS) space."""
        h = 1e-5
        n_p = len(params)
        H = np.zeros((n_p, n_p))
        f0 = self._negloglik(params, t_fail, t_cens)

        for i in range(n_p):
            for j in range(i, n_p):
                p_ij = params.copy(); p_ij[i] += h; p_ij[j] += h
                p_i  = params.copy(); p_i[i]  += h
                p_j  = params.copy(); p_j[j]  += h
                H[i, j] = (
                    self._negloglik(p_ij, t_fail, t_cens)
                    - self._negloglik(p_i, t_fail, t_cens)
                    - self._negloglik(p_j, t_fail, t_cens)
                    + f0
                ) / h**2
                H[j, i] = H[i, j]

        try:
            cov_nat = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            return np.diag([np.nan] * n_p)

        # Transform from (mu, log_sigma, logit_DS) to (mu, sigma, DS)
        mu_h, ls_h, lds_h = params
        sig = np.exp(ls_h)
        ds = 1.0 / (1 + np.exp(-lds_h))
        J = np.array([
            [1, 0,            0],
            [0, sig,          0],
            [0, 0,  ds * (1 - ds)],
        ])
        return J @ cov_nat @ J.T

    # ── Distribution functions ───────────────────────────────────

    def cdf(self, t):
        t = np.asarray(t, dtype=float)
        return self.DS * (1.0 - np.exp(-((t / self.alpha) ** self.beta)))

    def sf(self, t):
        return 1.0 - self.cdf(t)

    def pdf(self, t):
        t = np.asarray(t, dtype=float)
        z = t / self.alpha
        return self.DS * (self.beta / self.alpha) * z ** (self.beta - 1) * np.exp(-(z ** self.beta))

    def hf(self, t):
        p = self.pdf(t)
        s = self.sf(t)
        return np.where(s > 0, p / s, 0.0)

    def quantile(self, p):
        """t such that F(t) = p. Only valid for p < DS."""
        p = float(p)
        if p >= self.DS:
            return np.inf
        return self.alpha * (-np.log(1 - p / self.DS)) ** (1.0 / self.beta)

    # ── Confidence intervals ─────────────────────────────────────

    def cdf_ci(self, t, cl=0.9):
        """CI for F(t) using Delta method + Logit."""
        if self.cov is None or np.any(np.isnan(self.cov)):
            return np.full_like(t, np.nan), np.full_like(t, np.nan)
        t = np.asarray(t, dtype=float)
        z_ci = norm.ppf(0.5 + cl / 2)
        z = (np.log(t) - self.mu) / self.sigma
        w = np.exp(z)

        F = self.cdf(t)
        F = np.clip(F, 1e-15, 1 - 1e-15)
        logit_F = np.log(F / (1 - F))

        # F = DS*(1-exp(-w))
        # dF/dmu    = DS * exp(-w) * w * (-1/sigma) [since dw/dmu = -w/sigma? Actually dz/dmu=-1/sigma, dw/dz=w]
        #           = DS * w * exp(-w) * (-1/sigma)
        # dF/dsigma = DS * w * exp(-w) * (-z/sigma)
        # dF/dDS    = 1 - exp(-w)
        dF_dmu    = self.DS * w * np.exp(-w) * (-1.0 / self.sigma)
        dF_dsigma = self.DS * w * np.exp(-w) * (-z / self.sigma)
        dF_dDS    = 1.0 - np.exp(-w)

        dlogitF_dF = 1.0 / (F * (1 - F))
        g_mu    = dlogitF_dF * dF_dmu
        g_sigma = dlogitF_dF * dF_dsigma
        g_DS    = dlogitF_dF * dF_dDS

        grad = np.column_stack([g_mu, g_sigma, g_DS])
        var_logit = np.einsum("ni,ij,nj->n", grad, self.cov, grad)
        se_logit = np.sqrt(np.maximum(var_logit, 0))

        lo = 1.0 / (1 + np.exp(-(logit_F - z_ci * se_logit)))
        hi = 1.0 / (1 + np.exp(-(logit_F + z_ci * se_logit)))
        return lo, hi

    def quantile_ci(self, p, cl=0.9):
        """CI for t_p (B-life) using Delta method + log transform."""
        if self.cov is None or np.any(np.isnan(self.cov)):
            return np.nan, np.nan
        p = float(p)
        if p >= self.DS:
            return np.inf, np.inf
        z_ci = norm.ppf(0.5 + cl / 2)
        t_p = self.quantile(p)
        log_tp = np.log(t_p)

        # log(t_p) = mu + sigma * log(-log(1 - p/DS))
        c = np.log(-np.log(1 - p / self.DS))
        # d log(t_p) / d mu = 1
        # d log(t_p) / d sigma = c
        # d log(t_p) / d DS = sigma / (DS * (1 - p/DS) * log(1-p/DS)) -- chain rule
        r = p / self.DS
        dlog_tp_dDS = self.sigma / (self.DS * (1 - r) * np.log(1 - r))

        grad = np.array([1.0, c, dlog_tp_dDS])
        var_log_tp = grad @ self.cov @ grad
        se_log_tp = np.sqrt(max(var_log_tp, 0))

        lo = np.exp(log_tp - z_ci * se_log_tp)
        hi = np.exp(log_tp + z_ci * se_log_tp)
        return lo, hi

    def parameter_ci(self, cl=0.9):
        """CI for alpha, beta, DS."""
        if self.cov is None or np.any(np.isnan(self.cov)):
            return (np.nan, np.nan), (np.nan, np.nan), (np.nan, np.nan)
        z = norm.ppf(0.5 + cl / 2)

        se_mu = np.sqrt(max(self.cov[0, 0], 0))
        alpha_lo = np.exp(self.mu - z * se_mu)
        alpha_hi = np.exp(self.mu + z * se_mu)

        se_sigma = np.sqrt(max(self.cov[1, 1], 0))
        sigma_lo = self.sigma * np.exp(-z * se_sigma / self.sigma)
        sigma_hi = self.sigma * np.exp(+z * se_sigma / self.sigma)
        beta_lo = 1.0 / sigma_hi
        beta_hi = 1.0 / sigma_lo

        se_DS = np.sqrt(max(self.cov[2, 2], 0))
        DS_lo = max(0.0, self.DS - z * se_DS)
        DS_hi = min(1.0, self.DS + z * se_DS)
        return (alpha_lo, alpha_hi), (beta_lo, beta_hi), (DS_lo, DS_hi)


# ──────────────────────────────────────────────────────────────
# Weibull Probability Plot data
# ──────────────────────────────────────────────────────────────

def weibull_plot_data(times, censored=None):
    """
    Prepare data for Weibull probability plot.
    Returns (t_sorted, median_ranks, x_plot, y_plot) where
      x_plot = log(t), y_plot = log(-log(1-F)) (Gumbel / Weibull scale)
    """
    times = np.asarray(times, dtype=float)
    if censored is None:
        censored = np.zeros(len(times), dtype=bool)
    else:
        censored = np.asarray(censored, dtype=bool)

    n = len(times)
    t_fail = np.sort(times[~censored])
    nf = len(t_fail)

    if nf == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])

    ranks = np.arange(1, nf + 1, dtype=float)
    mr = (ranks - 0.3) / (n + 0.4)   # Benard's approximation

    # Weibull paper coordinates
    x = np.log(t_fail)
    y = np.log(-np.log(1 - mr))

    return t_fail, mr, x, y


def parse_input_data(text):
    """
    Parse pasted text data. Accepts:
      - Single column: failure times (all failures)
      - Two columns: time, censored_flag (0=failure, 1=suspended)
    Separators: comma, tab, whitespace.
    Returns (times array, censored bool array)
    """
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    # Skip header line if non-numeric first token
    if lines:
        first = lines[0].replace(",", " ").split()[0]
        try:
            float(first)
        except ValueError:
            lines = lines[1:]

    times = []
    censored = []
    for line in lines:
        parts = line.replace(",", " ").replace("\t", " ").split()
        if not parts:
            continue
        try:
            t = float(parts[0])
            c = bool(int(parts[1])) if len(parts) >= 2 else False
            times.append(t)
            censored.append(c)
        except (ValueError, IndexError):
            continue

    return np.array(times, dtype=float), np.array(censored, dtype=bool)
