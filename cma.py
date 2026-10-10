"""Continuous joint-placement CMA sampler and exact-sample adaptation."""
from dataclasses import dataclass
import math
import numpy as np
from ga import Individual

def _reflect_unit(values):
    return 1.0 - np.abs(np.mod(values, 2.0) - 1.0)

@dataclass
class JointCMA:
    mean: np.ndarray
    covariance: np.ndarray
    path_sigma: np.ndarray
    path_c: np.ndarray
    sigma: float = .08
    updates: int = 0

    @classmethod
    def from_genome(cls, genome, lower, span):
        mean = ((genome-lower)/span).ravel().copy()
        d = len(mean)
        return cls(mean, np.eye(d), np.zeros(d), np.zeros(d))

    def sample(self, count, lower, span, num_uavs, rng):
        vals, vecs = np.linalg.eigh(self.covariance)
        factor = vecs*np.sqrt(np.maximum(vals, 1e-12))
        samples = self.mean + self.sigma*(rng.standard_normal((count, len(self.mean))) @ factor.T)
        return [Individual(lower + _reflect_unit(row).reshape(num_uavs, 4)*span,
                           source=f"cma:{self.updates}") for row in samples]

    def update(self, verified, quality_key, lower, span, mu=2):
        # Every entry must be an actual MILP label from this distribution.
        if len(verified) < 2 or any(x.exact_result is None for x in verified):
            return False
        selected = sorted(verified, key=quality_key)[:min(mu, len(verified))]
        d = len(self.mean)
        weights = np.log(len(selected)+.5) - np.log(np.arange(1, len(selected)+1))
        weights /= weights.sum()
        mueff = 1. / (weights @ weights)
        cc = (4.+mueff/d)/(d+4.+2.*mueff/d)
        cs = (mueff+2.)/(d+mueff+5.)
        c1 = 2./((d+1.3)**2+mueff)
        cmu = min(1.-c1, 2.*(mueff-2.+1./mueff)/((d+2.)**2+mueff))
        damping = 1.+2.*max(0., math.sqrt((mueff-1.)/(d+1.))-1.)+cs
        positions = np.array([((x.evaluated_genome-lower)/span).ravel() for x in selected])
        steps = (positions-self.mean)/self.sigma
        yw = weights @ steps
        vals, vecs = np.linalg.eigh(self.covariance)
        invsqrt = (vecs / np.sqrt(np.maximum(vals, 1e-12))) @ vecs.T
        self.path_sigma = (1.-cs)*self.path_sigma + math.sqrt(cs*(2.-cs)*mueff)*(invsqrt @ yw)
        self.path_c = (1.-cc)*self.path_c + math.sqrt(cc*(2.-cc)*mueff)*yw
        self.mean = weights @ positions
        self.covariance = ((1.-c1-cmu)*self.covariance
                           + c1*np.outer(self.path_c, self.path_c)
                           + cmu*(steps.T*weights) @ steps + 1e-12*np.eye(d))
        self.covariance = (self.covariance+self.covariance.T)*.5
        chi = math.sqrt(d)*(1.-1./(4*d)+1./(21*d*d))
        self.sigma = float(np.clip(self.sigma*math.exp(np.clip(
            cs/damping*(np.linalg.norm(self.path_sigma)/chi-1.), -1., 1.)), .005, .35))
        self.updates += 1
        return True
