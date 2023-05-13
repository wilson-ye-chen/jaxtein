import numpy as np
import jax.numpy as jnp
from jax import jit, jacfwd, jacrev
from jax.scipy.optimize import minimize
from mcmclib.metropolis import mala_adapt

from jaxtein.kernel import k_kgm
from jaxtein.stein import SteinKernel
from jaxtein.impsamp import PiDistribution
from jaxtein.impsamp import SteinImportanceSampling

class SisEmpStudy:
    def __init__(self, logp, dlogp, d2logp, x0):
        self.logp = logp
        self.dlogp = dlogp
        self.d2logp = d2logp
        self.x0 = x0
        self.n_dim = len(x0)
        self.n_eph = 8
        self.n_mala = 100_000
        self.m_mala = [10, 30, 50, 100, 300, 500, 1000]
        self.n_rep = 10
        self.map = None
        self.cinv = None
        self.mala_p = None
        self.mala_kgm3 = None
        self.mala_was1 = None
        self.idx0 = None
        self.ksd_p = None
        self.ksd_kgm3 = None
        self.ksd_was1 = None

    def run(self):
        # Find the MAP
        fun = lambda x: -self.logp(x)
        sol = minimize(fun, self.x0, method='BFGS')
        self.map = sol.x

        # Negative Hessian at MAP
        self.cinv = -self.d2logp(sol.x)

        # Pi-KGM3
        k = lambda x, y: k_kgm(x, y, 3., self.map, self.cinv)
        k0 = SteinKernel(self.logp, self.dlogp, k)
        pi_kgm3 = PiDistribution(self.logp, self.dlogp, self.d2logp, k0)

        # Pi-WAS1
        logpi_was1 = lambda x: self.n_dim / (self.n_dim + 1) * self.logp(x)
        dlogpi_was1 = jit(jacrev(logpi_was1))

        # MALA
        a = self.n_eph * [0.4]
        e = (self.n_eph - 1) * [1000] + [self.n_mala]
        c0 = jnp.eye(self.n_dim)
        self.mala_p = mala_adapt(
            self.logp, self.dlogp, self.map, 1., c0, a, e)[2][-1]
        self.mala_kgm3 = mala_adapt(
            pi_kgm3.logpdf, pi_kgm3.dlogpdf, self.map, 1., c0, a, e)[2][-1]
        self.mala_was1 = mala_adapt(
            logpi_was1, dlogpi_was1, self.map, 1., c0, a, e)[2][-1]

        # SIS-KGM3
        sis = SteinImportanceSampling(k0)

        # Allocate space for results
        n_sub = len(self.m_mala)
        self.ksd_p = np.empty((self.n_rep, n_sub))
        self.ksd_kgm3 = np.empty((self.n_rep, n_sub))
        self.ksd_was1 = np.empty((self.n_rep, n_sub))

        # Random starting indices
        last = self.n_mala - self.m_mala[-1] + 1
        self.idx0 = np.random.randint(0, last, size=self.n_rep)

        # KSD
        for i in range(self.n_rep):
            for j in range(n_sub):
                # Sub-chain
                iStr = self.idx0[i]
                iEnd = iStr + self.m_mala[j]
                # P
                w, _, k0f = sis.solve(self.mala_p[iStr:iEnd])
                self.ksd_p[i, j] = np.sqrt(np.dot(np.dot(w, k0f), w))
                # KGM3
                w, _, k0f = sis.solve(self.mala_kgm3[iStr:iEnd])
                self.ksd_kgm3[i, j] = np.sqrt(np.dot(np.dot(w, k0f), w))
                # WAS1
                w, _, k0f = sis.solve(self.mala_was1[iStr:iEnd])
                self.ksd_was1[i, j] = np.sqrt(np.dot(np.dot(w, k0f), w))

            print(f'Rep {i} done.')
