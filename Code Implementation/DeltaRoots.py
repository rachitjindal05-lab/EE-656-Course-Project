import numpy as np
from itertools import combinations
from math import prod
from numpy import polyval


class DeltaRoots:

    def __init__(self, S: np.ndarray, delta: np.ndarray):
        self.S = S
        self.delta = delta.real
        self.roots, self.eta = self._roots()

    def _finalCoeff(self):
        d = self.delta
        S = self.S
        M = len(S)
        coef = np.zeros((M, M))
        for i in range(M):
            denominator = 1
            for j in range(M):
                if i != j:
                    denominator *= d[i] - d[j]
            d_new = np.delete(d, i)
            for b in range(M):
                coef[i][b] = (
                    ((-1) ** b)
                    * sum(prod(comb) for comb in combinations(d_new, b))
                    * S[i]
                    / denominator
                )
        final_coeff = []
        for q in range(M):
            final_coeff.append(sum(coef[:, q]))
        return final_coeff

    def _roots(self):
        """
        Finds the real roots of the Lagrange equation and also gives eta
        to be used in the next iteration
        """
        final_coeff = self._finalCoeff()
        new_coeff = np.array(final_coeff)
        new_coeff[-1] -= 1
        roots = np.real(np.roots(new_coeff))
        eta = roots[0]
        St_max = float(polyval(final_coeff, eta))
        for root in roots:
            St_root = float(polyval(final_coeff, root))
            if St_root > St_max:
                eta = root
                St_max = St_root
        return roots, eta
