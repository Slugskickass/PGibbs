import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt


class GibbsData:
    def __init__(self, m, bic, thc, K, n):
        eff = int(np.ceil((m - bic) / thc)) + 1
        self.mu = np.empty((eff, K))
        self.lam = np.empty((eff, K))
        self.P = np.empty((eff, K))
        self.N = np.empty((n, K))

def updatemean(x, cl, m, p):
    """
    This function update the mean of the current data
    :param x:
    :param cl:
    :param m:
    :param p:
    :return:
    """
    n = np.size(x)
    xb = np.mean(x)
    ps = n * cl + p
    ms = (n * cl * xb + p * m) / ps
    mean_answer = ms + np.random.normal(0, 1) / np.sqrt(ps)
    array_sum = np.sum(mean_answer)
    if np.isnan(array_sum):
        print(mean_answer, 'sal')
        print(ms, 'ma')
        print(ps, 'ps')
        print(n, 'n')
        print(cl, 'cl')
        print(xb, 'xb')
        print(p, 'p')
        print(m, 'm')
    return mean_answer

def updatelambda(x, cm, a, b):
    ash = np.size(x) / 2 + a
    bs = b + sum((x - cm)**2) / 2
    salc = np.random.gamma(ash, 1 / bs)
    return salc

def updp(x, P, mu, la):
    K = np.size(P)
    n = np.size(x)
    salc = np.empty([n, K])

    for k in range(K):
        salc[:, k] = P[k] * norm.pdf(x, mu[k], 1 / np.sqrt(la[k]))

    mar = np.outer(np.sum(salc, axis=1), * np.ones([1, K]))
    salc = salc / mar

    return salc

def newaloc(pij):
    S = np.zeros(np.shape(pij))
    for index, item in enumerate(pij):
        S[index] = np.random.multinomial(1, item, 1)
    nj = np.sum(S, axis=0)
    while np.sum(nj < 1):
        for index, item in enumerate(pij):
            S[index] = np.random.multinomial(1, item, 1)
        nj = np.sum(S, axis=0)
    _, cal = np.where(S == 1)
    return cal, S, nj

def newP(nj, a):
    ash = nj + a
    sal = np.random.gamma(ash, 1)
    sal = sal / np.sum(sal)
    return sal

M = 10000  # Number of interations
bi = 0  # Burn in period (currently not used)
th = 1  # Number of times between storing the data
K = 3  # The number of distributions
pmean = [0, 0.1]  # The prior mean, unchanged
plas = [2, 0.6]   # The prior precision, unchanged
pps = [1 / 2, 1 / 2, 1 / 2]

x = np.genfromtxt('forMiguel.csv')  # Loads in the test data
np.random.shuffle(x)
n = np.size(x)
eff = np.ceil((M - bi) / th)

sal = GibbsData(M, bi, th, K, n)

nr = np.linalg.matrix_rank(pmean)
if nr == 1:
    pmean = np.ones((K, 2)) * pmean

nr = np.linalg.matrix_rank(plas)
if nr == 1:
    plas = np.ones((K, 2)) * plas

cm = np.linspace(max(x), min(x), K)  # Current mean, this holds the currect estimation of the mean for each distribution

cl = np.linspace(1, 10, K)

cP = np.ones((K, 1)) / K
cP = cP.flatten()
cS = np.random.multinomial(1, cP, n)

_, ca = np.where(cS == 1)

i = 1
j = 0
for i in range(M):
    for k in range(K):
        xk = x[np.where(ca == k)]
        cm[k] = updatemean(xk, cl[k], pmean[k, 0], pmean[k, 1])
        cl[k] = updatelambda(xk, cm[k], plas[k, 0], plas[k, 1])
    cpij = updp(x, cP, cm, cl)

    ca, cS, cnj = newaloc(cpij)
    cP = newP(cnj, pps)

    if i > bi:
        if not np.mod(i, th):
       #     sal.N[j] = sal.N + cS
            sal.P[j, :] = cP
            sal.lam[j, :] = cl
            sal.mu[j, :] = cm
            j = j + 1
    if not np.mod(j, 1000) and j > 1:
        print(int(np.floor(100*(j/M))))
        plt.plot(sal.mu[:M-(M-j)])
        plt.show()

plt.plot(sal.lam[:-2])
plt.show()