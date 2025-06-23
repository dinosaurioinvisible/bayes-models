
import scipy.stats as sts
import numpy as np
import matplotlib.pyplot as plt

# from web, just to test the basic fxs

'''
P(hypothesis|data) = P(data|hypothesis) * P(hypothesis) / P(data)
P(H|D) = P(D|H) * P(H) / P(D)

P(H) = prior -> bias/expectations (initial belief)
P(D) = evidence -> experience = marginal likelihood (prob of data, avged across all hypotheses)
P(D|H) = likelihood -> probability of data given specific hypothesis=theta
P(H|D) = posterior > updated belief
'''

# prior
# hypothesis ~ Normal_dist(mean, sd)

mu = np.linspace(1.6, 2.0)                              # default num=50, endpoint=True
du = np.ones(mu.size)/mu.size                           # uniform dist
db = sts.beta.pdf(mu, 2, 5, loc = 1.6, scale = 0.4)     # beta dist
db = db/db.sum()

plt.plot(mu, db, label = 'Beta Dist = H')
plt.plot(mu, du, label = 'Uniform Dist')
# plt.xlabel("$\mu$")
# plt.ylabel("Probability density")
# plt.legend()
# plt.show()

x = 1.75
likelihood = sts.norm.pdf(x, mu, scale=0.1)
likelihood = likelihood/likelihood.sum()

plt.plot(mu, likelihood, label = 'likelihood of $\mu$ = P(D|H)')
# plt.xlabel("$\mu$")
# plt.ylabel("Probability Density/Likelihood")
# plt.legend()
# plt.show()

# let's say the evidence is:

data = np.random.randint(160,200,50)/100
evidence = sts.norm.pdf(data, mu, scale=0.1)
evidence /= evidence.sum()

plt.plot(mu, evidence, label = 'evidence = P(D)', linestyle='--', color='grey')

# so the updated belief

# prior = db
posterior_db = likelihood * db / evidence
posterior_db /= posterior_db.sum()

# prior = du
posterior_du = likelihood * du / evidence
posterior_du /= posterior_du.sum()

plt.plot(mu, posterior_db, label = 'posterior (beta)')
plt.plot(mu, posterior_du, label = 'posterior (uniform)')

# plt.title(f'Likelihood of $\mu$ given observation {x}m')
plt.xlabel("$\mu$")
plt.ylabel("Probability Density/Likelihood")
plt.legend()
plt.show()