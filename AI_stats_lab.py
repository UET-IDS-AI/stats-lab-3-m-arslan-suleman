import numpy as np
import math

# =========================================================
# QUESTION 1 – Card Experiment
# =========================================================
def card_experiment():
    np.random.seed(42)
    
    # STEP 2: Theoretical probabilities
    P_A = 4 / 52                 # First card is Ace
    P_B_given_A = 3 / 51         # Second card Ace given first is Ace
    P_AB = P_A * P_B_given_A     # Intersection
    P_B = 4 / 52                 # Second card Ace (unconditional)
    
    # STEP 4: Simulation
    n_sim = 200_000
    deck = np.arange(52)  # Cards 0-51, first 4 are Aces
    ace_indices = set(range(4))
    
    first_aces = 0
    second_given_first_ace = 0
    
    for _ in range(n_sim):
        draw = np.random.choice(deck, size=2, replace=False)
        if draw[0] in ace_indices:
            first_aces += 1
            if draw[1] in ace_indices:
                second_given_first_ace += 1
    
    empirical_P_A = first_aces / n_sim
    empirical_P_B_given_A = second_given_first_ace / first_aces
    
    absolute_error = abs(empirical_P_B_given_A - P_B_given_A)
    
    return P_A, P_B, P_B_given_A, P_AB, empirical_P_A, empirical_P_B_given_A, absolute_error

# =========================================================
# QUESTION 2 – Bernoulli
# =========================================================
def bernoulli_lightbulb(p=0.05):
    np.random.seed(42)
    
    # Theoretical
    theoretical_P_X_1 = p
    theoretical_P_X_0 = 1 - p
    
    # Simulation
    n_sim = 100_000
    X = np.random.binomial(1, p, size=n_sim)
    empirical_P_X_1 = np.mean(X)
    
    absolute_error = abs(empirical_P_X_1 - theoretical_P_X_1)
    
    return theoretical_P_X_1, theoretical_P_X_0, empirical_P_X_1, absolute_error

# =========================================================
# QUESTION 3 – Binomial
# =========================================================
def binomial_bulbs(n=10, p=0.05):
    np.random.seed(42)
    
    # Theoretical probabilities
    def comb(n, k):
        return math.comb(n, k)
    
    P_0 = comb(n, 0) * p**0 * (1-p)**n
    P_2 = comb(n, 2) * p**2 * (1-p)**(n-2)
    P_ge_1 = 1 - P_0
    
    # Simulation
    n_sim = 100_000
    X = np.random.binomial(n, p, size=n_sim)
    empirical_P_ge_1 = np.mean(X >= 1)
    
    absolute_error = abs(empirical_P_ge_1 - P_ge_1)
    
    return P_0, P_2, P_ge_1, empirical_P_ge_1, absolute_error

# =========================================================
# QUESTION 4 – Geometric
# =========================================================
def geometric_die():
    np.random.seed(42)
    p = 1/6
    
    # Theoretical probabilities
    P_1 = p
    P_3 = (1-p)**2 * p
    P_gt_4 = (1-p)**4
    
    # Simulation
    n_sim = 200_000
    X = np.random.geometric(p, size=n_sim)
    empirical_P_gt_4 = np.mean(X > 4)
    
    absolute_error = abs(empirical_P_gt_4 - P_gt_4)
    
    return P_1, P_3, P_gt_4, empirical_P_gt_4, absolute_error

# =========================================================
# QUESTION 5 – Poisson
# =========================================================
def poisson_customers(lam=12):
    np.random.seed(42)
    
    # Theoretical probabilities
    P_0 = math.exp(-lam) * lam**0 / math.factorial(0)
    P_15 = math.exp(-lam) * lam**15 / math.factorial(15)
    
    # P(X >= 18) = 1 - sum_{k=0}^{17} P(X=k)
    P_cdf_17 = sum(math.exp(-lam) * lam**k / math.factorial(k) for k in range(18))
    P_ge_18 = 1 - P_cdf_17
    
    # Simulation
    n_sim = 100_000
    X = np.random.poisson(lam, size=n_sim)
    empirical_P_ge_18 = np.mean(X >= 18)
    
    absolute_error = abs(empirical_P_ge_18 - P_ge_18)
    
    return P_0, P_15, P_ge_18, empirical_P_ge_18, absolute_error
