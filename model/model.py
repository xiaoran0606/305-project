# %%
# Import libraries
import numpy as np
import pandas as pd
import torch
import torch.distributions as dist
from scipy.stats import gamma
from concurrent.futures import ProcessPoolExecutor
import functools
from sklearn.linear_model import Ridge
import pickle

# %% [markdown]
# ### Preprocessing

# %%
# Preparing weights
days = np.arange(1, 36) # people stay infectious for about 5 weeks
daily_weights = gamma.pdf(days, a=6.5, scale=0.62) # previous result
daily_weights /= daily_weights.sum()

L = 5 # choose one week as a time step
weekly_weights = np.array([daily_weights[i*7:(i+1)*7].sum() for i in range(L)])
weekly_weights /= weekly_weights.sum()

# Store weights
weights = torch.tensor(weekly_weights, dtype=torch.float32)

# %%
# Load data
freq = "seasonal" # TODO: change
df = pd.read_csv(f'../data/covid_{freq}.csv', parse_dates=['date'])
block_cov = pd.read_csv( f'../data/block_cov_{freq}.csv')
block_cov = block_cov.set_index(['country', 'time_block'])

# %%
print(df.columns.tolist())
print(block_cov.columns.tolist())

# %%
# Convert continents to indices
countries = df['country'].unique()
cont_map  = df.drop_duplicates('country').set_index('country')['continent'].to_dict()
cont_list = sorted(set(cont_map.values()))
cont_idx  = {c: i for i, c in enumerate(cont_list)}
n_cont    = len(cont_list)

# %% [markdown]
# ### Model set up
# Note: our model (when L=1) is the special case of the LDS introduced in class where (for each country c) $A=C=1, b_{s(t),c}=\beta_0 + \beta_1*PC_{1,c} + \beta_2*PC_{2,c} + \beta_3*V_{str_{s(t),c}} + \beta_4*V_{vac_{s(t),c}} + \theta_c, d_c=\log(pc)$. It's worthing noting that $b_{s(t),c}$ changes overtime and $d_c$ is country-dependent.
# 
# 
# More specifically, 
# 
# \begin{align*}
# &S_{t,c} = S_{t-1,c} + \mu_{t,c} + \epsilon,   \epsilon \sim N(0, \sigma^2)\\
# &Y_{t,c} = S_{t,c}   + \alpha_c  + \eta,   \eta \sim N(0, \tau^2)
# \end{align*}
# 
# where $\mu_{t,c}=\beta_0 + \beta_1*PC_{1,c} + \beta_2*PC_{2,c} + \beta_3*V_{str_{s(t),c}} + \beta_4*V_{vac_{s(t),c}} + \theta_c$, $\alpha_c=\log{(pc)}$
# 
# ### Kalman filter + smoothing
# For each country c, we apply Kalman filter and smoothing individually. 
# 
# Let 
# 
# - mu_p_t = $E[S_{t}|Y_{1:t-1}]$
# 
# - mu_f_t=$E[S_{t}|Y_{1:t}]$
# - mu_s_t=$E[S_{t}|Y_{1:T}]$
# - Sigma_p_t = $Var[S_{t}|Y_{1:t-1}]$
# - Sigma_f_t=$Var[S_{t}|Y_{1:t}]$
# - Sigma_s_t=$Var[S_{t}|Y_{1:T}]$.
# 
# 
# Per lecture note, we know that (using the names of variables)
# 
# #### Update step:
# - innov = Y_t - $\log{pc}$ - mu_p_t (innovation)
# - H_t = Sigma_p_t + tau2 (innovation covariance)
# - K_t = Sigma_p_t / H_t (Kalman gain)
# 
# 
# - mu_f_t = mu_p_t + K_t * innov
# - Sigma_f_t = (1 - K_t) * Sigma_p_t
# 
# #### Predict step:
# - mu_p_t+1 = mu_f_t + \mu_{t,c}
# 
# 
# - Sigma_p_t+1 = Sigma_f_t + sigma2
# 
# #### Smoother:
# - G_t = Sigma_f_t / Sigma_p_t+1
# 
# 
# - mu_s_t = mu_f_t + G_t * (mu_s_t+1 - mu_p_t+1)
# - Sigma_s_t = Sigma_f_t + G_t^2 * (Sigma_s_t+1 - Sigma_p_t+1)
# 
# #### Marginal:
# $\log{p(Y_{1:T})}=\sum_{t=1}^T\log{p(Y_t|Y_{1:t-1})}$, where $p(Y_t|Y_{1:t-1})\sim N(Y_t | \mu_{p_t}+\log{p(c)},H_t)$
# 
# 
# 

# %%
# helper functions
def get_block_covariates(country, time_block):
    key = (country, time_block)
    if key in block_cov.index:
        return (float(block_cov.loc[key, 'stringency_lag']),
                float(block_cov.loc[key, 'log_vac_lag']))
    return 0.0, 0.0


def compute_mu_t_c(df_c, beta, theta_c):
    U_c1 = float(df_c['pc1'].iloc[0])
    U_c2 = float(df_c['pc2'].iloc[0])
    mu   = np.zeros(len(df_c))
    for i, (_, row) in enumerate(df_c.iterrows()):
        V_str, V_vac = get_block_covariates(df_c['country'].iloc[0], row['time_block']) # constant within same season
        mu[i] = (beta[0] + beta[1]*U_c1 + beta[2]*U_c2
                 + beta[3]*V_str + beta[4]*V_vac + theta_c)
    return mu


def kalman_filter(Y, log_pc, mu_seq, tau2, mu_s_1, Sigma_s_1, sigma2 = 0.03):
    T       = len(Y)
    mu_f    = np.zeros(T)
    Sigma_f = np.zeros(T)
    mu_p    = np.zeros(T)
    Sigma_p = np.zeros(T)
    log_ml  = 0.0

    for t in range(T):
        # predict
        if t == 0:
            mu_p[t]    = mu_s_1
            Sigma_p[t] = Sigma_s_1
        else:
            mu_p[t]    = mu_f[t-1] + mu_seq[t]
            Sigma_p[t] = Sigma_f[t-1] + sigma2

        # update
        innov      = Y[t] - log_pc - mu_p[t]
        H_t        = Sigma_p[t] + tau2
        K_t        = Sigma_p[t] / H_t
        mu_f[t]    = mu_p[t]    + K_t * innov
        Sigma_f[t] = (1 - K_t) * Sigma_p[t]

        # log marginal
        log_ml += -0.5 * (np.log(2 * np.pi * H_t) + innov**2 / H_t)

    return mu_f, Sigma_f, mu_p, Sigma_p, log_ml

def rts_smoother(mu_f, Sigma_f, mu_p, Sigma_p):
    T             = len(mu_f)
    mu_s          = mu_f.copy()
    Sigma_s       = Sigma_f.copy()
    Sigma_s_cross = np.zeros(T - 1)

    for t in range(T - 2, -1, -1):
        G_t               = Sigma_f[t] / Sigma_p[t + 1]
        mu_s[t]           = mu_f[t]    + G_t * (mu_s[t+1]    - mu_p[t+1])
        Sigma_s[t]        = Sigma_f[t] + G_t**2 * (Sigma_s[t+1] - Sigma_p[t+1])
        Sigma_s_cross[t]  = G_t * Sigma_s[t+1]

    return mu_s, Sigma_s, Sigma_s_cross

# Input: deterministic parameters from initialization / M step
#   - alpha, beta, theta, tau2
# Output: posterior mean
def e_step(df, alpha, beta, theta, tau2, sigma2 = 0.03):
    # for each country c
    results = {}
    log_ml  = 0.0

    for country in countries:
        df_c = (df[df['country'] == country]
                .sort_values('date')
                .reset_index(drop=True))

        Y      = df_c['log_cases'].values.astype(float)
        log_pc = alpha[country]
        mu_seq = compute_mu_t_c(df_c, beta, theta[country])

        # initialise: mu_s_1 = Y - log_pc
        mu_s_1    = float(Y[0] - log_pc)
        Sigma_s_1 = sigma2

        # filter
        mu_f, Sigma_f, mu_p, Sigma_p, lml = kalman_filter(
            Y, log_pc, mu_seq, tau2, mu_s_1, Sigma_s_1)

        # smoother
        mu_s, Sigma_s, Sigma_s_cross = rts_smoother(
            mu_f, Sigma_f, mu_p, Sigma_p)

        log_ml += lml

        # sufficient statistics for M-step
        ES   = mu_s                                  # E[S_t | Y_{1:T}]
        ES2  = Sigma_s + mu_s**2                     # E[S_t^2 | Y_{1:T}]
        ESS  = Sigma_s_cross + mu_s[1:] * mu_s[:-1] # E[S_t S_{t-1} | Y_{1:T}]

        results[country] = {
            'ES'      : ES,
            'ES2'     : ES2,
            'ESS'     : ESS,
            'Y'       : Y,
            'mu_seq'  : mu_seq,
            'PC_c1'   : float(df_c['pc1'].iloc[0]),
            'PC_c2'   : float(df_c['pc2'].iloc[0]),
            'blocks'  : df_c['time_block'].tolist(),
            'dates'   : df_c['date'].tolist(),
        }

    return results, log_ml

def m_step(df, results, alpha, tau2, lam=1.0):
    
    # ── Step 1: ridge regression for beta, theta ──────────────────────────
    # Response:  E[Delta S_{t,c}] = E[S_{t,c}] - E[S_{t-1,c}]
    # Design:    [1, PC_{c1}, PC_{c2}, V_str_{s(t),c}, V_vac_{s(t),c},
    #             continent dummies]

    rows_y, rows_x = [], []

    for country in countries:
        res  = results[country]
        df_c = df[df['country'] == country].sort_values('date').reset_index(drop=True)
        T    = len(df_c)
        ES   = res['ES']
        PC_c1 = res['PC_c1']
        PC_c2 = res['PC_c2']
        cont  = cont_map[country]

        delta_S = ES[1:] - ES[:-1]   # E[Delta S_{t,c}], length T-1

        for t in range(T - 1):
            V_str, V_vac = get_block_covariates(country, df_c['time_block'].iloc[t+1])
            cont_dummy   = np.zeros(n_cont)
            cont_dummy[cont_idx[cont]] = 1.0

            rows_y.append(delta_S[t])
            rows_x.append(np.concatenate(
                [[1.0, PC_c1, PC_c2, V_str, V_vac], cont_dummy]))

    y_reg = np.array(rows_y)
    X_reg = np.array(rows_x)

    ridge     = Ridge(alpha=lam, fit_intercept=False)
    ridge.fit(X_reg, y_reg)
    coefs     = ridge.coef_

    beta_new  = coefs[:5]                          # [beta0..beta4]
    theta_raw = coefs[5:]
    theta_raw -= theta_raw.mean()                  # sum-to-zero identifiability
    theta_new  = {country: float(theta_raw[cont_idx[cont_map[country]]])
                  for country in countries}
    mu_new = ridge.predict(X_reg)

    # ── Step 2: update tau2 ───────────────────────────────────────────────
    # tau2 = (1/N) sum_{t,c} E[(Y_{t,c} - S_{t,c} - alpha_c)^2]
    #      = (Y - alpha_c)^2 - 2*(Y - alpha_c)*E[S_t] + E[S_t^2]

    tau2_num, N_tau = 0.0, 0

    for country in countries:
        res    = results[country]
        Y      = res['Y'];  ES = res['ES'];  ES2 = res['ES2']
        log_pc = alpha[country]

        Y_tilde   = Y - log_pc
        tau2_num += float(np.sum(Y_tilde**2 - 2*Y_tilde*ES + ES2))
        N_tau    += len(Y)

    tau2_new = max(tau2_num / N_tau, 1e-6)

    # ── Step 3: update alpha_c (MAP closed-form) ──────────────────────────
    # Prior: log P_c ~ N(-1, 1)  →  alpha_c = log P_c ~ N(-1, 1)
    # MAP:   alpha_c = [sum_t(Y_t - E[S_t]) / tau2 - 1] / [T_c/tau2 + 1]

    alpha_new = {}
    for country in countries:
        res    = results[country]
        Y      = res['Y'];  ES = res['ES']
        T_c    = len(Y)
        alpha_new[country] = float(
            (np.sum(Y - ES) / tau2_new - 1.0) / (T_c / tau2_new + 1.0)
        )

    return beta_new, theta_new, tau2_new, alpha_new

# EM
def run_EM(df, n_iter=1000, lam=1.0, tol=1e-4):

    beta   = np.zeros(5)
    theta  = {c: 0.0 for c in countries}
    tau2   = 0.01
    alpha  = {c: 0.0 for c in countries}

    log_mls = []

    for it in range(n_iter):
        results, log_ml = e_step(df, alpha, beta, theta, tau2)
        log_mls.append(log_ml)
        print(f"Iter {it+1:3d}  |  log_ml = {log_ml:10.2f}  "
              f"| tau2 = {tau2:.5f}")

        beta, theta, tau2, alpha = m_step(
            df, results, alpha, tau2, lam=lam)

        if it > 0 and abs(log_mls[-1] - log_mls[-2]) < tol:
            print(f"Converged at iteration {it+1}")
            break

    return results, beta, theta, tau2, alpha, log_mls

# %%
if __name__ == '__main__':
    sigma2 = 0.03
    results, beta, theta, tau2, alpha, log_mls = run_EM(
        df, n_iter=1000, lam=1.0)

    print("\n── Final parameters ──")
    print(f"beta0={beta[0]:.4f}  beta1={beta[1]:.4f}  beta2={beta[2]:.4f}  "
          f"beta3={beta[3]:.4f}  beta4={beta[4]:.4f}")
    print(f"tau2={tau2:.5f}")

    # Save full posterior (for later sampling)
    save_dict = {
        'results': results,
        'beta': beta,
        'theta': theta,
        'sigma2': sigma2,
        'tau2': tau2,
        'alpha': alpha
    }

    with open(f'../data/posterior_results_{freq}.pkl', 'wb') as f:
        pickle.dump(save_dict, f)

    print(f"Saved posterior_results_{freq}.pkl")

    # Also save simple plug-in R
    records = []
    for country in countries:
        ES    = results[country]['ES']
        dates = results[country]['dates']

        for t in range(1, len(ES)):
            records.append({
                'country': country,
                'date': dates[t],
                'R_plug_in': np.exp(ES[t] - ES[t-1])
            })

    pd.DataFrame(records).to_csv(
        f'../data/R_plugin_{freq}.csv',
        index=False
    )

    print(f"Saved R_plugin_{freq}.csv")


