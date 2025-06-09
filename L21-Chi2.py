import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2

# —————————————————————————————
# 1) Distribuzione χ² teorica
# —————————————————————————————
df = 5                                # grado di libertà
x = np.linspace(0, 20, 500)          # intervallo di valori
pdf = chi2.pdf(x, df)                # densità di probabilità

plt.figure(figsize=(8, 4))
plt.plot(x, pdf, lw=2, label=f'χ²(df={df})')
plt.title('Densità della distribuzione Chi-quadro')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# —————————————————————————————
# 2) Test di bontà di adattamento
# —————————————————————————————
def goodness_of_fit(obs, exp):
    """
    Calcola la statistica χ² e il p-value per un test
    di bontà di adattamento (observed vs. expected).
    """
    stat = np.sum((obs - exp)**2 / exp)
    df = len(obs) - 1
    p = 1 - chi2.cdf(stat, df)
    return stat, df, p

osservati = np.array([30, 45, 25])
attesi     = np.array([35, 40, 25])
chi2_gof, df_gof, p_gof = goodness_of_fit(osservati, attesi)

print('— Bontà di adattamento —')
print(f'χ² = {chi2_gof:.4f}, df = {df_gof}, p-value = {p_gof:.4f}')


# —————————————————————————————
# 3) Test di indipendenza (contingenza)
# —————————————————————————————
def chi2_independence(table):
    """
    Calcola χ², gradi di libertà e p-value per una
    tabella di contingenza di dimensione (R×C).
    """
    row_tot = table.sum(axis=1)        # totali per riga
    col_tot = table.sum(axis=0)        # totali per colonna
    grand_tot = table.sum()            # totale generale

    # matrice dei valori attesi
    expected = np.outer(row_tot, col_tot) / grand_tot

    stat = np.sum((table - expected)**2 / expected)
    df = (table.shape[0] - 1) * (table.shape[1] - 1)
    p = 1 - chi2.cdf(stat, df)
    return stat, df, p

cont_table = np.array([
    [50, 30, 20],
    [40, 60, 50],
    [10, 20, 30]
])
chi2_ind, df_ind, p_ind = chi2_independence(cont_table)

print('\n— Test di indipendenza —')
print(f'χ² = {chi2_ind:.4f}, df = {df_ind}, p-value = {p_ind:.4e}')


# —————————————————————————————
# 4) Effetto dei gradi di libertà sul PDF
# —————————————————————————————
df_list = [1, 2, 3, 5, 10]
x = np.linspace(0, 20, 500)

plt.figure(figsize=(8, 4))
for d in df_list:
    plt.plot(x, chi2.pdf(x, d), label=f'df={d}')
plt.title('PDF χ² per diversi gradi di libertà')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# —————————————————————————————
# 5) Regione di rifiuto per α = 0.05
# —————————————————————————————
df_rr = 3
alpha = 0.05

# quantile critico
x_crit = chi2.ppf(1 - alpha, df_rr)

x = np.linspace(0, 15, 500)
y = chi2.pdf(x, df_rr)

plt.figure(figsize=(8, 4))
plt.plot(x, y, lw=2, label=f'χ²(df={df_rr})')
# evidenzio la regione di rifiuto
mask = x >= x_crit
plt.fill_between(x[mask], y[mask], color='red', alpha=0.5,
                 label=f'Rifiuto α={alpha}')
plt.axvline(x_crit, color='k', linestyle='--')
plt.title('Regione di rifiuto per il test χ²')
plt.xlabel('Statistica χ²')
plt.ylabel('f(x)')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

print(f'\nValore critico χ²(df={df_rr}, α={alpha}) = {x_crit:.4f}')
