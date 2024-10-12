import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom, norm

def simulate_galton_board(n, N):
    positions = np.zeros(N)
    
    for i in range(N):
        position = 0
        for _ in range(n):
            if np.random.rand() < 0.5:
                position += 1
        positions[i] = position
    return positions

def theoretical_binom(n, N):
    x = np.arange(0, n+1)
    binom_pmf = binom.pmf(x, n, 0.5) * N
    return x, binom_pmf

def theoretical_normal(n, N):
    mu = n / 2
    sigma = np.sqrt(n / 4)
    x = np.arange(0, n+1)
    normal_pdf = norm.pdf(x, mu, sigma) * N
    return x, normal_pdf

def main():
    n = int(input("Input the number of levels (n): "))
    N = int(input("Input the number of balls (N): "))

    positions = simulate_galton_board(n, N)
    counts, bins = np.histogram(positions, bins=np.arange(n+2) - 0.5, density=False)

    x_binom, binom_pmf = theoretical_binom(n, N)
    x_norm, normal_pdf = theoretical_normal(n, N)

    plt.bar(bins[:-1], counts, width=1, color='blue', alpha=0.6, label='Experimental')
    plt.plot(x_binom, binom_pmf, 'ro-', label='Binomial Distribution')
    plt.plot(x_norm, normal_pdf, 'k-', label='Normal Distribution')
    plt.xlabel('Position')
    plt.ylabel('Number of balls')
    plt.legend()
    plt.show()

    mse_binom = np.mean((counts - binom_pmf)**2)
    mse_normal = np.mean((counts - normal_pdf)**2)

    print(f'Mean Squared Error (Binomial): {mse_binom}')
    print(f'Mean Squared Error (Normal): {mse_normal}')

if __name__ == '__main__':
    main()