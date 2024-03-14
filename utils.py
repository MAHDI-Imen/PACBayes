import numpy as np
import matplotlib.pyplot as plt

def generate_toy_data(n_samples=15, show=False, sigma=0.25, type="sin", w=None):
    if type == "sin":
        noise = np.random.randn(n_samples) * sigma # gaussian noise
        x = np.random.uniform(0, 2 * np.pi, n_samples) # x uniformly distributed in [0, 2pi]
        y = np.sin(x) + noise # y = sin(x) + noise
    elif type == "cos":
        noise = np.random.randn(n_samples) * sigma
        x = np.random.uniform(0, 2 * np.pi, n_samples)
        y = np.cos(x) + noise
    elif type == "gaussian":
        if w is None:
            w = np.random.randn(50)
            w = w / np.linalg.norm(w) / 2
        noise = np.random.randn(n_samples) * sigma
        d = w.shape[0]
        x = np.random.randn(n_samples, d)
        noise = np.random.randn(n_samples) * sigma
        y = x @ w + noise
        
        
    if show:
        if type == "sin":
            plt.scatter(x, y, s=3, c='black')
            plt.xlim(0, 2 * np.pi + 0.75)
            plt.ylim(-2, 1.5)
            # x axis ticks to 2 pi  multiple of 1/2 pi
            plt.xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi], ["0", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2\pi$"])
        elif type == "cos":
            plt.scatter(x, y, s=3, c='black')
            plt.xlim(0, 2 * np.pi + 0.75)
            plt.ylim(-2, 1.5)
            plt.xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi], ["0", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2\pi$"])
        elif type == "gaussian":
            plt.scatter(x[:, 0], y, s=3, c='black')
            plt.xlim(-3, 3)
            plt.ylim(-3, 3)
     
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(f"Toy data with sigma = {sigma} and n_samples = {n_samples}")
        plt.grid()
        plt.show()        
    return x, y


def KL_divergence(d, A, A_inv, w_hat, sigma_pi_sq=1/0.005):
    kl = 0.5 * ((1/sigma_pi_sq) * (np.trace(A_inv) + w_hat.T @ w_hat) - (d) + np.log(np.linalg.det(A)) + (d) * np.log(sigma_pi_sq))

    return kl

def empirical_risk(y, w, phi_x, bounded=False, a=1, b=4, sigma_sq=0.5):
    if bounded:
        # limit the values of y- phi_x @ w to be between a and b
        bounded_loss = np.maximum(a, np.minimum(b, y - phi_x @ w))
        return 0.5 * ((1/sigma_sq) * np.mean(bounded_loss**2) + np.log(2 * np.pi * sigma_sq))
    else:
        return  0.5 * ((1/sigma_sq) * np.mean((y - phi_x @ w)**2) + np.log(2 * np.pi * sigma_sq))

def empirical_loss(n, y, w_hat, phi_x, A_inv, bounded=False, sigma_sq=0.5):
    return n * empirical_risk(y, w_hat, phi_x, bounded, sigma_sq=sigma_sq) + 0.5 * (1/sigma_sq) * np.trace(phi_x.T @ phi_x @ A_inv)



def get_losses(x, y, d, n, bounded=False, sigma_sq=0.5, sigma_pi_sq=1/0.005, phi=None):
    if phi is None:
        phi_x = x.copy()
    else:
        phi_x = phi(x, d)
    A = (1/sigma_sq) * phi_x.T @ phi_x + (1/sigma_pi_sq) * np.eye(d+1)
    A_inv = np.linalg.inv(A)
    w_hat = (1/sigma_sq) * A_inv @ phi_x.T @ y
    return empirical_loss(n, y, w_hat, phi_x, A_inv, bounded, sigma_sq=sigma_sq), KL_divergence(d, A, A_inv, w_hat, sigma_pi_sq=sigma_pi_sq)