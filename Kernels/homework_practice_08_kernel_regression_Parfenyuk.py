import numpy as np
from sklearn.base import RegressorMixin
from sklearn.gaussian_process.kernels import RBF
from tqdm import tqdm

class KernelRidgeRegression(RegressorMixin):
    def __init__(self, lr=0.01, regularization=1.0, tolerance=1e-2, max_iter=1000, batch_size=64, kernel_scale=1.0):
        self.lr = lr
        self.regularization = regularization
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.kernel = RBF(kernel_scale)

        self.w = None
        self.b = None
        self.loss_history = []
        self.x_train = None
        self.kernel_x = None

    def calc_loss(self, x: np.ndarray, y: np.ndarray) -> float:
        return 0.5 * np.linalg.norm(self.kernel_x @ self.w - y) ** 2 + 0.5 * self.regularization * np.dot(self.w.T, self.kernel_x @ self.w)

    def calc_grad(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
        batch_indices = np.random.randint(0, x.shape[0], self.batch_size)
        kernel_batch = self.kernel(x[batch_indices], x)
        Q = kernel_batch.T @ kernel_batch @ self.w + kernel_batch.T @ np.ones((self.batch_size, 1)) + kernel_batch.T @ y[batch_indices] + self.regularization * self.kernel_x @ self.w
        grad_w = Q
        grad_b = np.mean(self.b + kernel_batch @ self.w - y[batch_indices])
        return grad_w, grad_b
   

    def fit(self, x: np.ndarray, y: np.ndarray) -> "KernelRidgeRegression":
        self.w = np.random.rand(x.shape[0]).reshape(x.shape[0], 1)
        self.b = np.random.rand(1).reshape(-1, 1)
        self.kernel_x = self.kernel(x)
        self.x_train = x
        for _ in range(self.max_iter):
            self.loss_history.append(self.calc_loss(x, y).item())
            delta = self.step(x, y)
            if np.linalg.norm(delta)**2  < self.tolerance:
                break

        self.loss_history.append(self.calc_loss(x, y).item())
        return self
    
    def step(self, x: np.ndarray, y: np.ndarray):
      return self.update_weights(self.calc_grad(x, y))

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        weight_difference = -self.lr * gradient[0]
        weight_difference = -self.lr * gradient[0]
        self.w += weight_difference
        self.b -= self.lr * gradient[1]

        return weight_difference


    def fit_closed_form(self, x: np.ndarray, y: np.ndarray) -> "KernelRidgeRegression":
        self.x_train = x
        self.kernel_x = self.kernel(x, x)
        n = x.shape[0]
        self.w = np.linalg.inv(self.kernel_x + self.regularization * np.eye(n)) @ y
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
      kernel_pred = self.kernel(x, self.x_train)
      return kernel_pred @ self.w + self.b
