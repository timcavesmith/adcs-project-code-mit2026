import numpy as np
from utils import *

class bearing_sensor:
    def __init__(self, body_vector, bias_vector, M, covW):
        self.set_body_vector(body_vector)
        self.set_bias(bias_vector)
        self.set_M(M)
        self.set_covW(covW)

    def set_body_vector(self, body_vector):
        body_vector = np.asarray(body_vector)
        if body_vector.shape not in ((3,), (3, 1)):
            raise ValueError("body_vector must be shape (3,) or (3, 1)")
        self.body_vector = body_vector.reshape(3,)

    def set_bias(self, bias_vector):
        bias_array = np.asarray(bias_vector)
        if bias_array.shape not in ((3,), (3, 1)):
            raise ValueError("bias_vector must be shape (3,) or (3, 1)")
        self.bias_vector = bias_array.reshape(3,)

    def set_M(self, M):
        M = np.asarray(M)
        if M.shape != (3, 3):
            raise ValueError("M must be shape (3, 3)")
        self.M = M

    def set_covW(self, covW):
        covW = np.asarray(covW)
        if covW.shape != (3, 3):
            raise ValueError("covW must be shape (3, 3)")
        self.covW = covW

    def getMsmt(self, q_true):
        q_true = q_true.reshape(4,)
        y_true = Q(q_true).T @ self.body_vector
        w = np.random.multivariate_normal(np.zeros(3), self.covW)
        y = self.M @ y_true + self.bias_vector + w.T
        return y.reshape(3,)


class star_tracker:
    def __init__(self, covW):
        self.set_covW(covW)

    def set_covW(self, covW):
        covW = np.asarray(covW)
        if covW.shape != (3, 3):
            raise ValueError("covW must be shape (3, 3)")
        self.covW = covW

    def getMsmt(self, q_true):
        q_true = q_true.reshape(4,)
        w = np.random.multivariate_normal(np.zeros(3), self.covW)
        q = q_true @ R(expq(w))
        return q.reshape(4,)


class gyro:
    def __init__(self, sensor_rate, M, covBias, covW):
        self.sensor_rate = sensor_rate
        self.set_M(M)
        self.set_covBias(covBias)
        self.set_covW(covW)
        self.b = 0

    def set_M(self, M):
        M = np.asarray(M)
        if M.shape != (3, 3):
            raise ValueError("M must be shape (3, 3)")
        self.M = M

    def set_covBias(self, covBias):
        covBias = np.asarray(covBias)
        if covBias.shape != (3, 3):
            raise ValueError("covBias must be shape (3, 3)")
        self.covBias = covBias

    def set_covW(self, covW):
        covW = np.asarray(covW)
        if covW.shape != (3, 3):
            raise ValueError("covW must be shape (3, 3)")
        self.covW = covW

    def getMsmt(self, w_true):
        w_true = w_true.reshape(3,)
        w = np.random.multivariate_normal(np.zeros(3), self.covW * np.sqrt(self.sensor_rate))  # noise scaled by sensor rate
        self.b += np.random.multivariate_normal(np.zeros(3), self.covBias / np.sqrt(self.sensor_rate))  # bias drift scaled by sensor rate
        y = self.M @ w_true + self.b + w
        return y