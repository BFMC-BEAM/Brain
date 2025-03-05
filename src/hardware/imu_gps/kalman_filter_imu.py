import numpy as np
from pykalman import KalmanFilter

class KalmanFilterIMU:
    def __init__(self, dt, process_noise_cov, measurement_noise_cov, initial_error_cov):
        self.dt = dt
        initial_state = [0, 0, 0, 0]  # [pos_x, pos_y, vel_x, vel_y]
        transition_matrix = [[1, 0, dt, 0],
                              [0, 1, 0, dt],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1]]
        observation_matrix = [[0, 0, 0, 0],
                              [0, 0, 0, 0],
                              [1, 0, 0, 0],
                              [0, 1, 0, 0]]  # Observamos la aceleración
        self.kf = KalmanFilter(transition_matrices=transition_matrix,
                                observation_matrices=observation_matrix,
                                initial_state_mean=initial_state,
                                initial_state_covariance=initial_error_cov,
                                transition_covariance=process_noise_cov,
                                observation_covariance=measurement_noise_cov)
        self.state = initial_state
        self.covariance = initial_error_cov

    def update(self, acceleration):
        observation = [0, 0, acceleration[0], acceleration[1]]  # Observamos la aceleración
        self.state, self.covariance = self.kf.filter_update(self.state, self.covariance, observation)
        return np.array([self.state[0], self.state[1]]), np.array([self.state[2], self.state[3]])