"""
Extended Kalman Filter for Constant Acceleration (CA) model in 3D.
State: [px, py, pz, vx, vy, vz, ax, ay, az]^T (9 dimensions)
"""

import numpy as np


def _as_col(x):
    """Convert array to column vector."""
    x = np.asarray(x, dtype=float)
    return x.reshape((-1, 1))


class EKFTrack:
    """
    EKF for a constant-acceleration (CA) model in 3D.
    State: [px, py, pz, vx, vy, vz, ax, ay, az]^T (9 dimensions)
    """
    def __init__(self, x0, P0, sigma_j=1.0):
        self.x = _as_col(x0)              # (9,1)
        self.P = np.array(P0, dtype=float)  # (9,9)
        self.sigma_j = float(sigma_j)     # jerk std deviation (m/s^3)

    def predict(self, dt):
        """Predict state forward by dt seconds."""
        dt = float(max(dt, 0.0))
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt3 * dt
        dt5 = dt4 * dt
        
        # State transition for CA model
        # p' = p + v*dt + 0.5*a*dt^2
        # v' = v + a*dt
        # a' = a
        F = np.eye(9)
        F[0, 3] = dt;  F[0, 6] = 0.5 * dt2  # px
        F[1, 4] = dt;  F[1, 7] = 0.5 * dt2  # py
        F[2, 5] = dt;  F[2, 8] = 0.5 * dt2  # pz
        F[3, 6] = dt  # vx
        F[4, 7] = dt  # vy
        F[5, 8] = dt  # vz

        # Process noise: white-noise jerk model
        q = self.sigma_j ** 2
        Q1 = np.array([
            [dt5/20, dt4/8,  dt3/6],
            [dt4/8,  dt3/3,  dt2/2],
            [dt3/6,  dt2/2,  dt    ]
        ]) * q
        
        Q = np.zeros((9, 9))
        Q[np.ix_([0, 3, 6], [0, 3, 6])] = Q1  # x-axis
        Q[np.ix_([1, 4, 7], [1, 4, 7])] = Q1  # y-axis
        Q[np.ix_([2, 5, 8], [2, 5, 8])] = Q1  # z-axis

        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def update(self, z, R, meas_type='pos_vel_acc'):
        """
        Update with measurement.
        
        Args:
            z: Measurement vector
            R: Measurement noise covariance
            meas_type: 'pos' (3D), 'pos_vel' (6D), or 'pos_vel_acc' (9D)
            
        Returns:
            nis: Normalized Innovation Squared (for gating/diagnostics)
        """
        z = _as_col(z)
        R = np.array(R, dtype=float)

        if meas_type == 'pos_vel_acc':
            H = np.eye(9)
        elif meas_type == 'pos_vel':
            H = np.zeros((6, 9))
            H[0, 0] = 1; H[1, 1] = 1; H[2, 2] = 1
            H[3, 3] = 1; H[4, 4] = 1; H[5, 5] = 1
        else:  # pos only
            H = np.zeros((3, 9))
            H[0, 0] = 1; H[1, 1] = 1; H[2, 2] = 1

        hx = H @ self.x
        y = z - hx
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        I = np.eye(9)
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ R @ K.T

        nis = float((y.T @ np.linalg.inv(S) @ y).squeeze())
        return nis

    def get_state(self):
        """Return state as dictionary."""
        x = self.x.squeeze()
        return {
            "posX": float(x[0]), "posY": float(x[1]), "posZ": float(x[2]),
            "velX": float(x[3]), "velY": float(x[4]), "velZ": float(x[5]),
            "accX": float(x[6]), "accY": float(x[7]), "accZ": float(x[8]),
        }


def build_measurement_matrix(meas_type):
    """Build observation matrix H based on measurement type."""
    if meas_type == 'pos_vel_acc':
        return np.eye(9)
    elif meas_type == 'pos_vel':
        H = np.zeros((6, 9))
        H[0, 0] = 1; H[1, 1] = 1; H[2, 2] = 1
        H[3, 3] = 1; H[4, 4] = 1; H[5, 5] = 1
        return H
    else:  # pos only
        H = np.zeros((3, 9))
        H[0, 0] = 1; H[1, 1] = 1; H[2, 2] = 1
        return H
