import numpy as np

class CartPole:
    # Simplified continuous CartPole physics
    # Physical constants
    def __init__(self):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masscart + self.masspole
        self.length = 0.5
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.theta_threshold_radians = 12 * 2 * np.pi / 360 # 12 degrees in radians
        self.x_threshold = 2.4 # cart position bounds
        self.reset()

    def reset(self, noise=0.05):
        self.x = np.random.uniform(-noise, noise) # cart position
        self.x_dot = np.random.uniform(-noise, noise) # cart velocity
        self.theta = np.random.uniform(-noise, noise) # pole angle
        self.theta_dot = np.random.uniform(-noise, noise) # pole angular velocity

    def step(self, dt, action):
        force = float(np.sign(action)) * self.force_mag
        costheta = np.cos(self.theta)
        sintheta = np.sin(self.theta)
        # CartPole dynamics
        tmp = (force + self.polemass_length * self.theta_dot**2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * tmp) / \
                   (self.length * (4.0/3.0 - self.masspole * costheta**2 / self.total_mass))
        xacc = tmp - self.polemass_length * thetaacc * costheta / self.total_mass
        # Eulers integration
        self.x += dt * self.x_dot
        self.x_dot += dt * xacc
        self.theta += dt * self.theta_dot
        self.theta_dot += dt * thetaacc
        done = (abs(self.x) > self.x_threshold) or (abs(self.theta) > self.theta_threshold_radians)
        # Reward is 1 per time step as long as we are not done
        return (1.0 if not done else 0.0), done

    def state(self):
        return np.array([self.x, self.x_dot, self.theta, self.theta_dot], dtype=float)
