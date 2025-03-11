"""
Intelligent Control and Fault Diagnosis course - Spring 2025 - Amirkabir University of Technology (Tehran Polytechnic)
@author: Mahdi Shahrajabian 
Inverted Pendulum on a Cart System Envitonment

"""

import numpy as np
import gymnasium as gym
from gym import spaces

class InvertedPendulumCart:
    def __init__(self, m, M, L, u_min, u_max, Q, R, r_T, dt=0.01):

        # System Parameters 
        self.g = 9.81 # Gravitational acceleration (m/s^2)
        self.m = m # Mass of the pendulum (kg)
        self.M = M # mass of the cart (kg)
        self.L = L # Length of the pendulum (m)

        # Simulation parameters 
        self.dt = dt  # Time step (s)
        self.t = 0.0
        self.time = []

        # Cost parameters
        self.Q = Q
        self.R = R
        self.r_T = r_T
        # States: [x x_dot theta theta_dot]
        self.low_obs = np.array([-10, -np.finfo(np.float32).max, -8*np.pi/9, -np.finfo(np.float32).max], dtype=np.float32)
        self.high_obs = np.array([10, np.finfo(np.float32).max, 8*np.pi/9, np.finfo(np.float32).max], dtype=np.float32)

        self.min_action = u_min
        self.max_action = u_max
        
        self.action_space = spaces.Box(
            low=self.min_action,
            high=self.max_action,
            dtype=np.float32)
        
        self.observation_space = spaces.Box(
            low=self.low_obs,
            high=self.high_obs,
            dtype=np.float32)

        # Initial Condition Range 
        self.initial_min_vals = np.array([-1, -0.5, -np.pi/4, -0.2], dtype=np.float32)
        self.initial_max_vals = np.array([1, 0.5, np.pi/4, 0.2], dtype=np.float32)
        
    def _wrap_angle(self,angle):
        """Wrap angle between -pi and pi"""
        return (angle + np.pi) % (2 * np.pi) - np.pi
        
    def denormalize_action(self, action):
        """
        Denormalizes an action from the range [-1, 1] back to the original range [action_min, action_max].
        """ 
        return (action + 1) * (self.max_action - self.min_action) / 2 + self.min_action
    
    def _dynamics(self, t, state, u):
        # X = [x xdot theta theta_dot];
        # U = F;
        xdot = state[1]
        theta = state[2]
        theta_dot = state[3]
        u = np.squeeze(u)
        X_dot = [xdot, 
                 theta_dot,
                 (u+self.m*self.L*(theta_dot**2)*np.sin(theta)-self.m*self.g*np.sin(theta)*np.cos(theta))/(self.M+self.m*(1-np.cos(theta)**2)), 
                  (-u*np.cos(theta)-self.m*self.L*(theta_dot**2)*np.sin(theta)*np.cos(theta) +
                   (self.M+self.m)*self.g*np.sin(theta))/(self.L*(self.M+self.m*(1-np.cos(theta)**2)))]
                
        return np.array(X_dot)

    def _runge_kutta(self, state, u, t):
        """Runge-Kutta 4th order integration."""
        k1 = self._dynamics(t, state, u)
        k2 = self._dynamics(t, state + 0.5 * self.dt * k1, u)
        k3 = self._dynamics(t, state + 0.5 * self.dt * k2, u)
        k4 = self._dynamics(t, state + self.dt * k3, u)

        new_state = state.copy() + self.dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        return new_state

    def reset(self, seed=None):
        """Reset the environment."""
        if seed is not None:
            np.random.seed(seed)
        self.state = np.random.uniform(low=self.initial_min_vals, high=self.initial_max_vals) 
        self.t = 0.0
        self.time = []
        self.state_prev = self.state
        obs = self.state
        return obs

    def step(self, u, x_d):
        """Take a step in the environment."""

        u = self.denormalize_action(u)
        # Clip inputs to action space bounds
        u = np.clip(u, self.min_action, self.max_action)
        self.state_prev = self.state
        self.state = self._runge_kutta(self.state, u, self.t)

        # Wrap angles to the range [-pi, pi] for (phi,theta,psi,n1,n2,n3)
        self.state[2] = self._wrap_angle(self.state[2])

        self.t += self.dt
        self.time.append(self.t)
        x = (self.state.copy()).reshape(-1,1)
        e = (x-x_d)
        # Compute reward and check terminal conditions
        # Quadratic Cost 
        u = u.reshape(-1,1)
        cost = float(e.T @ self.Q @ e + u.T @ self.R @ u)
        # Terminal Reward
        done = False
        r_T = 0
        # Infeasible states 
        for i in range(len(self.state)):
            if self.state[i] > self.high_obs[i]:
                self.state[i] = self.high_obs[i]
                done = True
                r_T = self.r_T
                break
            if self.state[i] < self.low_obs[i]:
                self.state[i] = self.low_obs[i]
                done = True
                r_T = self.r_T
                break
                
        reward = r_T - cost 
        return self.state.copy(), reward, done