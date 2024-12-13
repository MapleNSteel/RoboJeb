import numpy as np
def Integrate(state, derivative, dt):
    k1 = derivative(state)
    k2 = derivative(state + 0.5 * dt * k1)
    k3 = derivative(state + 0.5 * dt * k2)
    k4 = derivative(state + dt * k3)
    
    state_new = state + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    return state_new
