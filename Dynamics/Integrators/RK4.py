import numpy as np
def Integrate(state, control, derivative, dt):
    k1 = derivative(state, control)
    k2 = derivative(state + 0.5 * dt * k1, control)
    k3 = derivative(state + 0.5 * dt * k2, control)
    k4 = derivative(state + dt * k3, control)
    
    state_new = state + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    return state_new
