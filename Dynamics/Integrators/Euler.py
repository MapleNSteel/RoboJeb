import numpy as np
def Integrate(state, control, derivative, dt):
    k1 = derivative(state, control)
    
    state_new = state + dt * k1
    return state_new
