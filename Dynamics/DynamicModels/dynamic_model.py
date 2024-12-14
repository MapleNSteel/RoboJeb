from abc import ABC, abstractmethod
import inspect

class ModelParams(ABC):
    def __init__(self):
         pass
    
class ModelState(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def __add__(self, other):
        pass

    @abstractmethod    
    def __mul__(self, alpha):
        pass

    @abstractmethod
    def __rmul__(self, alpha):
        pass
    
class DynamicModel(ABC):

    def __init__(self, integrator):
        self.integrator = integrator

    @abstractmethod
    def Derivative(self, state, control, model_params):
        pass

    @abstractmethod
    def Update(self, state, control, model_params, dt):
        pass