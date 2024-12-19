import krpc
import numpy as np

from .space_center import SpaceCenter

class KRPCInterface:
    def __init__(self):
        self.connection = krpc.connect(address='host.docker.internal')

        self.space_center = SpaceCenter(self.connection)
        self.active_vessel = self.space_center.GetActiveVessel()

        self.surface_gravity = self.active_vessel.GetOrbit().body.surface_gravity

    def GetConnection(self):
        return self.connection

    def GetSpaceCenter(self):
        return self.space_center
    
    def GetActiveVessel(self):
        return self.active_vessel
