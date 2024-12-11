from .vessel import Vessel

class SpaceCenter:
    def __init__(self, connection):
        self.space_center = connection.space_center
    
    def GetActiveVessel(self):
        return Vessel(self.space_center.active_vessel)