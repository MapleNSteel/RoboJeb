import numpy as np

class Vessel:
    def __init__(self, vessel):
        self.vessel = vessel
        self.parts = self.vessel.parts

        self.orbiting_body = self.GetOrbit().body
        
        self.flight = self.vessel.flight(self.orbiting_body.reference_frame)
    
    # Observations
    def GetCOMAltitude(self):
        return self.flight.mean_altitude
    
    def GetCOMSurfaceAltitude(self):
        return self.flight.surface_altitude

    def GetRadialVelocity(self):
        return self.flight.vertical_speed
    
    # State information
    def GetMass(self):
        return self.vessel.mass
    
    def GetMassDerivative(self):
        return self.GetActiveFuelConsumption()
    
    def GetInertiaTensor(self):
        return self.vessel.inertia_tensor
    
    def GetInertiaTensorDerivative(self):
        active_fuel_consumption = self.GetActiveFuelConsumption()
        active_fuel_parts = self.GetActiveFuelParts()

        active_fuel_parts_inertia_tensor_derivative = active_fuel_consumption*np.sum([np.array(fuel_part.inertia_tensor)/fuel_part.mass for fuel_part in active_fuel_parts], axis = 0)

        return active_fuel_parts_inertia_tensor_derivative
    
    def GetCOMPosition(self):
        return self.flight.center_of_mass
    
    def GetCOMVelocity(self):
        return self.flight.velocity
    
    def GetCOMRotation(self):
        return self.flight.rotation

    def GetCOMAngularVelocity(self):
        return self.flight.angular_velocity
    
    # The following are states related to the structure and form of the aircraft
    def GetParts(self):
        return self.parts
    
    def GetActiveEngines(self):
        return [part.engine for part in self.parts.all if part.engine and part.engine.active]
    
    def GetActiveFuelConsumption(self):
        body = self.vessel.orbit.body
        gravitational_parameter = body.gravitational_parameter  # GM
        radius = self.vessel.orbit.radius
        gravity = gravitational_parameter / radius**2

        print(f"gravity: {gravity}")

        return sum([active_engine.thrust / (active_engine.specific_impulse*gravity) for active_engine in self.GetActiveEngines()])
    
    def GetActiveFuelParts(self):
        active_fuel_parts = set()  # Use a set to avoid duplicates

        # Get all the active engines and check for their propellants
        active_engines = self.GetActiveEngines()

        # Iterate over all parts
        for part in self.vessel.parts.all:
            resources = part.resources

            # Iterate through each resource in the part
            for resource_name in resources.names:
                if self._is_active_fuel_part(resource_name, resources, active_engines):
                    active_fuel_parts.add(part)

        return list(active_fuel_parts)
    
    def GetOrbit(self):
        return self.vessel.orbit

    def GetOrbitalReferenceFrame(self):
        return self.vessel.orbital_reference_frame
    
    def GetSurfaceReferenceFrame(self):
        return self.vessel.surface_reference_frame
    
    # Control Setters
    def SetThrottleControl(self, throttle_command):
        self.vessel.control.throttle = throttle_command
    
    def _is_active_fuel_part(self, resource_name, resources, active_engines):
        """Helper function to check if a resource is used by an active engine and has a non-zero amount."""
        for engine in active_engines:
            if resource_name in engine.propellant_names and resources.amount(resource_name) > 0:
                return True
        return False
    
    # Control information
    def GetAvailableThrust(self):
        return self.vessel.available_thrust
    
    def GetAvailableTorque(self):
        return self.vessel.available_torque
    
    def GetBurnTime(self):
        fuel_consumption_rate = self.GetActiveFuelConsumption()
        active_fuel_parts = self.GetActiveFuelParts()

        wet_mass = sum([active_fuel_part.mass for active_fuel_part in active_fuel_parts])
        dry_mass = sum([active_fuel_part.dry_mass for active_fuel_part in active_fuel_parts])

        fuel_mass = wet_mass-dry_mass

        print(f"wet_mass: {wet_mass}, dry_mass: {dry_mass}")

        if (fuel_consumption_rate != 0 ):
            burn_time = fuel_mass / fuel_consumption_rate if fuel_consumption_rate > 0 else float('inf')
            print(f"burn_time: {burn_time}")
    
        return burn_time
    