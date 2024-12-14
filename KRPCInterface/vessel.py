import numpy as np
import quaternion

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
    
    def GetMomentOfInertia(self):
        return self.vessel.moment_of_inertia
    
    def GetInertiaTensor(self):
        inertia_tensor = np.array(self.vessel.inertia_tensor).reshape(3, 3)

        return inertia_tensor
    
    def GetInertiaTensorDerivative(self):
        active_fuel_consumption = self.GetActiveFuelConsumption()
        active_fuel_parts = self.GetActiveFuelParts()

        active_fuel_parts_inertia_tensor_derivative = active_fuel_consumption*np.sum([np.array(fuel_part.inertia_tensor)/fuel_part.mass for fuel_part in active_fuel_parts], axis = 0)

        return np.array(active_fuel_parts_inertia_tensor_derivative).reshape(3, 3)
    
    def GetCOMPosition(self):
        return self.flight.center_of_mass
    
    def GetCOMVelocity(self):
        return self.flight.velocity
    
    def GetAttitude(self):
        return [self.flight.roll, self.flight.pitch, self.flight.heading]
    
    def GetCOMRotation(self):
        rotation = self.vessel.rotation(self.orbiting_body.reference_frame)
        rotation = np.quaternion(rotation[3], rotation[0], rotation[1], rotation[2])
        
        return rotation

    def GetCOMAngularVelocity(self):
        return np.array(self.vessel.angular_velocity(self.orbiting_body.reference_frame)).reshape(3,)
    
    def GetCOMAngularVelocityBodyFrame(self):
        rotation = self.GetCOMRotation()
        angular_velocity = np.array(self.GetCOMAngularVelocity())

        angular_velocity = rotation.conjugate() * np.quaternion(0, *angular_velocity) * rotation
        angular_velocity = np.array([angular_velocity.x, angular_velocity.y, angular_velocity.z])

        return np.array(angular_velocity)

    # The following are states related to the structure and form of the aircraft
    def GetParts(self):
        return self.parts
    
    def GetActiveEngines(self):
        return [part.engine for part in self.parts.all if part.engine and part.engine.active]
    
    def GetActiveFuelConsumption(self):
        gravity = self.GetGravity()

        return sum([active_engine.thrust / (active_engine.specific_impulse*gravity) for active_engine in self.GetActiveEngines()])
    
    def GetDecoupleStageFuelParts(self, decouple_stage):
        resources_in_current_stage = self.vessel.resources_in_decouple_stage(decouple_stage)

        return [resource.part for resource in resources_in_current_stage]
    
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
    
    def GetGravity(self):
        body = self.vessel.orbit.body
        gravitational_parameter = body.gravitational_parameter  # GM
        radius = self.vessel.orbit.radius
        gravity = gravitational_parameter / radius**2

        return gravity
    
    # Control Setters
    def SetThrottleControl(self, throttle_command):
        self.vessel.control.throttle = throttle_command

    def SetAttitudeControl(self, attitude):
        self.vessel.control.roll = attitude[0]
        self.vessel.control.pitch = attitude[1]
        self.vessel.control.yaw = attitude[2]

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

        if (fuel_consumption_rate == 0 ):
            return None
        
        burn_time = fuel_mass / fuel_consumption_rate if fuel_consumption_rate > 0 else float('inf')
        print(f"burn_time: {burn_time}")

        return burn_time
    