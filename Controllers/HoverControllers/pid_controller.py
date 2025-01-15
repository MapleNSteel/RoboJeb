import time

class HoverPIDController:
    def __init__(self, krpc_interface, target_surface_altitude, Kp, Ki, Kd):
        self.krpc_interface = krpc_interface
        self.target_surface_altitude = target_surface_altitude

        self.Kp = Kp * self.krpc_interface.GetActiveVessel().GetAvailableThrust()
        self.Ki = Ki * self.krpc_interface.GetActiveVessel().GetAvailableThrust()
        self.Kd = Kd * self.krpc_interface.GetActiveVessel().GetAvailableThrust()

        self.start = time.time()
        self.error_integral = 0
    
    def GetControlInput(self, surface_altitude, radial_velocity):
        altitude_error = surface_altitude-self.target_surface_altitude
        altitude_speed_error = radial_velocity-0.

        dt = (time.time() - self.start)
        self.error_integral += altitude_error * dt
        self.start = time.time()

        thrust_0 = (self.krpc_interface.GetActiveVessel().GetMass() * self.krpc_interface.surface_gravity)
        thrust = -(self.Kp * altitude_error + self.Ki * self.error_integral + self.Kd * altitude_speed_error)

        u = (thrust_0 + thrust)/self.krpc_interface.GetActiveVessel().GetAvailableThrust()

        throttle_command = max(u, 0.)

        return surface_altitude, radial_velocity, altitude_error, altitude_speed_error, throttle_command
        
