import time

class HoverPIDController:
    def __init__(self, krpc_interface, target_surface_altitude, Kp, Ki, Kd):
        self.krpc_interface = krpc_interface
        self.target_surface_altitude = target_surface_altitude

        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd

        self.start = time.time()
        self.error_integral = 0
    
    def GetControlInput(self):
        surface_altitude = self.krpc_interface.GetActiveVessel().GetCOMSurfaceAltitude()
        radial_velocity = self.krpc_interface.GetActiveVessel().GetRadialVelocity()

        altitude_error = surface_altitude-self.target_surface_altitude
        altitude_speed_error = radial_velocity-0

        dt = (time.time() - self.start)
        self.error_integral += altitude_error * dt
        self.start = time.time()

        u0 = (self.krpc_interface.GetActiveVessel().GetMass() * self.krpc_interface.surface_gravity / self.krpc_interface.GetActiveVessel().GetAvailableThrust())
        throttle_command = max(u0 - self.Kp * altitude_error - self.Ki * self.error_integral - self.Kd * altitude_speed_error, 0.)

        return surface_altitude, radial_velocity, altitude_error, altitude_speed_error, throttle_command
        
