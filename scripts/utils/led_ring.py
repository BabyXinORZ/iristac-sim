import mitsuba as mi
import drjit as dr 
from typing import Tuple
mi.set_variant("cuda_ad_rgb")

class LED_params():
    def __init__(self, N_LED:int, LED0_pos:mi.Point3f) -> None:
        self.N = N_LED
        self.P = None
        self.D = None
        self.E = mi.TensorXf(0,[1,N_LED ,1,3])
        
        angle = 2*dr.pi/N_LED*dr.arange(mi.Float,N_LED)
        Q = dr.rotate(mi.Quaternion4f,mi.Vector3f(0, 0, 1),angle)
        self.P = dr.quat_to_matrix(Q, 3)@LED0_pos

        self.D = -self.P
        self.D.z = 0
        self.D = dr.normalize(self.D)
    
    def set_sine_intensity(self, A:float, B:float) -> None:
        i_list = dr.arange(mi.Float,self.N)
        self.E[:,:,:,0] = A*dr.sin(2*dr.pi/self.N *i_list)+B
        self.E[:,:,:,1] = A*dr.sin(2*dr.pi/self.N *(i_list-self.N /3))+B
        self.E[:,:,:,2] = A*dr.sin(2*dr.pi/self.N *(i_list+self.N /3))+B
