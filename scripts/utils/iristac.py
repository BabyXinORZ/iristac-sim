
import mitsuba as mi
import drjit as dr 

from utils import led_ring
from utils.meshbuilder import grid_mesh_build
from utils import camera

mi.set_variant("cuda_ad_rgb")


class sensor():
    def __init__(self, cam:camera.Camera, LED_conf:led_ring.LED_params) -> None:
        self.cam = cam
        self.LED = LED_conf
        self.Z0_scene = self.buildEstimationScene()
        self.image_res = self.cam.image_res
        self.K = [0.2,0.2,0.2]
        self.compute_R()

    def get_surface_normal(self, I:mi.TensorXf)->mi.TensorXf:
        R_inverse = dr.inverse(self.R)
        N_Pixel = self.image_res[0]*self.image_res[1]
        I_arr = dr.gather(mi.Vector3f,I.array,dr.arange(mi.UInt,N_Pixel))
        return R_inverse@I_arr

    def set_Z0(self,Z:mi.TensorXf) -> None:
        self.Z0_scene = self.buildEstimationScene(Z)
        self.compute_R()

    def set_LED(self,LED_conf:led_ring.LED_params) -> None:
        self.LED = LED_conf
        self.compute_R()

    def get_tactile_img(self):
        I_compute_arr = self.R@self.si.n
        return mi.TensorXf(dr.ravel(I_compute_arr),(self.cam.image_res[1],self.cam.image_res[0],3))

    def compute_R(self)->mi.TensorXf:
        I_inc,D_inc = self.incident_intensity()
        N_Pixel = self.image_res[0]*self.image_res[1]
        R0 = mi.TensorXf(D_inc.array,[N_Pixel,self.LED.N,3,1])*I_inc
        for i in range(3):
            R0[:,:,:,i] = R0[:,:,:,i].array*self.K[i]
        R_float = dr.zeros(mi.Float,N_Pixel*3*3)
        R_index1 = dr.arange(mi.UInt,N_Pixel*3)
        R_index2 = dr.arange(mi.UInt,N_Pixel)
        a,_,c,d = dr.meshgrid(
            dr.arange(mi.UInt,N_Pixel),
            dr.zeros(mi.UInt,self.LED.N),
            dr.arange(mi.UInt,3),
            dr.arange(mi.UInt,3),
            indexing='ij' 
        )
        dr.scatter_reduce(dr.ReduceOp.Add, R_float, value=R0.array, index=a*3*3+c*3+d)
        R_array = dr.gather(mi.Vector3f,R_float,R_index1)
        R_matrix = dr.gather(mi.Matrix3f,R_array,R_index2)
        self.R = dr.transpose(R_matrix)

    def incident_intensity(self):
        self.si = self.cam.shot(self.Z0_scene)
        P_LED = self.LED.P
        N_LED = self.LED.N
        D_LED = self.LED.D
        # compute incident direction D_incident
        P_LED_t = mi.TensorXf(dr.ravel(P_LED),shape=[1,N_LED,3])
        N_Pixel = self.image_res[0]*self.image_res[1]
        P_Pixel_t = mi.TensorXf(dr.ravel(self.si.p),shape=[N_Pixel,1,3])
        D_incident = P_Pixel_t - P_LED_t
        D_incident_n = mi.TensorXf(dr.block_sum(dr.sqr(D_incident.array),3),shape=[N_Pixel,N_LED,1])
        D_incident = D_incident/dr.sqrt(D_incident_n)

        # compute angle between incident and LED normal
        D_LED_t = mi.TensorXf(dr.ravel(D_LED),shape=[1,N_LED,3])
        cosine2 = D_incident*D_LED_t
        cosine2 = mi.TensorXf(dr.block_sum(cosine2.array,3),shape=[N_Pixel,N_LED,1,1])
        # compute light max intensity  at the angle of the LED
        I_LED = dr.select(cosine2>0,cosine2,0)
        # I_LED = cosine2
        ### Exclude rays that are blocked by surface ###
        ray_test_o = P_Pixel_t - mi.TensorXf(0,shape=[1,N_LED,3])
        ray_test_o[:,:,2] = ray_test_o[:,:,2].array-1e-3
        id_test = dr.arange(mi.UInt,N_Pixel*N_LED)
        ray_test_o = dr.gather(mi.Point3f, ray_test_o.array, id_test)
        ray_test_d = dr.gather(mi.Vector3f, -D_incident.array, id_test)
        ray_test =  mi.Ray3f(o=ray_test_o,d=ray_test_d)
        ray_test_result = self.Z0_scene.ray_test(ray_test)
        self.I_LED = dr.select(mi.TensorXb(ray_test_result,[N_Pixel,N_LED,1,1]),0,I_LED)
        return self.I_LED*self.LED.E, D_incident

    def buildEstimationScene(self, Z:mi.TensorXf=None) -> mi.Scene:
        range_start = (self.cam.range[0][0],self.cam.range[0][1])
        range_end = (self.cam.range[1][0],self.cam.range[1][1])
        if Z is None:
            Z = mi.TensorXf(0,[self.cam.image_res[1],self.cam.image_res[0]])
            mesh = grid_mesh_build(range_start,range_end,Z)
        else:
            mesh = grid_mesh_build(range_start,range_end,Z)
        return mi.load_dict({
            "type": "scene",
            "facedisk": mesh,
        })

