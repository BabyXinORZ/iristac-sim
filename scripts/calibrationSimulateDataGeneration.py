#%%
import mitsuba as mi
import drjit as dr 
import matplotlib.pyplot as plt
mi.set_variant("cuda_ad_rgb")
#%%
Intrinsics = mi.Matrix3f([[422.3949, 0,313.5413],[0,422.4885,233.1770],[0,0,1]])
Z_camera = -44
image_res = (640,480)
cam_conner1 = dr.inverse(Intrinsics)@mi.Point3f(0,0,1)*-Z_camera
cam_conner2 = dr.inverse(Intrinsics)@mi.Point3f(image_res[0]-1,image_res[1]-1,1)*-Z_camera


# ### Image pixel resolution ###
# image_res = (256,256)
# ### camera origin          ###
cam_origin = mi.Point3f(0,0,Z_camera)
cam_range = ((cam_conner1[0,0],cam_conner1[1,0]),(cam_conner2[0,0],cam_conner2[1,0]))
cam_fov= (cam_conner2[0,0]-cam_conner1[0,0],cam_conner2[1,0]-cam_conner1[1,0])
from utils import camera 
cam = camera.Camera(cam_origin, image_res,cam_range)

import utils.led_ring

N_LED = 24
LED0_pos = mi.Point3f(34.41,0,-12.5)
LED_ring = utils.led_ring.LED_params(N_LED, LED0_pos)

range_start = (cam.range[0][0],cam.range[0][1])
range_end = (cam.range[1][0],cam.range[1][1])
Z = mi.TensorXf(0,[cam.image_res[1],cam.image_res[0]])

from utils.meshbuilder import grid_mesh_build

mesh = grid_mesh_build(range_start,range_end,Z)
Z0_scene = mi.load_dict({
    "type": "scene",
    "facedisk": mesh,
})

si = cam.shot(Z0_scene)
P_LED = LED_ring.P
N_LED = LED_ring.N
D_LED = LED_ring.D



P_LED_t = mi.TensorXf(dr.ravel(P_LED),shape=[1,N_LED,3])
N_Pixel = image_res[0]*image_res[1]
P_Pixel_t = mi.TensorXf(dr.ravel(si.p),shape=[N_Pixel,1,3])
D_incident = P_Pixel_t - P_LED_t
D_incident_n = mi.TensorXf(dr.block_sum(dr.sqr(D_incident.array),3),shape=[N_Pixel,N_LED,1])
D_incident = D_incident/dr.sqrt(D_incident_n)

# compute angle between incident and LED normal
D_LED_t = mi.TensorXf(dr.ravel(D_LED),shape=[1,N_LED,3])
cosine2 = D_incident*D_LED_t
cosine2 = mi.TensorXf(dr.block_sum(cosine2.array,3),shape=[480,640,N_LED])
cosine2 = dr.select(cosine2>0,cosine2,0)
I_LED = dr.exp(-1/(cosine2+0.05)/2)*dr.exp(0.5)

D_inc_index = dr.arange(mi.UInt,N_Pixel)
I_temp = mi.TensorXf(0,(480,640))

normal_t = mi.TensorXf(dr.ravel(si.n),shape=[N_Pixel,1,3])
cosine1 = D_incident*normal_t
cosine1 = mi.TensorXf(dr.block_sum(cosine1.array,3),shape=[480,640,N_LED])
I_LED_t = cosine1*I_LED

k_rgb = [0.8,0.7,0.9]

for i_LED in range(24):
    I_temp_LED = I_LED_t[:,:,i_LED]
    for i_intensity in range(256):
        # noise = mi.sample_tea_float32(i_LED*256+i_intensity,noise_index)-0.5
        # I_noise = I_noise+noise/10
        I_noise = I_temp_LED*i_intensity/256
        for i_RGB in range(3):
            k = k_rgb[i_RGB]
            I_save = mi.TensorXf(0,(480,640,3))
            I_save[:,:,i_RGB] = I_noise.array*k
            path = "../data/calibration_sim/{0:02d}_{1:03d}_{2:01d}.png".format(i_LED,i_intensity,i_RGB)
            bmp = mi.Bitmap(I_save)
            bmp = bmp.convert(mi.Bitmap.PixelFormat.RGB, mi.Struct.Type.UInt8, False)
            bmp.write(path)
            print("\r", end=path, flush=True)

