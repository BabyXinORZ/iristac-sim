import mitsuba as mi
import drjit as dr 
from typing import Tuple
mi.set_variant("cuda_ad_rgb")

class Camera():

    # def __init__(self, origin: mi.Point3f, image_res:Tuple[float,float],fov:Tuple[float,float]) -> None:
    #     self.fov = fov
    #     self.image_res = image_res
    #     x, y = dr.meshgrid(
    #         dr.linspace(mi.Float, -fov[0]/2, fov[0]/2, image_res[0]),
    #         dr.linspace(mi.Float, -fov[1]/2, fov[1]/2, image_res[1])
    #     )
    #     ray_dir = mi.Vector3f(x, y, 0)-origin
    #     self.ray = mi.Ray3f(o=origin, d=ray_dir)

    def __init__(self, origin: mi.Point3f, image_res:Tuple[float,float],range:Tuple[Tuple[float,float],Tuple[float,float]]) -> None:
        self.range = range
        self.image_res = image_res
        x, y = dr.meshgrid(
            dr.linspace(mi.Float, range[0][0], range[1][0], image_res[0]),
            dr.linspace(mi.Float, range[0][1], range[1][1], image_res[1])
        )
        ray_dir = mi.Vector3f(x, y, 0)-origin
        self.ray = mi.Ray3f(o=origin, d=ray_dir)

    def shot(self, scene: mi.Scene)->mi.SurfaceInteraction3f:
        return scene.ray_intersect(self.ray)