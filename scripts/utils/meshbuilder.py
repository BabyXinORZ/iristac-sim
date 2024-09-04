import mitsuba as mi
import drjit as dr 
mi.set_variant("cuda_ad_rgb")

def grid_mesh_build(start,stop,num,z):
    x1 = start[0];x2 = stop[0]
    y1 = start[1];y2 = stop[1]    
    n1 = num[0];n2 = num[1]

    x, y = dr.meshgrid(
        dr.linspace(mi.Float, x1,  x2, n1),
        dr.linspace(mi.Float, y1,  y2, n2)
    )
    vertex_pos = mi.Point3f(x, y, z)

    idx,idy =  dr.meshgrid(
        dr.arange(mi.UInt32, n1-1),
        dr.arange(mi.UInt32, n2-1)
    )

    indices1 = idx+idy*n1
    indices2 = idx+idy*n1+1
    indices3 = idx+(idy+1)*n1
    indices4 = idx+(idy+1)*n1+1

    face_indices1 = mi.Vector3u(indices1,indices2,indices3)
    face_indices2 = mi.Vector3u(indices2,indices4,indices3)
    face_indices = dr.repeat(face_indices1,2)
    dr.scatter(face_indices,face_indices2,dr.arange(mi.UInt32, (n1-1)*(n2-1))*2+1)

    mesh = mi.Mesh(
        "surface",
        vertex_count=n1*n2,
        face_count=  (n1-1)*(n2-1),
        has_vertex_normals=False,
        has_vertex_texcoords=False,
    )

    mesh_params = mi.traverse(mesh)
    mesh_params["vertex_positions"] = dr.ravel(vertex_pos)
    mesh_params["faces"] = dr.ravel(face_indices)

    mesh_params.update()
    return mesh