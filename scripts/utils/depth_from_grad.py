import numpy as np



def frankotchellappa(grad_x, grad_y):
    rows, cols = grad_x.shape
    u_grid, v_grid = np.meshgrid(np.fft.fftfreq(cols),
                            np.fft.fftfreq(rows),indexing="xy")
    grad_x_F = np.fft.fft2(grad_x)
    grad_y_F = np.fft.fft2(grad_y)

    nominator = (-1j * u_grid * grad_x_F) + (-1j * v_grid * grad_y_F) 
    denominator = (u_grid**2) + (v_grid**2) + 1e-16
    Z_F = nominator / denominator/np.pi/2
    Z = np.real(np.fft.ifft2(Z_F))
    Z -=np.mean([Z[0,:],Z[-1,:],Z[0,:],Z[-1,:]])
    return Z


from scipy.fftpack import dst
from scipy.fftpack import idst


def fast_poisson(gx,gy):
    m,n = gx.shape
    gxx = np.zeros((m,n))
    gyy = np.zeros((m,n))
    f = np.zeros((m,n))
    img = np.zeros((m,n))
    gyy[1:,:-1] = gy[1:,:-1] - gy[:-1,:-1]
    gxx[:-1,1:] = gx[:-1,1:] - gx[:-1,:-1]
    f = gxx + gyy 
    f2 = f[1:-1,1:-1].copy()

    f_sinx = dst(f2,norm='ortho')
    f_sinxy = dst(f_sinx.T,norm='ortho').T

    x_mesh, y_mesh = np.meshgrid(range(n-2),range(m-2)) 
    x_mesh = x_mesh +1
    y_mesh = y_mesh +1
    denom = (2*np.cos(np.pi*x_mesh/(n-1))-2) + (2*np.cos(np.pi*y_mesh/(m-1))-2)

    f3 = f_sinxy/denom
    f_realx = idst(f3,norm='ortho')
    f_realxy = idst(f_realx.T,norm='ortho').T
    img[1:-1,1:-1] = f_realxy.copy()

    return img