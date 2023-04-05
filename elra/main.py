import numpy as np
import scipy.special as sc
from tqdm.auto import tqdm

ICE_DENSITY = 916 # (kg/m3)
MANTLE_DENSITY = 3300 # (kg/m3)
GRAVIATIONAL_ACC = 9.81

def deform(grid_x, grid_y, topography, ice_thickness, D=None):
    
    if D is None:
        D = 10**25    # flexural rigidity (N m)
        
    q = ICE_DENSITY * GRAVIATIONAL_ACC * ice_thickness
    
    dx = np.abs(grid_x[1] - grid_x[0])
    dy = np.abs(grid_y[1] - grid_y[0])
    area = dx * dy  

    P = area * q
    deformation = np.zeros_like(topography)

    L = (D/(MANTLE_DENSITY * GRAVIATIONAL_ACC))**(1/4)

    x_mesh, y_mesh = np.meshgrid(grid_x, grid_y)
    for i in tqdm(range(len(grid_y)), leave=False):
        for j in range(len(grid_x)):

            dx_mesh = x_mesh - grid_x[j]
            dy_mesh = y_mesh - grid_y[i]
            r = np.sqrt(dx_mesh**2 + dy_mesh**2)

            w_sum = 0
            for ii in range(len(grid_y)):
                for jj in range(len(grid_x)):
                    if r[ii,jj] < 6*L:
                        w_temp = ((P[ii,jj] * L**2)/(2*np.pi*D))*sc.kei(r[ii,jj]/L)
                        w_sum = w_sum + w_temp

            deformation[i,j] = w_sum
            
    topography_deform = topography + deformation
            
    return topography_deform