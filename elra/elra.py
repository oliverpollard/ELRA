import numpy as np
import scipy.special as sc
from tqdm.auto import tqdm
import xarray as xr
import rioxarray
import sys

ICE_DENSITY = 916  # (kg/m3)
MANTLE_DENSITY = 3300  # (kg/m3)
GRAVIATIONAL_ACC = 9.81


def deform(grid_x, grid_y, topography, ice_thickness, D=None):
    if D is None:
        D = 10**25  # flexural rigidity (N m)

    q = ICE_DENSITY * GRAVIATIONAL_ACC * ice_thickness

    dx = np.abs(grid_x[1] - grid_x[0])
    dy = np.abs(grid_y[1] - grid_y[0])
    area = dx * dy

    P = area * q
    deformation = np.zeros_like(topography)

    L = (D / (MANTLE_DENSITY * GRAVIATIONAL_ACC)) ** (1 / 4)

    x_mesh, y_mesh = np.meshgrid(grid_x, grid_y)
    for i in tqdm(range(len(grid_y)), leave=False):
        for j in range(len(grid_x)):
            dx_mesh = x_mesh - grid_x[j]
            dy_mesh = y_mesh - grid_y[i]
            r = np.sqrt(dx_mesh**2 + dy_mesh**2)

            w_sum = 0
            for ii in range(len(grid_y)):
                for jj in range(len(grid_x)):
                    if r[ii, jj] < 6 * L:
                        w_temp = ((P[ii, jj] * L**2) / (2 * np.pi * D)) * sc.kei(
                            r[ii, jj] / L
                        )
                        w_sum = w_sum + w_temp

            deformation[i, j] = w_sum

    topography_deform = topography + deformation

    return topography_deform


class Deformer:
    def __init__(self, topography_da, coarsen_window):
        self.topography_da = topography_da
        self.coarsen_window = coarsen_window
        self.topography_coarse_da = topography_da.coarsen(
            dict(x=coarsen_window, y=coarsen_window), boundary="pad"
        ).mean()
        self.crs = topography_da.rio.crs

    @property
    def x(self):
        return self.topography_da.x.values

    @property
    def y(self):
        return self.topography_da.y.values

    @property
    def x_coarse(self):
        return self.topography_coarse_da.x.values

    @property
    def y_coarse(self):
        return self.topography_coarse_da.y.values

    @property
    def topography_coarse(self):
        return self.topography_coarse_da.values

    def deform(self, ice_da):
        ice_coarse_da = ice_da.coarsen(
            dict(x=self.coarsen_window, y=self.coarsen_window), boundary="pad"
        ).mean()
        ice_coarse = ice_coarse_da.values

        topography_deformed_coarse = deform(
            grid_x=self.x_coarse,
            grid_y=self.y_coarse,
            topography=self.topography_coarse,
            ice_thickness=ice_coarse,
        )
        # convert to xarray dataset
        topography_deformed_coarse_ds = xr.Dataset(
            data_vars=dict(
                z=(
                    ["y", "x"],
                    topography_deformed_coarse,
                ),
            ),
            coords={
                "y": self.y_coarse,
                "x": self.x_coarse,
            },
            attrs=None,
        )
        if self.crs is not None:
            topography_deformed_coarse_ds.rio.write_crs(self.crs, inplace=True)

        # interpolate back to grid and save
        topography_deformed_ds = topography_deformed_coarse_ds.interp(
            dict(x=self.x, y=self.y)
        )
        return topography_deformed_ds


def main():
    args = sys.argv
    topography_nc_file = args[1]
    ice_nc_file = args[2]
    time = float(args[3])
    coarsen_window = float(args[4])
    output_nc_file = args[5]
    print(args)

    topography_da = xr.open_dataset(topography_nc_file).z
    ice_da = xr.open_dataset(ice_nc_file).ice_thickness
    deformer = Deformer(topography_da=topography_da, coarsen_window=10)
    if time is not None:
        ice_da = ice_da.sel(time=time)

    deformed_topography = deformer.deform(ice_da=ice_da)
    deformed_topography.to_netcdf(output_nc_file)
