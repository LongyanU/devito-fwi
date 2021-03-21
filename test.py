from seismic import demo_model, AcquisitionGeometry, Receiver
from seismic import plot_shotrecord
import numpy as np
from fwi import fm_single, fm_multi
from scipy import optimize
from distributed import Client, wait, LocalCluster
import dask
dask.config.set({'distributed.worker.multiprocessing-method': 'spawn'})

# Set up velocity model
shape = (101, 101)      # Number of grid points (nx, nz).
spacing = (10., 10.)    # Grid spacing in m. The domain size is now 1km by 1km.
origin = (0, 0)         # Need origin to define relative source and receiver locations.
nbl = 40

# True model
model1 = demo_model('circle-isotropic', vp_circle=3.0, vp_background=2.5,
    origin=origin, shape=shape, spacing=spacing, nbl=nbl)

# Initial model
model0 = demo_model('circle-isotropic', vp_circle=2.5, vp_background=2.5,
    origin=origin, shape=shape, spacing=spacing, nbl=nbl, grid = model1.grid)

# Set up acquisiton geometry
t0 = 0.
tn = 1000. 
f0 = 0.010

# Set up source geometry, but define 5 sources instead of just one.
nsources = 5
src_coordinates = np.empty((nsources, 2))
src_coordinates[:, 1] = np.linspace(0, model1.domain_size[0], num=nsources)
src_coordinates[:, 0] = 20.  # Source depth is 20m

# Initialize receivers for synthetic and imaging data
nreceivers = 101
rec_coordinates = np.empty((nreceivers, 2))
rec_coordinates[:, 1] = np.linspace(spacing[0], model1.domain_size[0] - spacing[0], num=nreceivers)
rec_coordinates[:, 0] = 980.    # Receiver depth
# Set up geometry objects for observed and predicted data
geometry1 = AcquisitionGeometry(model1, rec_coordinates, src_coordinates, t0, tn, f0=f0, src_type='Ricker')
geometry0 = AcquisitionGeometry(model0, rec_coordinates, src_coordinates, t0, tn, f0=f0, src_type='Ricker')

cluster = LocalCluster(n_workers=1, death_timeout=600)
client = Client(cluster)

obs = fm_multi(client, geometry1, save=False, stride=1.)

plot_shotrecord(obs[2].data, model1, t0, tn)