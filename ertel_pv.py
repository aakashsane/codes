# calculate ertel PV

import numpy as np
import netCDF4 as ncd
import matplotlib.pyplot as plt
import vertplot as vert

sig=np.linspace(-1,0,16);
z=vert.z4ms(sig,100,0.1);

d=ncd.Dataset('croco_avg.00024.nc')
g=ncd.Dataset('croco_osom_grd.nc')
mask_psi=g.variables['mask_psi'][:]

zeta=d.variables['zeta'][:]
h=g.variables['h'][:]
zlev=np.zeros([12,16,1100, 1000])
zlev=vert.zgrid4ms(sig,h,zeta)
dz1=np.diff(zlev,axis=1)  # dz is at rho points! 

pm1=g.variables['pm'][:]
pn1=g.variables['pn'][:]
f1=g.variables['f'][:]

pm=np.tile(pm1,(12,15,1,1))
pn=np.tile(pn1,(12,15,1,1))
f=np.tile(f1,(12,14,1,1))

u=d.variables['u'][:]
v=d.variables['v'][:]
rho=d.variables['rho'][:]+1025

# Ertel PV, term:  (dv/dx - du/dy) * drho / dz * (-g/rho0)   k term

# compute dv/dxi at psi points:
dxm1 = 0.25 * (pm[:,:, :-1, 1:] + pm[:,:, 1:, 1:] + pm[:,:, :-1, :-1] + pm[:,:, 1:, :-1])
dvdxi = np.diff(v, n=1, axis=3) * dxm1

# compute du/deta at psi points
dym1 = 0.25 * (pn[:,:, :-1, 1:] + pn[:,:, 1:, 1:] + pn[:,:, :-1, :-1] + pn[:,:, 1:, :-1])
dudeta = np.diff(u, n=1, axis=2) * dym1

# Omega is Ertel PV at horizontal horiz psi points and vertical rho points
omega=dvdxi - dudeta

ddz1 = 0.5 * (dz1[:,:-1, :, :] + dz1[:,1:, :, :])

drho=np.diff(rho,axis=1)
dz_w=np.diff(dz1,axis=1)

drhodz=drho/dz_w  # horiz rho points and vertical rho-1 points
drhodz=0.25 * (drhodz[:,:, :-1, 1:] + drhodz[:,:, 1:, 1:] +
              drhodz[:,:, :-1, :-1] + drhodz[:,:, 1:, :-1]) # at psi points

f2=0.25 * (f[:,:, :-1, 1:] + f[:,:, 1:, 1:] +
          f[:,:, :-1, :-1] + f[:,:, 1:, :-1]) # at psi points

omega=0.5*(omega[:,:-1,:,:]+omega[:,1:,:,:])

epvk=(f2+omega)*drhodz  # epvk is at psi points

mask_psi[mask_psi==0]=np.nan
for i in range(12):
    for j in range(14):
        epvk[i,j,:,:]=epvk[i,j,:,:]*mask_psi*(-9.8/1025)

# Ertel PV, term:  (dv/dz)*(drho/dx)* ( -g /rho0)    

dz_psi=0.25 * (dz1[:,:, :-1, 1:] + dz1[:,:, 1:, 1:] + dz1[:,:, :-1, :-1] + dz1[:,:, 1:, :-1])
v_psi=0.5* (v[:,:,:,:-1]+v[:,:,:,1:])
dv_z_psi=np.diff(v_psi,axis=1)
dz_psi_14=0.5*(dz_psi[:,0:-1,:,:]+dz_psi[:,1:,:,:])
dvdz=dv_z_psi/dz_psi_14   # psi ppints and w1 to w_n-1 points

# drho / dx
## drho
drho_x= np.diff(rho,axis=3)
drho_x_psi=  0.5* ( drho_x[:,:,:-1,:] + drho_x[:,:,1:,:])
## dx
pm_psi=0.25 * (pm[:,:, :-1, 1:] + pm[:,:, 1:, 1:] + pm[:,:, :-1, :-1] + pm[:,:, 1:, :-1])
drhodx_psi=drho_x_psi*pm_psi

drhodx_psi = 0.5*( drhodx_psi[:,:-1,:,:]+drhodx_psi[:,1:,:,:])

epvi=dvdz*drhodx_psi

mask_psi[mask_psi==0]=np.nan
for i in range(12):
    for j in range(14):
        epvi[i,j,:,:]=epvi[i,j,:,:]*mask_psi*(-9.8/1025)

# Ertel PV, term:  (du/dz)*(drho/dy) * ( -g /rho0)   

u_psi=0.5* (u[:,:,:-1,:]+u[:,:,1:,:])
du_z_psi=np.diff(u_psi,axis=1)
dudz=du_z_psi/dz_psi_14  # psi ppints and w1 to w_n-1 points

# drho / dy
## drho
drho_y= np.diff(rho,axis=2)
drho_y_psi=  0.5* ( drho_y[:,:,:,:-1] + drho_y[:,:,:,1:])
## dy
pn_psi=0.25 * (pn[:,:, :-1, 1:] + pn[:,:, 1:, 1:] + pn[:,:, :-1, :-1] + pn[:,:, 1:, :-1])
drhody_psi=drho_y_psi*pn_psi
drhody_psi = 0.5*( drhody_psi[:,:-1,:,:]+drhody_psi[:,1:,:,:])

epvj=dudz*drhody_psi

for i in range(12):
    for j in range(14):
        epvj[i,j,:,:]=epvj[i,j,:,:]*mask_psi*(-9.8/1025)

# final epv
epv=epvk+epvj-epvi
