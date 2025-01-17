# -*-Python-*-
# Created by wu at 19 Jun 2019  16:33
import numpy as np
import scipy
import copy
from pam_globs import *


def get_plasma_info(time):
    """
    This function generates plasma formation that PAM uses

    :param time: time array
    """

    root['OUTPUTS']['plasma'] = {}#OMFITtree()
    ntransport = root['INPUTS']['pam.in']['input']['ntransport']
    ntheta = root['INPUTS']['pam.in']['input']['ntheta']
    rho_transport = root['OUTPUTS']['plasma']['rho_transport'] = np.linspace(0, 1, ntransport)
    gEQDSK = root['INPUTS']['gEQDSK']

    rhovn = gEQDSK['RHOVN']
    rhop_geqdsk = np.sqrt(np.linspace(0, 1, len(rhovn)))
    root['OUTPUTS']['plasma']['rhop_transport'] = np.interp(rho_transport, rhovn, rhop_geqdsk)

    drho = abs(gEQDSK['AuxQuantities']['PSI'][0] - gEQDSK['AuxQuantities']['PSI'][1])
    dVoldrho = np.gradient(gEQDSK['fluxSurfaces']['geo']['vol']) / np.gradient(gEQDSK['RHOVN'])
    dVoldrho = np.interp(rho_transport, rhovn, dVoldrho)
    root['OUTPUTS']['plasma']['dVoldrho'] = dVoldrho
    root['OUTPUTS']['plasma']['time'] = time
    root['OUTPUTS']['plasma']['vol'] = vol = np.interp(rho_transport, rhovn, gEQDSK['fluxSurfaces']['geo']['vol'])
    R0 = gEQDSK['RMAXIS']
    root['OUTPUTS']['plasma']['rvol'] = np.sqrt(vol / (2 * np.pi**2 * R0))
    rho_in = root['INPUTS']['pam.in']['equilibrium']['rho']
    ne_in = root['INPUTS']['pam.in']['equilibrium']['ne']
    Te_in = root['INPUTS']['pam.in']['equilibrium']['Te']

    root['OUTPUTS']['plasma']['ne_time'] = np.zeros([len(time), ntransport])
    root['OUTPUTS']['plasma']['ne_time'][0, :] = np.interp(rho_transport, rho_in, ne_in)

    root['OUTPUTS']['plasma']['nd_time'] = np.zeros([len(time), ntransport])
    root['OUTPUTS']['plasma']['nd_time'][0, :] = np.interp(rho_transport, rho_in, ne_in)

    root['OUTPUTS']['plasma']['Te_time'] = np.zeros([len(time), ntransport])
    root['OUTPUTS']['plasma']['Te_time'][0, :] = np.interp(rho_transport, rho_in, Te_in)


def get_interps():
    """
    This function returns interpolations of the derivative of rho with
    respect to R and Z coordinates. This is needed to calculate cloud
    size in rho space
    """

    gEQDSK = root['INPUTS']['gEQDSK']
    R = copy.deepcopy(gEQDSK['AuxQuantities']['R'])
    Z = copy.deepcopy(gEQDSK['AuxQuantities']['Z'])

    dR = abs(R[0] - R[1])
    dZ = abs(Z[0] - Z[1])
    drhorz = np.gradient(gEQDSK['AuxQuantities']['RHORZ'], dZ, dR)
    drhodR = scipy.interpolate.interp2d(R, Z, drhorz[1])
    drhodZ = scipy.interpolate.interp2d(R, Z, drhorz[0])

    return drhodR, drhodZ


def generate_RZ_grid():
    """
    This function generates the R,Z grid for PAM 2D Gaussian deposition model
    """

    ntransport = root['INPUTS']['pam.in']['input']['ntransport']
    tgrid = root['INPUTS']['pam.in']['input']['ntheta']
    gEQDSK = root['INPUTS']['gEQDSK']
    geo = gEQDSK['fluxSurfaces']['geo']
    rgrid = ntransport
    rho = np.linspace(0, 1, ntransport)
    rho_efit = gEQDSK['RHOVN']

    R = geo['R_centroid'][0] * np.ones([len(rho_efit), tgrid])
    Z = geo['Z_centroid'][0] * np.ones([len(rho_efit), tgrid])

    modBp = np.zeros([len(rho_efit), tgrid])
    gEQDSK['fluxSurfaces'].resample(npts=tgrid)
    R0 = gEQDSK['fluxSurfaces']['R0']
    Z0 = gEQDSK['fluxSurfaces']['Z0']
    for i in gEQDSK['fluxSurfaces']['flux']:
        Rtmp = gEQDSK['fluxSurfaces']['flux'][i]['R'].copy()
        Ztmp = gEQDSK['fluxSurfaces']['flux'][i]['Z'].copy()

        modBptmp = np.sqrt(gEQDSK['fluxSurfaces']['flux'][i]['Br'] ** 2 + gEQDSK['fluxSurfaces']['flux'][i]['Bz'] ** 2)
        thetatmp = np.arctan2(Ztmp - Z0, Rtmp - R0) % (2 * np.pi)
        theta = sorted(thetatmp[1:])
        R[i, 1:] = [r for t, r in sorted(zip(thetatmp, Rtmp[1:]))]

        Z[i, 1:] = [z for t, z in sorted(zip(thetatmp, Ztmp[1:]))]
        modBp[i, 1:] = [Bp for t, Bp in sorted(zip(thetatmp, modBptmp[1:]))]

    R[:, 0] = R[:, -1]
    Z[:, 0] = Z[:, -1]
    modBp[:, 0] = modBp[:, -1]

    root['OUTPUTS']['plasma']['theta'] = np.append(theta, theta[0] + 2 * np.pi)
    root['OUTPUTS']['plasma']['Rgrid'] = R = scipy.interpolate.interp1d(rho_efit, R, axis=0)(rho)
    root['OUTPUTS']['plasma']['Zgrid'] = Z = scipy.interpolate.interp1d(rho_efit, Z, axis=0)(rho)
    root['OUTPUTS']['plasma']['modBp'] = scipy.interpolate.interp1d(rho_efit, modBp, axis=0)(rho)
    root['OUTPUTS']['plasma']['vol'] = vol = np.trapz(np.pi * R**2, Z, axis=1)
    root['OUTPUTS']['plasma']['dVoldrho'] = np.gradient(vol, rho)
