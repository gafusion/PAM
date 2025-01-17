# -*-Python-*-
# Created by mcclenaghan at 08 Feb 2021  16:16
from pam_globs import *
import numpy as np

import OMFITlib_pelletInfo as pei
from OMFITlib_pelletDriftModel import get_shift


def update_density_2DGaussian(current_pellet, pellet_name, specie, drhodR, drhodZ, rho2ds, it):
    """
    This function adds pellet material onto grid as a 2D Gaussian

    :param current_pellet: current pellet location
    :param pellet_name: current pellet name
    :param specie: particle specie
    :drhodR: 2D spline of drho/dR
    :drhodZ: 2D spline of drho/dZ
    :param rho2ds: 2D spline of rho(R,Z)
    :param it: time step
    """

    dVoldrho = root['OUTPUTS']['plasma']['dVoldrho']
    vol = root['OUTPUTS']['plasma']['vol']
    ntransport = root['INPUTS']['pam.in']['input']['ntransport']
    dt = root['INPUTS']['pam.in']['input']['dt'] * 1e-3
    rho_transport = root['OUTPUTS']['plasma']['rho_transport']
    tgrid = root['INPUTS']['pam.in']['input']['ntheta']

    xpo = root['INPUTS']['pam.in'][pellet_name]['xpo']
    cloudFactorR = root['INPUTS']['pam.in'][pellet_name]['cloudFactorR']
    cloudFactorZ = root['INPUTS']['pam.in'][pellet_name]['cloudFactorZ']
    cloud_delay = root['INPUTS']['pam.in'][pellet_name]['cloud_delay']
    vR, vZ, vphi = pei.getInitVelocity(root['INPUTS']['pam.in'][pellet_name])
    shiftR = current_pellet['shiftR'][it]

    if current_pellet['rcloud'][it] > 0:

        rcloudR = current_pellet['rcloud'][it]
        rcloudZ = current_pellet['rcloud'][it]  #
    else:
        rcloudR = cloudFactorR * current_pellet['rPellet'][0] * 1e-2
        rcloudZ = cloudFactorZ * current_pellet['rPellet'][0] * 1e-2
    pelletpath_R = current_pellet['R'][it] - cloud_delay * rcloudR * vR / np.hypot(vR, vZ)
    pelletpath_Z = current_pellet['Z'][it] - cloud_delay * rcloudZ * vZ / np.hypot(vR, vZ)
    current_pellet['Rdep'][it] = pelletpath_R
    current_pellet['Zdep'][it] = pelletpath_Z

    R = root['OUTPUTS']['plasma']['Rgrid']
    Z = root['OUTPUTS']['plasma']['Zgrid']

    nsource = np.exp(
        -0.5 * ((pelletpath_R - R + 0.5 * shiftR) ** xpo / (rcloudR + 0.25 * shiftR) ** xpo + (pelletpath_Z - Z) ** xpo / rcloudZ**xpo)
    )

    # normalisation is valid only for xpo == 2
    nsource /= (2 * np.pi) ** 2 * (rcloudR + 0.25 * shiftR) * (pelletpath_R + 0.5 * shiftR) * rcloudZ

    nsource *= current_pellet['G' + specie][it]
    modBp = root['OUTPUTS']['plasma']['modBp']
    dl = 1 / modBp[:, 1:] * np.hypot(np.diff(R, axis=1), np.diff(Z, axis=1))

    nablation = np.trapz(nsource[:, 1:] * dl, axis=1) / np.trapz(dl, axis=1)

    if current_pellet['rPellet'][it] > 0:
        current_pellet['n' + specie + '_time'][it, :] = dt * nablation
        current_pellet['n' + specie] += dt * nablation

        if root['INPUTS']['pam.in']['input']['save2Ddata']:
            current_pellet['n' + specie + '_2D'] += dt * nsource


def get_rho_cloud(current_pellet, cloudFactor, R, Z, drhodR, drhodZ):
    """
    This function gets the cloud width in rho space

    :param current_pellet: current pellet location
    :param pellet_name: current pellet name
    :param specie: particle specie
    :drhodR: 2D spline of drho/dR
    :drhodZ: 2D spline of drho/dZ
    :param rho2ds: 2D spline of rho(R,Z)
    :param it: time step
    """
    rcloud = cloudFactor * current_pellet['rPellet'][0] * 1e-2
    drhodR_pelletpath = drhodR(R, Z)
    drhodZ_pelletpath = drhodZ(R, Z)

    # Calculate Gaussian width in rho space
    gradrho = np.sqrt(drhodR_pelletpath**2 + drhodZ_pelletpath**2)
    rhocloud = rcloud * gradrho
    return rhocloud


def update_density_RadialGaussian(current_pellet, pellet_name, specie, drhodR, drhodZ, rho2ds, it):
    """
    This function adds pellet material as aradial Gaussian in rho space

    :param current_pellet: current pellet location
    :param pellet_name: current pellet name
    :param specie: particle specie
    :drhodR: 2D spline of drho/dR
    :drhodZ: 2D spline of drho/dZ
    :param rho2ds: 2D spline of rho(R,Z)
    :param it: time step
    """

    dVoldrho = root['OUTPUTS']['plasma']['dVoldrho']
    ntransport = root['INPUTS']['pam.in']['input']['ntransport']
    dt = root['INPUTS']['pam.in']['input']['dt'] * 1e-3
    rho_transport = root['OUTPUTS']['plasma']['rho_transport']
    cloudFactor = root['INPUTS']['pam.in'][pellet_name]['cloudFactor']
    rho_pellet = current_pellet['rho'][it]
    R = current_pellet['R'][it]
    Z = current_pellet['Z'][it]
    rhocloud = get_rho_cloud(current_pellet, cloudFactor, R, Z, drhodR, drhodZ)

    # ablation = current_pellet['n' + specie]

    gaus = np.exp(-0.5 * (rho_transport - rho_pellet * np.ones(ntransport)) ** 2 / rhocloud**2) / (rhocloud * np.sqrt(2 * np.pi))
    nablation = current_pellet['G' + specie][it] * dt * gaus / dVoldrho

    if rhocloud > 0:
        current_pellet['n' + specie] += nablation


def update_densities(drhodR, drhodZ, rho2ds, it):
    """
    This function updates the density deposited from all pellets

    :drhodR: 2D spline of drho/dR
    :drhodZ: 2D spline of drho/dZ
    :param rho2ds: 2D spline of rho(R,Z)
    :param it: time step
    """

    for i in range(root['INPUTS']['pam.in']['input']['numPelletTypes']):
        pellet_name = 'pellet' + str(i + 1)
        source_model = root['INPUTS']['pam.in'][pellet_name]['source_model']
        if source_model == 'Point':
            print('Density update for Point model is not implemented')
            continue
        update_density_type = eval("update_density_" + source_model)
        for j in root['INPUTS']['pam.in']['pellet' + str(i + 1)]['injection_times']:
            current_pellet = root['OUTPUTS']['pellet' + str(i + 1)][j]
            for item in current_pellet:
                if item[0] == 'G':
                    specie = item[1:]
                    update_density_type(current_pellet, pellet_name, specie, drhodR, drhodZ, rho2ds, it)
