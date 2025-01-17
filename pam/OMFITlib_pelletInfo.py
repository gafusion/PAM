# -*-Python-*-
# Created by wuwen at 17 June 2019 7:29PM
import numpy as np
from pam_globs import *

# mass densities
_rho = {'d': 0.2, 't': 0.318, 'Ne': 1.44, 'C': 3.3}

# atomic weights
_aw = {'d': 2.014, 't': 3.016, 'Ne': 20.183, 'C': 12.011}

# atomic charge
_az = {'d': 1, 't': 1, 'Ne': 10, 'C': 6}

avog = 6.022e23  # Avogadros number


def getInitPosition(pellet):
    """
    This function returns initial position of a given pellet

    :param pellet: pellet namelist name from pam.in
    """
    if pellet['pcoordsys'] == 2:
        R0 = pellet['position'][0]
        Z0 = pellet['position'][1]
        phi = pellet['position'][2]
        return R0, Z0, phi


def getInitVelocity(pellet):
    """
    This function returns initial velocity of a given pellet

    :param pellet: pellet namelist name from pam.in
    """
    if root['INPUTS']['pam.in']['pellet1']['pcoordsys'] == 2:
        vR = pellet['velocity'][0]
        vZ = pellet['velocity'][1]
        vphi = pellet['velocity'][2]
        return vR, vZ, vphi


def getAtomicWeight(comp, ratio):
    """
    This function returns atomic weight of mixed specie pellet

    :param comp: list of species

    :param ratio: molar ratio of that specie
    """
    aw = np.asarray([_aw[e] for e in comp if e], dtype=np.float64)
    nr = np.asarray([e for e in ratio if e], dtype=np.float64) / np.sum(ratio)
    return np.sum(np.dot(aw, nr))


def getAtomicZ(comp, ratio):
    """
    This function returns average atomic charge of mixed specie pellet

    :param comp: list of species

    :param ratio: molar ratio of that specie
    """
    az = np.asarray([_az[e] for e in comp if e], dtype=np.float64)
    nr = np.asarray([e for e in ratio if e], dtype=np.float64) / np.sum(ratio)
    return np.sum(np.dot(az, nr))


def getDensity(comp, ratio):
    """
    This function returns average mass density of mixed specie pellet

    :param comp: list of species

    :param ratio: molar ratio of that specie
    """
    try:
        comp[ratio.index(0)] = None
    except Exception:
        pass
    aw = np.asarray([_aw[e] for e in comp if e], dtype=np.float64)
    awbyrho = np.asarray([_aw[e] / _rho[e] for e in comp if e], dtype=np.float64)
    nr = np.asarray([e for e in ratio if e], dtype=np.float64) / np.sum(ratio)
    return np.sum(np.dot(aw, nr)) / np.sum(np.dot(awbyrho, nr))


def getFraction(complist, fraclist, c):
    """
    This fraction returns the fraction of a give species for pellet layer

    :param complist: list of compositions of pellets

    :param fraclist: list of composition fractions of pellets
    """
    if c in complist:
        idx = complist.index(c)
        return fraclist[idx]
    else:
        return 0


def generate_parameter_list(parameter):
    """
    This function generates list various paramters for given pellet
    in pam.in such as ncomponents, complist, model, ratiolist, thickness ''

    :param parameters: name list parameter in pam.in
    """
    pel_list = []
    ipel = 0
    for i in range(root['INPUTS']['pam.in']['input']['numPelletTypes']):
        for j, jval in enumerate(root['INPUTS']['pam.in']['pellet' + str(i + 1)]['injection_times']):
            pel_list.append([])
            pellet = root['INPUTS']['pam.in']['pellet' + str(i + 1)]
            for iLayer in range(1, pellet['nlayers'] + 1):
                layer = 'layer' + str(iLayer)
                if isinstance(pellet[layer][parameter], np.ndarray):
                    pel_list[ipel].append(list(pellet[layer][parameter]))
                else:
                    pel_list[ipel].append(pellet[layer][parameter])

            ipel += 1
    return pel_list


def generate_rlayers(thicknesslist):
    """
    This function an array of pellet layer radii

    :param thicknesslist: array of layer thicknesses
    """
    rLayers = []
    rPellet0 = []
    ipel = 0
    for i in range(root['INPUTS']['pam.in']['input']['numPelletTypes']):

        for j, jval in enumerate(root['INPUTS']['pam.in']['pellet' + str(i + 1)]['injection_times']):
            rLayers.append([])

            rLayers[ipel] = [0]
            for iLayer, rLayer in enumerate(thicknesslist[ipel]):
                rLayers[ipel].append(rLayers[ipel][iLayer] + rLayer)
            rPellet0.append(rLayers[ipel][-1])
            ipel += 1

    return rLayers, rPellet0


def generate_pos_vel():
    """
    This function returns arrays of inital postionsand velocities
    """
    R0 = []
    Z0 = []
    phi0 = []
    vR = []
    vZ = []
    vphi = []
    for i in range(root['INPUTS']['pam.in']['input']['numPelletTypes']):
        pellet_type = root['INPUTS']['pam.in']['pellet' + str(i + 1)]
        for j, jval in enumerate(root['INPUTS']['pam.in']['pellet' + str(i + 1)]['injection_times']):
            R0_tmp, Z0_tmp, phi0_tmp = getInitPosition(pellet_type)
            vR_tmp, vZ_tmp, vphi_tmp = getInitVelocity(pellet_type)

            R0.append(R0_tmp)
            Z0.append(Z0_tmp)
            phi0.append(phi0_tmp)

            vR.append(vR_tmp)
            vZ.append(vZ_tmp)
            vphi.append(vphi_tmp)

    R0 = np.array(R0)
    Z0 = np.array(Z0)
    phi0 = np.array(phi0)

    vR = np.array(vR)
    vZ = np.array(vZ)
    vphi = np.array(vphi)

    return R0, Z0, phi0, vR, vZ, vphi


def generate_tinj():
    """
    This function returns an array of pellet  injection times
    """
    tinj = []
    for i in range(root['INPUTS']['pam.in']['input']['numPelletTypes']):
        for t in root['INPUTS']['pam.in']['pellet' + str(i + 1)]['injection_times']:
            tinj.append(t)
    return tinj


def get_ilayer(y, rLayers, ipel, it=None):
    """
    This function returns the layer index based on current pellet radius
    and radii of the pellet players.

    :param y: current pellet radius
    :param rLayers: radii of pellet layers
    :param ipel: pellet index
    :param it: time index
    """
    nComp = generate_parameter_list('ncomponents')
    pos = y[ipel] - np.array(rLayers[ipel])
    if max(pos) >= 0:
        ilayer = max([i for i in range(len(pos)) if pos[i] >= 0])
    else:
        ilayer = 0
    if ilayer >= len(nComp[ipel]):
        ilayer = len(nComp[ipel]) - 1

    return ilayer
