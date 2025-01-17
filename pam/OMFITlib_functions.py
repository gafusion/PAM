# -*-Python-*-
# Created by mcclenaghanj at 15 Feb 2019  21:50
import numpy as np
from pam_globs import *

import OMFITlib_pelletInfo as pei

def compile_ablation_data(treelocs, R, Z, rho, Te, ne, y, Gd, Gt, GC, GNe, ipel, it):
    """
    This function puts pellet ablation info into current pellet tree

    :param treelocs:
    :param R:
    :param Z:
    :param rho:
    :param Te:
    :param ne:
    :param y:
    :param Gd:
    :param Gt:
    :param GC:
    :param GNe:
    :param ipel:
    :param it:
    """
    current_pellet = treelocs[ipel]
    current_pellet['R'][it] = R * 1e-2
    current_pellet['Z'][it] = Z * 1e-2
    current_pellet['rho'][it] = rho
    current_pellet['Te'][it] = Te
    current_pellet['ne'][it] = ne

    current_pellet['rPellet'][it] = y
    if Gd > 0.0:
        current_pellet['Gd'][it] = Gd
    if Gt > 0.0:
        current_pellet['Gt'][it] = Gt
    if GC > 0.0:
        current_pellet['GC'][it] = GC
    if GNe > 0.0:
        current_pellet['GNe'][it] = GNe


def generate_initial_outputs(time):
    """
    This function generate initial output data, and arrays

    :param time: time array
    """
    complist = pei.generate_parameter_list('complist')
    ipel = 0
    treelocs = []
    for i in range(root['INPUTS']['pam.in']['input']['numPelletTypes']):
        root['OUTPUTS']['pellet' + str(i + 1)] = {}#OMFITtree()
        for j in root['INPUTS']['pam.in']['pellet' + str(i + 1)]['injection_times']:
            current_pellet = root['OUTPUTS']['pellet' + str(i + 1)][j] = {}#OMFITtree()

            treelocs.append(current_pellet)

            current_pellet['time'] = time
            ntransport = root['INPUTS']['pam.in']['input']['ntransport']
            ntheta = root['INPUTS']['pam.in']['input']['ntheta']
            species = complist[ipel]
            species = [item for sublist in species for item in sublist]
            for specie in species:
                current_pellet['G' + specie] = np.zeros(len(time))
                current_pellet['n' + specie] = np.zeros(ntransport)

                current_pellet['n' + specie + '_time'] = np.zeros([len(time), ntransport])
                if root['INPUTS']['pam.in']['input']['save2Ddata']:
                    current_pellet['n' + specie + '_2D'] = np.zeros([ntransport, ntheta])

            current_pellet['rho_transport'] = root['OUTPUTS']['plasma']['rho_transport']
            for item in ['R', 'Z', 'Rdep', 'Zdep', 'phi', 'rho', 'Te', 'ne', 'rPellet', 'shiftR', 'rcloud', 'pecloud']:
                current_pellet[item] = np.zeros(len(time))

            ipel += 1

    return treelocs
