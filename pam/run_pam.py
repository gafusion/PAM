# -*-Python-*-
# Created by mcclenaghanj at 20 Feb 2019  15:18

"""
This script calculates the pellet ablation and radius vs. time for a given pellet
Current this script only supports Deuterium-tritium pellets and soon Neon-deuterium
To make things confusing, the inital formulation with cm while inputs/outputs are in m in materials, density


defaultVars parameters
----------------------
gEQDSK: location of gfile in OMFIT tree
Te_vs_rho (1d array): Electron temperature (keV) profile in radial coordinate rho
ne_vs_rho (1d array): Electron density (10^20 m^-3) profile in radial coordinate rho
rho_profgrid(1d array): rho grid in which the Te and ne are specified on (This has only been tested with uniform grid)
t0 (float): starting time of the simulation(ms)
tf (float): end time of simulation (ms)
dt (float): time step of simulation (ms)
pellet (namelist): location of pellet information(material,thickness, layers, etc) in native (PAM) format
tinj (float): injection time of pellet (ms)
"""

import OMFITlib_functions as func
import OMFITlib_pelletAblationModel as pam
import OMFITlib_pelletInfo as pei
import OMFITlib_plasmaInfo as pli
import OMFITlib_pelletDensityModel as pdm
import OMFITlib_pelletDriftModel as pdrm

import OMFITlib_aurora as pam_aurora
from scipy.constants import N_A
import numpy as np
import scipy
import copy
from pam_globs import *
import pickle

inputs = root['INPUTS']['pam.in']

gEQDSK=root['INPUTS']['gEQDSK']
Te_vs_rho=inputs['equilibrium']['Te']
ne_vs_rho=inputs['equilibrium']['ne']
rho_profgrid=inputs['equilibrium']['rho']
t0=inputs['input']['time_start']
tf=inputs['input']['time_end']
dt=inputs['input']['dt']
tinj=inputs['pellet1']['injection_times'][0]


#root['OUTPUTS'].clear()

Refit = copy.deepcopy(gEQDSK['AuxQuantities']['R'] * 1e2)
Zefit = copy.deepcopy(gEQDSK['AuxQuantities']['Z'] * 1e2)
rr, zz = np.meshgrid(Refit, Zefit)
RHORZ = copy.deepcopy(gEQDSK['AuxQuantities']['RHORZ'])

rho2ds = scipy.interpolate.RectBivariateSpline(Refit, Zefit, RHORZ.T)

# info of each layer from pam.in
nComp = pei.generate_parameter_list('ncomponents')
complist = pei.generate_parameter_list('complist')
modellist = pei.generate_parameter_list('model')
ratiolist = pei.generate_parameter_list('ratiolist')
thicknesslist = pei.generate_parameter_list('thickness')

rLayers, rPellet0 = pei.generate_rlayers(thicknesslist)
R0, Z0, phi0, vR, vZ, vphi = pei.generate_pos_vel()
tinj = pei.generate_tinj()

rho_start = rho2ds.ev(1e2 * R0, 1e2 * Z0)
if any(rho_start < 1):
    print('Warning: Pellet starting inside plasma at rho = ', rho_start)

time = np.arange(t0, tf, dt)

pli.get_plasma_info(time)
pli.generate_RZ_grid()
drhodR, drhodZ = pli.get_interps()

treelocs = func.generate_initial_outputs(time)


rho_alls = np.zeros([len(R0), len(time)])
Rs = 1e2 * np.array([R0 + vR * (t - tinj) * 1e-3 for t in time])
Zs = 1e2 * np.array([Z0 + vZ * (t - tinj) * 1e-3 for t in time])
rhos_all = rho2ds.ev(Rs, Zs)
ipel = 0
for i in range(inputs['input']['numPelletTypes']):
    for j in inputs['pellet' + str(i + 1)]['injection_times']:
        current_pellet = root['OUTPUTS']['pellet' + str(i + 1)][j]
        current_pellet['rho_min'] = np.min(rhos_all[:, ipel])
        ipel += 1

use_aurora = inputs['input']['use_aurora']
if use_aurora:
    pam_aurora.set_aurora(time)


def rpdot(y, t, ipel, it=None, update=False):
    if t > tinj[ipel]:
        R = (R0[ipel] + vR[ipel] * (t - tinj[ipel]) * 1e-3) * 1e2
        Z = (Z0[ipel] + vZ[ipel] * (t - tinj[ipel]) * 1e-3) * 1e2
    else:
        R = R0[ipel] * 1e2
        Z = Z0[ipel] * 1e2

    rho = rho2ds(R, Z)

    out_plasma = root['OUTPUTS']['plasma']

    Gd = Gt = GC = GNe = 0.0
    drpdt = Te = ne = 0

    if rho < out_plasma['rho_transport'][-1]:
        ne = np.interp(rho, out_plasma['rho_transport'], out_plasma['ne_time'][it, :])
        Te = np.interp(rho, out_plasma['rho_transport'], out_plasma['Te_time'][it, :])

    # zero ablation rate outside of the plasma or if it is fully ablated
    if rho < out_plasma['rho_transport'][-1] and y[ipel] > 0:

        # decide which layer is currently at and determine model specified to use
        ilayer = pei.get_ilayer(y, rLayers, ipel)
        model = modellist[ipel][ilayer]

        Bt = 1e2 * abs(gEQDSK['BCENTR']) * gEQDSK['RCENTR'] / R
        Bt_exp = inputs['input']['Bt_exp'] if inputs['input']['btDepend'] else 0

        # P.P. model for deuterium-tritium ablation

        if model == 'dt':
            WD = pei.getAtomicWeight(['d'], [1])
            WT = pei.getAtomicWeight(['t'], [1])
            fracD = pei.getFraction(complist[ipel][ilayer], ratiolist[ipel][ilayer], 'd')
            fracT = pei.getFraction(complist[ipel][ilayer], ratiolist[ipel][ilayer], 't')
            Wavg = fracT * WT + fracD * WD

            drpdt = pam.parks_dt(Bt, Bt_exp, fracD, Te, ne, y[ipel])
            Z_D = Z_T = 1

            pellet_density = pei.getDensity(['d', 't'], [fracD, 1.0 - fracD])
            Gd = -fracD * y[ipel] ** 2 * drpdt * 4 * np.pi * pellet_density * N_A * 1e3 / Wavg * Z_D
            Gt = -fracT * y[ipel] ** 2 * drpdt * 4 * np.pi * pellet_density * N_A * 1e3 / Wavg * Z_T

        # P.P. model for ablation of neon-deuterium
        # Could this be combined with dt model for N-D-T pellet?

        elif model == 'Ned':
            WD = pei.getAtomicWeight(['d'], [1])
            WNe = pei.getAtomicWeight(['Ne'], [1])

            fracD = pei.getFraction(complist[ipel][ilayer], ratiolist[ipel][ilayer], 'd')
            Wavg = (1.0 - fracD) * WNe + fracD * WD

            fracD = pei.getFraction(complist[ipel][ilayer], ratiolist[ipel][ilayer], 'd')
            fracNe = pei.getFraction(complist[ipel][ilayer], ratiolist[ipel][ilayer], 'Ne')
            density = pei.getDensity(['d', 'Ne'], [fracD, 1.0 - fracD])

            Z_D = 1
            Z_Ne = 10

            drpdt = pam.parks_Ned(Bt, Bt_exp, fracD, Te, ne, y[ipel])
            Gd = -fracD * y[ipel] ** 2 * drpdt * 4 * np.pi * density * N_A * 1e3 / Wavg * Z_D
            GNe = -fracNe * y[ipel] ** 2 * drpdt * 4 * np.pi * density * N_A * 1e3 / Wavg * Z_Ne

        elif model == 'C':

            Wavg = pei.getAtomicWeight(['C'], [1])
            fracC = pei.getFraction(complist[ipel][ilayer], ratiolist[ipel][ilayer], 'C')
            drpdt = pam.parks_carbon(Bt, Bt_exp, fracC, Te, ne, y[ipel])

            Z_C = 6
            GC = -fracC * y[ipel] ** 2 * drpdt * 4 * np.pi * pei.getDensity('C', [1.0]) * N_A * 1e3 / Wavg * Z_C

        # if isnan(Gd):
        #     Gd = 0.0
        # if isnan(Gt):
        #    Gt = 0.0
        # if isnan(GC):
        #    GC = 0.0
        # if isnan(GNe):
        #    GNe = 0.0

        elif model == 'pellet_parks':
            drpdt = pam.pellet_parks(y[ipel], Te, ne)

    if update:
        func.compile_ablation_data(treelocs, R, Z, rho, Te, ne, y[ipel], Gd, Gt, GC, GNe, ipel, it)

    return drpdt


def rpdot_all(t, y, it=None, update=False):

    if it > 0 and update:
        if use_aurora:
            # calculate change state evolution and radiated power
            pam_aurora.run_aurora(it)

        # initial value is from the previous step
        out_plasma = root['OUTPUTS']['plasma']
        for key in out_plasma:
            if key.endswith('_time'):
                out_plasma[key][it] = out_plasma[key][it - 1]

        if inputs['input']['update_plasma']:
            # account for density rise and Te drop due to a dilution
            # if use_aurora, also account for radiation looses
            pam_aurora.update_plasma(it)

    # iterate over all pellets
    drpdt = np.zeros_like(y)
    for ipel in range(len(y)):
        drpdt[ipel] = rpdot(y, t, ipel, it=it, update=update)

    if update:
        pdrm.get_shift(drhodR, drhodZ, rho2ds, it)
        pdm.update_densities(drhodR, drhodZ, rho2ds, it)
    return drpdt


def runge_kutta(dydx, y0, x):
    y = y0
    h = x[1] - x[0]
    for ix, x0 in enumerate(x):#, mess=lambda x: f'Iteration: {x[0]}'):

        "Apply Runge Kutta Formulas to find next value of y"

        k1 = h * dydx(x0, y, it=ix, update=True)
        k2 = h * dydx(x0 + 0.5 * h, y + 0.5 * k1, it=ix)
        k3 = h * dydx(x0 + 0.5 * h, y + 0.5 * k2, it=ix)
        k4 = h * dydx(x0 + h, y + k3, it=ix)

        # Update next value of y
        y += (k1 + 2 * k2 + 2 * k3 + k4) / 6.0

        y[y < 0] = 0

    not_started = np.array(np.where(np.equal(y, y0))[0])
    if len(not_started):
        print(f'Error: Pellets {not_started+1} have not reached plasma in the simulated time interval {t0:.1f}-{tf:.1f} ms')

    not_ablated = np.array(np.where(np.divide(y, y0) > 0.05)[0])
    if len(not_ablated):
        print(f'Warning: Pellets {not_ablated+1} were not fully ablated in the simulated time interval {t0:.1f}-{tf:.1f}  ms')


runge_kutta(rpdot_all, rPellet0, time)


with open('out_pam.pkl', 'wb') as f:
    pickle.dump(root['OUTPUTS'], f)

#runid = evalExpr(root['SETTINGS']['EXPERIMENT']['runid'])
#root['RUN_DB'].setdefault(runid, OMFITtree())
#root['RUN_DB'][runid]['INPUTS'] = copy.deepcopy(root['INPUTS'])
#root['RUN_DB'][runid]['OUTPUTS'] = copy.deepcopy(root['OUTPUTS'])
