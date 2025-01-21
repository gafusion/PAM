# -*-Python-*-
# Created by mcclenaghanj at 19 Nov 2020  14:49
import numpy as np
import scipy
from pam_globs import *

#from classes.utils_fusion import  pedestal_finder
from OMFITlib_pelletInfo import getInitVelocity


def toroidal_drive(Lc0, Sigma0, pfact, tbar):
    """
    This is a function that approximates the toroidal drive from Parks et al. 2000
    by matching Figure 7 & 8

    :param Lc0: Initial cloud length

    :param pfact: Ratio of cloud pressure to background plasma pressure

    :param Sigma0:

    :param tbar: Normalized time (t*Lc0/c_s)

    """

    pfact_exp = 1.69423774
    Sigma0_exp = 0.85
    a1 = 0.52296257 / Lc0 / (Sigma0 ** (Sigma0_exp) * pfact ** (pfact_exp))
    a2 = 17.19090852

    Lc = 1.0 + (Sigma0 ** (Sigma0_exp) * (pfact) ** (pfact_exp)) * tbar

    dPSI = a1 * (np.exp(-a2 * (Lc - Lc0) / Lc0 / pfact ** (2 * pfact_exp) / Sigma0 ** (2 * Sigma0_exp)) - 1 / pfact) * Lc / Lc0 + (1 - a1) * (
        1 - 1 / pfact
    )
    dPSI = max(0.0, dPSI)
    return dPSI


def parks_analytic_shift_model(it, current_pellet, pellet_name, M0, c_perp, rho2ds):

    """
    This function caclulates R-major shift of the pellet from the  gradB drift effect
    based on P. Parks 2000 PoP paper.

    :param it: time step
    :param current_pellet: current pellet location
    :param pellet_name: current pellet name
    :param M0: mach number factor
    :param c_perp: pellet cloud width factor
    :param rho2ds: 2D spline of rho(R,Z)
    """

    vR, vZ, vphi = getInitVelocity(root['INPUTS']['pam.in'][pellet_name])
    rho_pellet = current_pellet['rho'][it]
    R_pellet = current_pellet['R'][it]
    Z_pellet = current_pellet['Z'][it]
    rp = current_pellet['rPellet'] * 1e-2
    mu0 = scipy.constants.mu_0
    mi = scipy.constants.m_p * 2
    echarge = scipy.constants.e
    cm3_to_m3 = 1e6
    rho_eqdsk = root['INPUTS']['gEQDSK']['fluxSurfaces']['geo']['rhon']
    p_eqdsk = root['INPUTS']['gEQDSK']['fluxSurfaces']['avg']['P']
    B_pellet = abs(root['INPUTS']['gEQDSK']['BCENTR']) * root['INPUTS']['gEQDSK']['RCENTR'] / R_pellet
    R_centroid = root['INPUTS']['gEQDSK']['fluxSurfaces']['geo']['R_centroid']
    R_pellet = current_pellet['R'][it]

    rho = root['INPUTS']['pam.in']['equilibrium']['rho']
    ne_inf = current_pellet['ne'][it] * 1e20
    Te_inf = current_pellet['Te'][it] * 1e3
    beta_inf = 4 * mu0 * scipy.constants.e * ne_inf * Te_inf / B_pellet**2
    if ne_inf <= 0 or Te_inf <= 0 or rp[it] <= 0 or len(current_pellet['time'][it:]) < 2:
        return 0, 0, 0

    vA = B_pellet / np.sqrt(mu0 * ne_inf * mi)
    Gabl = 0.0
    for G in ['Gd', 'Gt', 'GC', 'GNe']:
        if G in current_pellet:
            Gabl += current_pellet[G][it]

    T0 = 2.0
    if root['INPUTS']['pam.in']['input']['btDepend']:
        attenuation_factor = (B_pellet) ** 0.7
    else:
        attenuation_factor = 2.0**0.7

    vS = np.sqrt(echarge * T0 / mi)

    rperp = max(rp[0], c_perp * np.sqrt(Gabl**1.0 / ne_inf / (Te_inf) ** 1.5 / attenuation_factor))
    Lc = np.sqrt(R_centroid[0] * rperp)

    loglam = 23.5 - np.log((ne_inf * 1e-6) ** 0.5 / Te_inf ** (5 / 6))

    tau_inf = 2.24882e16 * Te_inf**2 / loglam
    n0 = Gabl / (2 * np.pi * (rperp) ** 2 * vS)
    a1 = 0.01 * M0

    pfact_exp = 1.69423774
    Sigma0_exp = 0.85
    n0 /= a1 * (T0 / ne_inf / Te_inf) ** (pfact_exp) * (Lc / tau_inf) ** (Sigma0_exp)
    n0 = n0 ** (1 / (1 + pfact_exp + Sigma0_exp))

    pressure_factor = T0 * n0 / max(Te_inf * ne_inf, 1e-10)

    Sigma0 = n0 * Lc / tau_inf
    mach = a1 * pressure_factor**pfact_exp * Sigma0**Sigma0_exp

    psiavg = 0.036 * Sigma0**1.1 * (pressure_factor - 1) ** 2.64
    rho_transport = root['OUTPUTS']['plasma']['rho_transport']
    Te_rho = root['OUTPUTS']['plasma']['Te_time'][it, :] * 1e3
    ne_rho = root['OUTPUTS']['plasma']['ne_time'][it, :] * 1e20
    pe_rho = scipy.constants.e * ne_rho * Te_rho

    def vRdot(t, x):

        R0 = x[0]
        Z0 = current_pellet['Z'][it]
        vR0 = x[1]
        tbar = t / Lc * vS

        rho_cloud = rho2ds(1e2 * R0, 1e2 * Z0)[0]
        if rho_cloud > 2.0:
            return [0, 0]

        try:
            peinf = scipy.interpolate.interp1d(rho_transport, pe_rho)(rho_cloud) # does this need to be interp1e????
        except Exception:
            peinf = scipy.constants.e * ne_inf * Te_inf
        peinf = min(scipy.constants.e * n0 * T0, max(scipy.constants.e * ne_inf * Te_inf, peinf))
        pressure_factor = scipy.constants.e * n0 * T0 / peinf

        PSI = toroidal_drive(Lc, Sigma0, pressure_factor, tbar)
        if PSI <= 0:
            return [0.0, 0.0]

        dRdt = vR0
        dvRdt = -2 * B_pellet**2 / vA / scipy.constants.mu_0 * vR0 / (mi * n0 * Lc) + 2 / R0 * PSI * (vS**2 / Lc)

        return [dRdt, dvRdt]

    t_eval = 1e-3 * (current_pellet['time'][it:] - current_pellet['time'][it])
    t_span = [0, (t_eval[-1] - t_eval[0])]
    sol = scipy.integrate.solve_ivp(vRdot, t_span, [R_pellet, vR], t_eval=t_eval, method='Radau')['y']
    Rshift0 = sol[0, -1] - R_pellet
    Rshift_t = np.zeros(len(current_pellet['time']))
    Rshift_t[it:] = sol[0, :] - R_pellet
    if np.isnan(Rshift0) or np.isinf(Rshift0):
        Rshift0 = 0.0

    return Rshift0, rperp, scipy.constants.e * n0 * T0


def update_densities(drhodR, drhodZ, rho2ds, it):
    """
    This function updates the density deposited from all pellets

    :drhodR: 2D spline of drho/dR
    :drhodZ: 2D spline of drho/dZ
    :param rho2ds: 2D spline of rho(R,Z)
    :param it: time step
    """


def get_shift(drhodR, drhodZ, rho2ds, it):
    """
    This function add R-major shift of the pellet

    :param current_pellet: current pellet location
    :param pellet_name: current pellet name
    :drhodR: 2D spline of drho/dR
    :drhodZ: 2D spline of drho/dZ
    :param rho2ds: 2D spline of rho(R,Z)
    :param it: time step
    """

    for i in range(root['INPUTS']['pam.in']['input']['numPelletTypes']):
        pellet_name = 'pellet' + str(i + 1)
        for j in root['INPUTS']['pam.in']['pellet' + str(i + 1)]['injection_times']:
            current_pellet = root['OUTPUTS']['pellet' + str(i + 1)][j]

            ntimes = len(current_pellet['time'])
            pamin = root['INPUTS']['pam.in'][pellet_name]
            model = pamin['Rshift']
            if model == 'PRL ITER scaling':
                current_pellet['shiftR'][it] = getShift_PRL(current_pellet)
            elif model == 'HPI2 scaling':
                current_pellet['shiftR'][it] = getShift_HPI2_scaling(current_pellet, pellet_name)
            elif model == 'Matsuyama':
                current_pellet['shiftR'][it] = getShift_Matsuyama(current_pellet, pellet_name, it)
            elif model == 'Parks simplified':
                cloud_mach = pamin['cloud_mach']
                c_perp = pamin['c_perp']
                (
                    current_pellet['shiftR'][it],
                    current_pellet['rcloud'][it],
                    current_pellet['pecloud'][it],
                ) = parks_analytic_shift_model(it, current_pellet, pellet_name, cloud_mach, c_perp, rho2ds)
            else:
                current_pellet['shiftR'][it] = root['INPUTS']['pam.in'][pellet_name]['Rshift']

    return


def getShift_PRL(current_pellet):

    """
    This function caclulates R-major shift of the pellet from the scaling from the
    shift in ITER using the PRL code in Baylor et al. NF 2007

    :param current_pellet: current pellet location
    """
    rho_transport = root['OUTPUTS']['plasma']['rho_transport']
    Te = root['OUTPUTS']['plasma']['Te_time'][0, :] * 1e3
    Te0 = Te[0]
    if 'Te_ped' in root['OUTPUTS']['plasma']:
        Teped = root['OUTPUTS']['plasma']['Te_ped']
    else:
        root['OUTPUTS']['plasma']['Te_ped'] = Teped = pedestal_finder(Te, rho_transport)[0]

    gEQDSK = root['INPUTS']['gEQDSK']
    Bt = abs(gEQDSK['BCENTR'])
    q95 = abs(np.interp(0.95, gEQDSK['RHOVN'], gEQDSK['QPSI']))
    rp = current_pellet['rPellet'][0]

    shiftR = 0.5 * 0.538 * Bt**-0.147 * Te0**-0.128 * Teped**0.5 * rp**0.764 * q95**-0.149

    return shiftR


def getShift_HPI2_scaling(current_pellet, pellet_name):

    """
    This function caclulates R-major shift of the pellet from the scaling from the
    HPI2 simulations HPI2 scaling HPI2 ( Koechl, Florian, et al. EUROfusion Preprint EFDA-JET-PR (12) 57 (2012).)

    It assumes that rho_min = tangent rho of pellet trajetory

    :param current_pellet: current pellet location
    :param pellet_name: current pellet name
    """
    c1 = 0.116
    c2 = 0.120
    c3 = 0.368
    c4 = 0.041
    c5 = 0.015
    c6 = 1.665
    c7 = 0.439
    c8 = 0.217
    c9 = -0.038
    c10 = 0.493
    c11 = 0.193
    c12 = -0.346
    c13 = -0.204
    c14 = 0.031
    c15 = 28.0

    vR, vZ, vphi = getInitVelocity(root['INPUTS']['pam.in'][pellet_name])
    vp = np.sqrt(vR**2 + vZ**2 + vphi**2)
    rp = current_pellet['rPellet'][0] * 10.0  # cm to mm
    ne0 = root['OUTPUTS']['plasma']['ne_time'][0, 0] * 10  # 1e19 m^-3
    Te0 = root['OUTPUTS']['plasma']['Te_time'][0, 0]
    Raxis = root['OUTPUTS']['plasma']['Rgrid'][0, 0]
    Zaxis = root['OUTPUTS']['plasma']['Zgrid'][0, 0]
    alpha = np.arctan2(current_pellet['Z'][0] - Zaxis, current_pellet['R'][0] - Raxis)
    lam = current_pellet['rho_min']
    gEQDSK = root['INPUTS']['gEQDSK']
    a0 = gEQDSK['fluxSurfaces']['geo']['a'][-1]
    R0 = gEQDSK['RCENTR']
    B0 = gEQDSK['BCENTR']
    kappa = gEQDSK['fluxSurfaces']['geo']['kap'][-1]
    shiftR = c1 * (vp / 100) ** c2 * rp**c3 * ne0**c4 * Te0**c5 * (abs(abs(alpha) - c6) + c8) ** c7
    shiftR *= (1 - lam) ** c9 * a0**c10 * R0**c11 * B0**c12 * kappa**c13

    return shiftR

def getShift_Matsuyama(current_pellet, pellet_name, it):
    Te = [0, 725, 1000, 2000, 5000, 10000, 20000]
    fracNe = 1 - np.array([1, 0.98, 0.9, 0.81, 0])
    Delta_M = [
        [0, 0.297, 0.525, 1.05, 2.055, 3.45, 10],
        [0, 0.073, 0.1, 0.213, 0.5, 0.835, 1.26],
        [0, 0.046, 0.077, 0.17, 0.413, 0.65, 0.92],
        [0, 0.039, 0.065, 0.15, 0.36, 0.57, 0.76],
        [0, 0.0256, 0.04, 0.08, 0.165, 0.232, 0.297],
    ]

    Delta_M_interp = scipy.interpolate.interp2d(Te, fracNe, Delta_M)
    fracNe = root['INPUTS']['pam.in'][pellet_name]['layer1']['ratiolist'][0]
    Te = current_pellet['Te'][it] * 1000
    shiftR = Delta_M_interp(Te, fracNe)

    return 0.5 * shiftR
