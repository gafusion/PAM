# -*-Python-*-
# Created by mcclenaghan at 22 Mar 2021  11:05
import numpy as np
import scipy
import copy
import aurora
import OMFITlib_pelletInfo as pei
from pam_globs import *


def set_aurora(time):
    """
    This function sets up an initial Aurora run for PAM

    :param time: array of times to simulate

    """

    # read in default Aurora namelist
    pamin = root['INPUTS']['pam.in']
    output = root['OUTPUTS']['aurora'] = {}#OMFITtree()
    namelist = output['namelist'] = aurora.default_nml.load_default_namelist()

    #namelist['device'] = root['SETTINGS']['EXPERIMENT']['device']
    # impurity used to explain Zeff
    imp = pamin['aurora'].get('intrinsic_imp', 'C')
    # main ion
    main_spec = pamin['aurora'].get('main_spec', 'd')
    # even if imp is not injected, it will contribute to increased radiation due to a change in n_e
    species = [main_spec, imp]
    for n in range(pamin['input']['numPelletTypes']):
        for l in range(pamin[f'pellet{n+1}']['nlayers']):
            species.extend(list(pamin[f'pellet{n+1}'][f'layer{l+1}']['complist']))

    species = output['species'] = np.unique(species)

    # Update background every n_rep iterations, each of dt [s] length

    n_step = pamin['aurora']['n_step']
    nr = len(root['OUTPUTS']['plasma']['rhop_transport'])
    dt = pamin['input']['dt'] * 1e-3  # s
    # make ten steps for each pellet source step
    dt_aurora = dt / n_step

    namelist['timing'] = {
        'dt_increase': [1.0, 0],
        'dt_start': [dt_aurora, 0],
        'steps_per_cycle': [1, 0],
        'times': [0.0, dt + dt_aurora],
    }

    namelist['dr_0'] = 1
    namelist['dr_1'] = 0.2

    if pamin['aurora'].get('cxr_flag', False):
        namelist["cxr_flag"] = True

    namelist['source_type'] = 'arbitrary_2d_source'
    namelist["explicit_source_rhop"] = [0, 1]
    namelist["explicit_source_time"] = namelist['timing']['times']
    namelist["explicit_source_vals"] = np.zeros((2, 2))

    equilibrium = pamin['equilibrium']
    rho = equilibrium['rho']
    ne_cm3 = equilibrium['ne'] * 1e14
    Te_eV = equilibrium['Te'] * 1e3
    Zeff = equilibrium['Zeff']

    geqdsk = root['INPUTS']['gEQDSK']
    grhop = np.sqrt(geqdsk['fluxSurfaces']['geo']['psin'])
    grho = geqdsk['fluxSurfaces']['geo']['rhon']

    # extrapolate outside of LCFS
    rhop = np.interp(rho, grho[1:], (grhop / grho)[1:]) * rho

    kp = namelist['kin_profs']
    kp['Te']['rhop'] = kp['ne']['rhop'] = kp['n0']['rhop'] = rhop
    kp['Te']['vals'] = np.maximum(Te_eV, 1)
    kp['ne']['vals'] = np.maximum(ne_cm3, 1)
    kp['n0']['vals'] = np.ones_like(rhop)

    # Background impurity radiation
    output['Prad_prof_init'] = 0

    # Now get aurora setup for each species
    for s in species:
        _s = s[0].upper() + s[1:]
        # atomic data are the same for H,D,T
        if _s in ['D', 'T']:
            _s = 'H'

        namelist_spec = copy.deepcopy(namelist)
        namelist_spec['imp'] = _s

        asim = aurora.core.aurora_sim(namelist_spec, geqdsk=geqdsk)

        scratch['asim_' + s] = asim.save_dict()

        output[f'n_init_' + s] = np.zeros((len(asim.rhop_grid), asim.Z_imp + 1))
        output[f'n_old_' + s] = np.zeros((len(asim.rhop_grid), asim.Z_imp + 1))

        output[f'n{s}_all'] = np.zeros((len(time) * n_step, asim.Z_imp + 1, len(asim.rhop_grid)), dtype=np.single)

        n_init = None
        # assume that Zeff is from background imp
        if s == imp:
            impZ = asim.Z_imp
            n_init = (Zeff - 1) / (impZ * (impZ - 1)) * ne_cm3
            output[f'n{s}_bckg'] = np.interp(asim.rhop_grid, rhop, n_init)

        if s == main_spec:
            n_init = (impZ - Zeff) / (impZ - 1) * ne_cm3
            output[f'n{s}_bckg'] = np.interp(asim.rhop_grid, rhop, n_init)

        output['ioniz_rate_' + s] = np.zeros((len(asim.rhop_grid), asim.Z_imp + 1))  #  ionizations/cm^3/s

        # calculate total radiation from background plasma
        if n_init is not None and pamin['aurora'].get('bckg_rad', True):
            nz = np.zeros((n_step, asim.Z_imp + 1, len(asim.rhop_grid)))
            nz[:, -1] = np.interp(asim.rhop_grid, rhop, n_init)
            rad = aurora.compute_rad(asim.imp, nz, asim.ne, asim.Te, Ti=asim.Te, n0=asim.n0, prad_flag=True)
            output['Prad_prof_init'] = output['Prad_prof_init'] + rad['tot'][0]

    output['rhop'] = copy.deepcopy(asim.rhop_grid)
    output['rvol'] = copy.deepcopy(asim.rvol_grid)

    R0 = geqdsk['RMAXIS'] * 1e2
    output['vol'] = output['rvol'] ** 2 * (2 * np.pi**2 * R0)  # cm^3

    output['aurora_time'] = np.ravel((time / 1e3 - dt / 2 + asim.time_grid[:-1, None]).T)
    output['Prad_tot'] = np.zeros_like(time)
    output['Erad_prof'] = np.zeros((len(time), len(asim.rhop_grid)), dtype=np.single)
    output['Prad_prof'] = np.zeros((len(output['aurora_time']), len(asim.rhop_grid)), dtype=np.single)

    output['ion_source'] = np.zeros(nr)  #  ions/cm^3/s
    output['electron_source'] = np.zeros(nr)  #  electrons/cm^3/s

    output['time'] = time


def run_aurora(it):
    """
    This function runs Aurora run to calculate ionization and radiation for PAM

    :param it: iteration

    """
    pamin = root['INPUTS']['pam.in']
    output = root['OUTPUTS']['aurora']
    plasma = root['OUTPUTS']['plasma']
    dt = pamin['input']['dt'] * 1e-3  # s
    n_step = pamin['aurora']['n_step']

    # get charge state densities from latest time step
    namelist = output['namelist']
    rhop = plasma['rhop_transport']

    kp = namelist['kin_profs']
    kp['ne']['rhop'] = kp['Te']['rhop'] = rhop
    kp['Te']['vals'] = 1e3 * plasma['Te_time'][it - 1]  # eV
    kp['ne']['vals'] = 1e14 * plasma['ne_time'][it - 1]  # cm^-3

    namelist["explicit_source_time"] = namelist['timing']['times']
    namelist["explicit_source_rhop"] = rhop
    rlcfs = np.interp(1, output['rhop'], output['rvol'])

    # guess of transport coefficients
    D = pamin['aurora']['D_z'] * np.ones_like(output['rhop']) * 1e4  # cm^2/s
    # determine V/D to mach input electron density profile
    ne0 = np.interp(output['rhop'], rhop, np.maximum(plasma['ne_time'][0], 1e-5))
    VoverD = np.gradient(np.log(ne0), output['rvol'])  # cm^-1
    V = VoverD * D

    if pamin['aurora'].get('cxr_flag', False):
        # neutral density profile [cm^-3]
        n0 = np.zeros_like(output['rhop'])
        for spec in output['species']:
            if spec in ['h', 'd', 't']:
                n0 += output[f'n_init_' + spec][:, 0]

        if all(n0 < 1):
            namelist["cxr_flag"] = False
        else:
            n0 = np.maximum(1, n0)
            kp['n0'] = {'times': [0], 'rhop': output['rhop'], 'vals': n0, 'fun': 'interpa'}
            namelist["cxr_flag"] = True

    asim = None
    output['ion_source'][:] = 0
    output['electron_source'][:] = 0

    # run AURORA for each species separatelly
    for spec in output['species']:
        spec_source = np.zeros_like(rhop)
        output['ioniz_rate_' + spec][:] = 0
        Z = pei.getAtomicZ([spec], [1])
        for n in range(pamin['input']['numPelletTypes']):
            for t, pellet_t in root['OUTPUTS'][f'pellet{n+1}'].items():
                if f'n{spec}_time' in pellet_t:
                    # neutral impurity source [neutrals/cm^3/s]
                    # it is electron source, not ion source, needs to be divided by charge
                    spec_source += pellet_t[f'n{spec}_time'][it - 1] * 1e-6 / dt / Z

        namelist["explicit_source_vals"] = np.tile(spec_source, (2, 1))

        # initialize aurora
        asim = aurora.aurora_sim(None)
        asim.load_dict(scratch['asim_' + spec])
        namelist['imp'] = asim.imp
        namelist['rvol_lcfs'] = asim.rvol_lcfs

        # set new kin profiles and 2D source
        asim.reload_namelist(namelist)
        asim.kin_profs = kp

        # update source, update ionization rates for new kinetics profiles
        asim.setup_kin_profs_depts()  # slow step

        # when nothing was ablated yet
        if np.any(spec_source) or np.any(output[f'n_init_' + spec] > 0):
            out = asim.run_aurora(D, V, nz_init=output[f'n_init_' + spec])

            # skip first timeslice which is already included as the last pone in previous step
            nz = out[0].T

            if not np.all(np.isfinite(nz)):
                printe('AURORA Calculation Failed')
                #OMFITx.End()

            # calculate the electron particle source as a result of impurity ionization/recombination
            ioniz_source = (asim.Sne_rates - asim.Rne_rates) * nz.T  #  ionizations/cm^3/s

            output['ioniz_rate_' + spec] = ioniz_source.mean(2)
            output['electron_source'] += np.interp(rhop, asim.rhop_grid, output['ioniz_rate_' + spec].sum(1))
            output['ion_source'] += spec_source  #  ions/cm^3/s

            # previous initial value
            output[f'n_old_' + spec] = nz[0].copy().T
            # new initial value
            output[f'n_init_' + spec] = nz[-1].copy().T

            # first timeslice was already included in the previous timestep, skip it
            nz = nz[1:]
            output[f'n{spec}_all'][it * n_step : n_step * (it + 1)] = nz

        else:
            nz = np.zeros_like(output[f'n{spec}_all'][:n_step])

        # add background density of carbon and deuterium
        if pamin['aurora'].get('bckg_rad', True) and f'n{spec}_bckg' in output:
            nz[:, -1] += output[f'n{spec}_bckg'][None]

        # skip radiation calculation if the density is zero
        if not np.any(nz):
            continue

        # calculate radiation, it always reloads the atomic data files :(
        rad = aurora.compute_rad(
            asim.imp, nz, asim.ne, asim.Te, Ti=asim.Te, n0=asim.n0, prad_flag=True, thermal_cx_rad_flag=namelist["cxr_flag"]
        )

        # radiated power in E/cm^3 in AURORA timebase
        output['Prad_prof'][it * n_step : n_step * (it + 1)] += rad['tot']

        if not np.all(np.isfinite(rad['tot'])):
            printe('Radiation Calculation Failed')
            OMFITx.End()

    # substract background radiation before pellet injection
    output['Prad_prof'][it * n_step : n_step * (it + 1)] -= output['Prad_prof_init']

    # radiated energy in J/cm^3
    output['Erad_prof'][it] = output['Prad_prof'][it * n_step : n_step * (it + 1)].mean(0) * dt

    # total radiated power
    output['Prad_tot'][it] = np.trapz(output['Erad_prof'][it], output['vol']) / dt  # [W] volume integrated power


def get_nall(rhop, rhop_grid, new=True):
    """
    This function calculates the total ion and electron density

    :param rhop: rhop on PAM grid
    :param rhop_grid: rhop on Aurora grid
    :param new: if old or new density for this iteration should be used
    """

    ne = np.zeros_like(rhop_grid)  # cm^-3
    nion = np.zeros_like(rhop_grid)  # cm^-3
    out_aurora = root['OUTPUTS']['aurora']

    for spec in root['OUTPUTS']['aurora']['species']:
        if new:
            # new initial value
            nz = out_aurora[f'n_init_' + spec].copy()
        else:
            # previous initial value
            nz = out_aurora[f'n_old_' + spec].copy()

        # add background deuterium and carbon
        if f'n{spec}_bckg' in out_aurora:
            nz[:, -1] += out_aurora[f'n{spec}_bckg']

        # number of electrons from impurity ionization
        ne += np.dot(nz, np.arange(nz.shape[1]))
        # number of ions
        nion += nz.sum(1)

    ne = np.interp(rhop, rhop_grid, ne, right=0)
    nion = np.interp(rhop, rhop_grid, nion, right=0)

    return ne, nion  # cm^-3


def dilution_cooling(rhop, rhop_grid, T_old):
    """
    This function calculates the dilution cooling, assuming constant pressure
    (local energy conservation)

    # NOTE assumes instant thermalization (Ti=Te)
    This calculation is valid only if particle transport is negligible
    compared to parallel heat transport

    :param rhop: rhop on PAM grid
    :param rhop_grid: rhop on Aurora grid
    :param T_old: electron temperature from previous iteration
    """

    ne_old, nion_old = get_nall(rhop, rhop_grid, new=False)
    ne_new, nion_new = get_nall(rhop, rhop_grid, new=True)

    output = root['OUTPUTS']['aurora']
    dt = root['INPUTS']['pam.in']['input']['dt'] / 1e3  # s
    source = output['electron_source'] + output['ion_source']  #  particles/cm^3/s

    T_new = T_old * (ne_old + nion_old) / (ne_old + nion_old + source * dt)

    return ne_new, T_new


def radiation_cooling(rhop, rhop_grid, T, Erad):
    """
    This function calculates radiation cooling
    # NOTE assumes instant thermalization (Ti=Te)

    :param rhop: rhop on PAM grid
    :param rhop_grid: rhop on Aurora grid
    :param T: electron temperature from current iteration
    :param Erad: radation energy in eV/cm^3

    """

    ne, nion = get_nall(rhop, rhop_grid)
    Erad = np.interp(rhop, rhop_grid, Erad)

    T_new = T - 2 * Erad / (3 * (ne + nion))
    return T_new


def ionization_cooling(rhop, rhop_grid, T):
    """
    This function calculates the ionization cooling
    # NOTE assumes instant thermalization (Ti=Te)

    :param rhop: rhop on PAM grid
    :param rhop_grid: rhop on Aurora grid
    :param T: electron temperature from current iteration
    """

    Eion = np.zeros_like(rhop_grid)
    out_aurora = root['OUTPUTS']['aurora']
    dt = root['INPUTS']['pam.in']['input']['dt'] / 1e3  # s
    for specie in root['OUTPUTS']['aurora']['species']:
        if specie == 'Ne':
            energies = [21.5646, 40.96328, 63.45, 97.12, 126.21, 157.93, 207.2759, 239.0989, 1195.8286, 1362.1995]
        elif specie == 'C':
            energies = 0.0104 * np.array([1086.5, 2352.6, 4620.5, 6222.7, 37831, 47277.0])
        elif specie == 'B':
            energies = 0.0104 * np.array([800.6, 2427.1, 3659.7, 25025.8, 32826.7])
        elif specie in ['h', 'd', 't']:
            energies = [13.6]
        else:
            printe(f'Ionization cooling not available for {specie}')
            OMFITx.End()

        cum_ion_energy = np.cumsum(energies)
        Eion += np.dot(out_aurora['ioniz_rate_' + specie][:, :-1], cum_ion_energy) * dt

    ne, nion = get_nall(rhop, rhop_grid)

    Eion = np.interp(rhop, rhop_grid, Eion)

    T_new = T - 2 * Eion / (3 * (ne + nion))
    return T_new


def diffuse(y, D, dt):
    """
    This function calculates diffused profiles

    :param y: variable to diffuse

    :param D: diffusion rate m^2/s

    :param dt: step size

    """

    rmin = 1e2 * root['OUTPUTS']['plasma']['rvol'][-1] * root['OUTPUTS']['plasma']['rho_transport']
    D = D * 100  # cm^2/s
    dvoldp = root['OUTPUTS']['plasma']['dVoldrho']

    def f(t, y):
        dydt = np.zeros(len(y))
        dydt = np.gradient(dvoldp * D * np.gradient(y, rmin), rmin)
        dydt[1:] /= dvoldp[1:]
        dydt[0] = dydt[1]
        dydt[-1] = 0.0
        y[0] = y[1]
        return dydt

    sol = scipy.integrate.solve_ivp(f, (0, dt), y)
    return sol.y[:, -1]


def update_plasma_aurora(it):
    """
    This function updates background plasma calculated from Aurora

    :param it: iteration

    """

    echarge = scipy.constants.e
    plasma = root['OUTPUTS']['plasma']
    Te_eV = plasma['Te_time'][it, :] * 1e3
    ne_cm3 = plasma['ne_time'][it, :] * 1e14
    nd_cm3 = plasma['nd_time'][it, :] * 1e14

    rhop = plasma['rhop_transport']

    pamin = root['INPUTS']['pam.in']
    n_step = pamin['aurora']['n_step']
    dt = pamin['input']['dt'] / 1e3
    out_aurora = root['OUTPUTS']['aurora']
    rhop_grid = out_aurora['rhop']

    # heat diffusion rate
    k_z = pamin['aurora']['k_z']
    # Te_eV = pres0 / (ne_cm3)  # gamma =1?

    Te_eV = diffuse(Te_eV, k_z, dt)

    # radiated energy in eV/cm^3
    Erad = out_aurora['Erad_prof'][it] / echarge

    if pamin['aurora'].get('ioniz_cooling', True):
        # small correction
        Te_eV = ionization_cooling(rhop, rhop_grid, Te_eV)
        Te_eV[Te_eV < 1] = 1

    if pamin['aurora'].get('rad_cooling', True):
        # can be comparable to dilution cooling
        Te_eV = radiation_cooling(rhop, rhop_grid, Te_eV, Erad)
        Te_eV[Te_eV < 1] = 1

    if pamin['aurora'].get('dil_cooling', True):
        # largest effect
        ne_cm3, Te_eV = dilution_cooling(rhop, rhop_grid, Te_eV)
    else:
        ne_cm3, _ = get_nall(rhop, rhop_grid)

    # get updated D density
    nd_cm3 = out_aurora[f'n_init_d'].sum(1)
    if 'nd_bckg' in out_aurora:
        nd_cm3 += out_aurora['nd_bckg']
    nd_cm3 = np.interp(rhop, rhop_grid, nd_cm3)

    if not np.all(np.isfinite(Te_eV)):
        raise OMFITexception('Te is not finite')

    return ne_cm3, nd_cm3, Te_eV


def update_plasma(it):
    """
    This function updates background plasma

    :param it: iteration

    """

    out_plasma = root['OUTPUTS']['plasma']
    pamin = root['INPUTS']['pam.in']['input']

    # account for radiation, ionization and dilution cooling
    if pamin['use_aurora']:
        ne_cm3, nd_cm3, Te_eV = update_plasma_aurora(it)

    # dilution cooling was already included by aurora
    if not pamin['use_aurora']:
        Te_eV = out_plasma['Te_time'][it, :] * 1e3
        ne_cm3 = out_plasma['ne_time'][it, :] * 1e14
        nd_cm3 = out_plasma['nd_time'][it, :] * 1e14

        ne_cm3_0 = np.copy(ne_cm3)
        # Dilution for non impurity deuterium and tritium
        for i in range(pamin['numPelletTypes']):
            for pellet_t in root['OUTPUTS'][f'pellet{i+1}'].values():
                if 'nd_time' in pellet_t:
                    ne_cm3 += 1e-6 * pellet_t['nd_time'][it - 1, :]
                    nd_cm3 += 1e-6 * pellet_t['nd_time'][it - 1, :]
                if 'nt_time' in pellet_t:
                    ne_cm3 += 1e-6 * pellet_t['nt_time'][it - 1, :]

        # pressure remains the same, causing a drop in T
        Te_eV *= ne_cm3_0 / ne_cm3
        Te_eV[Te_eV < 1] = 1

    out_plasma['nd_time'][it, :] = nd_cm3 * 1e-14
    out_plasma['ne_time'][it, :] = ne_cm3 * 1e-14
    out_plasma['Te_time'][it, :] = Te_eV * 1e-3
