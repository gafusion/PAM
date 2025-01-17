# -*-Python-*-
# Created by mcclenaghanj at 20 Aug 2019  09:03

"""
This script sets up PAM inputs from ods file.
"""

defaultVars(time_index=0, ods=root['INPUTS']['ods'], update_eqdata=True, update_pelletdata=True)

# Setup Equilibrium, temperature, and density
if update_eqdata:
    root['INPUTS']['gEQDSK'] = OMFITgeqdsk('gEQDSK').from_omas(ods)

    root['INPUTS']['pam.in']['equilibrium']['rho'] = rho201 = np.linspace(0, 1, 201)
    last_state = len(root['INPUTS']['ods'][u'core_profiles'][u'profiles_1d']) - 1
    rhoODS = ods[u'core_profiles'][u'profiles_1d'][last_state][u'grid'][u'rho_tor_norm']
    root['INPUTS']['pam.in']['equilibrium']['Te'] = np.interp(
        rho201, rhoODS, ods[u'core_profiles'][u'profiles_1d'][last_state][u'electrons'][u'temperature'] * 1.0e-3
    )
    root['INPUTS']['pam.in']['equilibrium']['ne'] = np.interp(
        rho201, rhoODS, ods[u'core_profiles'][u'profiles_1d'][last_state][u'electrons'][u'density_thermal'] * 1.0e-20
    )

# Setup pellet from ods

if 'pellets' in ods and update_pelletdata:
    root['INPUTS']['pam.in']['input']['numPelletTypes'] = len(ods['pellets']['time_slice'][time_index]['pellet'])
    for pelletNum in ods['pellets']['time_slice'][time_index]['pellet']:

        pellet = 'pellet' + str(pelletNum + 1)
        if pellet not in root['INPUTS']['pam.in']:
            root['INPUTS']['pam.in'][pellet] = root['INPUTS']['pam.in']['pellet1']

        root['INPUTS']['pam.in'][pellet]['vcoordsys'] = 2
        root['INPUTS']['pam.in'][pellet]['pcoordsys'] = 2
        root['INPUTS']['pam.in'][pellet]['position'][0] = ods['pellets']['time_slice'][time_index]['pellet'][pelletNum]['path_geometry'][
            'first_point'
        ]['r']
        root['INPUTS']['pam.in'][pellet]['position'][1] = ods['pellets']['time_slice'][time_index]['pellet'][pelletNum]['path_geometry'][
            'first_point'
        ]['z']
        root['INPUTS']['pam.in'][pellet]['position'][2] = ods['pellets']['time_slice'][time_index]['pellet'][pelletNum]['path_geometry'][
            'first_point'
        ]['phi']

        root['INPUTS']['pam.in'][pellet]['nlayers'] = nlayers = len(
            ods['pellets']['time_slice'][time_index]['pellet'][pelletNum]['shape']['size']
        )
        for ilayer in range(1, nlayers + 1):
            layer = 'layer' + str(ilayer)
            if layer not in root['INPUTS']['pam.in'][pellet]:
                root['INPUTS']['pam.in'][pellet][layer] = NamelistName()
            root['INPUTS']['pam.in'][pellet][layer]['ratiolist'] = []
            root['INPUTS']['pam.in'][pellet][layer]['complist'] = []

        for ispecie in root['INPUTS']['ods']['pellets.time_slice'][time_index]['pellet'][pelletNum]['species']:
            if 'layer' in root['INPUTS']['ods']['pellets.time_slice'][time_index]['pellet'][pelletNum]['species'][ispecie]['label']:
                layer = root['INPUTS']['ods']['pellets.time_slice'][time_index]['pellet'][pelletNum]['species'][ispecie]['label'].split()[
                    -1
                ]
            else:
                layer = 'layer1'
            ilayer = int(layer[-1])

            root['INPUTS']['pam.in'][pellet][layer]['complist'].append(
                ods['pellets']['time_slice'][time_index]['pellet'][pelletNum]['species'][ispecie]['label'].split()[0]
            )
            root['INPUTS']['pam.in'][pellet][layer]['ratiolist'].append(
                ods['pellets']['time_slice'][time_index]['pellet'][pelletNum]['species'][ispecie]['fraction']
            )
            root['INPUTS']['pam.in'][pellet][layer]['thickness'] = ods['pellets']['time_slice'][time_index]['pellet'][pelletNum]['shape'][
                'size'
            ][ilayer - 1]

            label = ods['pellets']['time_slice'][0]['pellet'][pelletNum]['species'][ispecie]['label']
            if label[0] == 'C':
                root['INPUTS']['pam.in'][pellet][layer]['model'] = 'C'
            else:
                root['INPUTS']['pam.in'][pellet][layer]['model'] = 'dt'

        if layer == 'layer1':
            root['INPUTS']['pam.in'][pellet]['ncomponents'] = len(
                root['INPUTS']['ods']['pellets.time_slice'][time_index]['pellet'][pelletNum]['species']
            )
        else:
            root['INPUTS']['pam.in'][pellet]['ncomponents'] += len(
                root['INPUTS']['ods']['pellets.time_slice'][time_index]['pellet'][pelletNum]['species']
            )

        # time = ods['pellets']['time']
        # root['INPUTS']['pam.in']['input']['time_start'] = time[0]*1e3
        # root['INPUTS']['pam.in']['input']['time_end'] = time[-1]*1e3
        root['INPUTS']['pam.in'][pellet]['injection_times'] = [ods['pellets']['time'][pelletNum] * 1e3]

        pellet_speed = ods['pellets']['time_slice'][time_index]['pellet'][pelletNum]['velocity_initial']
        path_geometry = ods['pellets']['time_slice'][time_index]['pellet'][pelletNum]['path_geometry']
        dR = path_geometry['second_point']['r'] - path_geometry['first_point']['r']
        dZ = path_geometry['second_point']['z'] - path_geometry['first_point']['z']
        dphi = (
            path_geometry['second_point']['phi'] * path_geometry['second_point']['r']
            - path_geometry['first_point']['phi'] * path_geometry['first_point']['r']
        )
        vunit = np.array([dR, dZ, dphi]) / np.sqrt(dR**2 + dZ**2 + dphi**2)
        root['INPUTS']['pam.in'][pellet]['velocity'] = vunit * pellet_speed

        cParamPelletOdsLoc = ods['pellets']['code.parameters']['time_slice'][time_index]['pellet'][pelletNum]
        for cParam in ['source_model', 'Rshift', 'cloudFactorZ', 'cloudFactorZ']:
            if cParam in cParamPelletOdsLoc:
                root['INPUTS']['pam.in'][pellet][cParam] = cParamPelletOdsLoc[cParam]

else:
    printw('Pellet Data not in ODS\n Skipping pellet setup')
