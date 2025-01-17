# -*-Python-*-
# Created by mcclenaghanj at 20 Aug 2019  14:10

"""
This script takes PAM output and input, and converts the data into the OMAS format.
OMAS ODS is created root['OUTPUTS']['ods']
"""

defaultVars(time_index=0, update_eqdata=True, update_pelletdata=True, update_coresources=False, new_sources=True)
# Setup ODS to equilibrium data
if update_eqdata:
    if 'ods' in root['INPUTS']:
        ods = root['OUTPUTS']['ods'] = copy.deepcopy(root['INPUTS']['ods'])
    else:
        ods = root['OUTPUTS']['ods'] = root['INPUTS']['gEQDSK'].to_omas(ods=ODS())
        prof1d = ods['core_profiles']['profiles_1d'][0]
        prof1d['grid']['rho_tor_norm'] = root['INPUTS']['pam.in']['equilibrium']['rho']
        prof1d['electrons']['density_thermal'] = root['INPUTS']['pam.in']['equilibrium']['ne'] * 1.0e20
        prof1d['electrons']['temperature'] = root['INPUTS']['pam.in']['equilibrium']['Te'] * 1.0e3

# Setup ODS for pellet data

if update_pelletdata:
    pellets = root['OUTPUTS']['ods']['pellets'] = ODS()
    pellets['code.parameters'] = CodeParameters()
    pellets['code.parameters']['time_slice'] = {}
    pellets['code.parameters']['time_slice'][time_index] = {}
    pellets['code.parameters']['time_slice'][time_index]['pellet'] = {}

    pelletNum = 0
    pellets['time_slice'][time_index]['time'] = []
    for i in range(root['INPUTS']['pam.in']['input']['numPelletTypes']):

        pellet = 'pellet' + str(i + 1)

        for t, tval in enumerate(root['INPUTS']['pam.in'][pellet]['injection_times']):

            # Save input data
            pelletOdsLoc = pellets['time_slice'][time_index]['pellet'][pelletNum]
            pellets['code.parameters']['time_slice'][time_index]['pellet'][pelletNum] = {}
            cParamPelletOdsLoc = pellets['code.parameters']['time_slice'][time_index]['pellet'][pelletNum]

            pelletOdsLoc['path_geometry']['first_point']['r'] = root['OUTPUTS'][pellet][tval]['R'][0]
            pelletOdsLoc['path_geometry']['second_point']['r'] = root['OUTPUTS'][pellet][tval]['R'][-1]
            pelletOdsLoc['path_geometry']['first_point']['z'] = root['OUTPUTS'][pellet][tval]['Z'][0]
            pelletOdsLoc['path_geometry']['second_point']['z'] = root['OUTPUTS'][pellet][tval]['Z'][-1]
            pelletOdsLoc['path_geometry']['first_point']['phi'] = root['OUTPUTS'][pellet][tval]['phi'][0]
            pelletOdsLoc['path_geometry']['second_point']['phi'] = root['OUTPUTS'][pellet][tval]['phi'][-1]

            vp = np.array(root['INPUTS']['pam.in']['pellet1']['velocity'])
            ods['pellets']['time_slice'][time_index]['pellet'][pelletNum]['velocity_initial'] = np.sqrt(sum(vp**2))

            cParamPelletOdsLoc['source_model'] = root['INPUTS']['pam.in'][pellet]['source_model']
            cParamPelletOdsLoc['Rshift'] = root['INPUTS']['pam.in'][pellet]['Rshift']
            cParamPelletOdsLoc['cloudFactorZ'] = root['INPUTS']['pam.in'][pellet]['cloudFactorZ']
            cParamPelletOdsLoc['cloudFactorR'] = root['INPUTS']['pam.in'][pellet]['cloudFactorR']

            specieNum = 0
            pelletOdsLoc['shape']['size'] = np.zeros(root['INPUTS']['pam.in'][pellet]['nlayers'])
            for ilayer in range(root['INPUTS']['pam.in'][pellet]['nlayers']):
                layer = layer = 'layer' + str(ilayer + 1)
                pelletOdsLoc['shape']['size'][ilayer] = root['INPUTS']['pam.in'][pellet][layer]['thickness']
                for ilabel, label in enumerate(root['INPUTS']['pam.in'][pellet][layer]['complist']):
                    pelletOdsLoc['species'][specieNum]['label'] = label + ' ' + layer
                    pelletOdsLoc['species'][specieNum]['fraction'] = root['INPUTS']['pam.in'][pellet][layer]['ratiolist'][ilabel]
                    specieNum += 1

            pellets['time_slice'][time_index]['time'] = np.append(pellets['time_slice'][time_index]['time'], tval * 1e-3)
            pellets['time'] = root['OUTPUTS'][pellet][tval]['time'] * 1e-3

            # Save OUTPUT data
            pelletOdsLoc['path_profiles']['position']['r'] = root['OUTPUTS'][pellet][tval]['R']
            pelletOdsLoc['path_profiles']['position']['z'] = root['OUTPUTS'][pellet][tval]['Z']
            pelletOdsLoc['path_profiles']['position']['phi'] = root['OUTPUTS'][pellet][tval]['phi']
            pelletOdsLoc['path_profiles']['rho_tor_norm'] = root['OUTPUTS'][pellet][tval]['rho']

            Gtot = np.sum([root['OUTPUTS'][pellet][tval][item] for item in root['OUTPUTS'][pellet][tval] if item[0] == 'G'], axis=0)
            pelletOdsLoc['path_profiles']['ablation_rate'] = Gtot

            pelletNum += 1

if update_coresources:
    if new_sources or 'core_sources.source' not in ods:
        ods['core_sources']['source'] = ODS()

    presource = len(ods['core_sources']['source'])
    for isource in range(root['INPUTS']['pam.in']['input']['numPelletTypes']):
        pellet = 'pellet' + str(isource + 1)
        spellet = np.zeros(len(root['INPUTS']['pam.in']['equilibrium']['rho']))
        isource_ods = isource + presource
        ods['core_sources']['source'][isource_ods]['identifier']['description'] = 'Pellets source calculated from PAM'
        ods['core_sources']['source'][isource_ods]['identifier']['index'] = 14
        ods['core_sources']['source'][isource_ods]['identifier']['name'] = pellet

        for pelletNum in root['OUTPUTS'][pellet]:
            for den in ['nd', 'nt', 'nC', 'nNe']:
                if den in root['OUTPUTS'][pellet][pelletNum]:
                    spellet += root['OUTPUTS'][pellet][pelletNum][den]
        freq = ods['pellets.code.parameters']['time_slice'][time_index]['pellet'][isource]['frequency']

        prof1d = ods['core_sources']['source'][isource_ods]['profiles_1d'][time_index]
        prof1d['grid']['rho_tor_norm'] = root['INPUTS']['pam.in']['equilibrium']['rho']
        prof1d['electrons']['particles'] = spellet * freq
