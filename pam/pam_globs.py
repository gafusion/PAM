
from omfit_classes.omfit_eqdsk import OMFITgeqdsk
from omfit_classes.omfit_namelist import OMFITnamelist
root = {}
root['OUTPUTS'] = {}
root['INPUTS'] = {}

root['INPUTS']['pam.in'] = OMFITnamelist('pam.in')
root['INPUTS']['gEQDSK'] = OMFITgeqdsk('gEQDSK')

scratch = {}
