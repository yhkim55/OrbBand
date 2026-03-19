import numpy as np
import matplotlib.pyplot as plt
from orbband import get_popul, contract_data, horizontal_bar_plot

################## INPUT ###################
INPUT_PATH = 'casscf.chk' # Must be chk or molden. SCF and MCSCF are supported.
ORB_IDX = np.arange(135, 146) # list of indices or orbitals to plot (0-indexed)
                              # or 'auto' for all active space (for MCSCF)
ORTH_METHOD = 'lowdin' # Supported: 'iao', 'lowdin', 'meta-lowdin'

ATOMS_TO_PLOT = ['Mn1', 'O2', 'C3', 'H4'] # list of atoms with their indices (1-indexed)
FIRST_DIFFUSE = True # If True, first diffuse shell (e.g. Mn 4s+4d, O 3s+3p) is plotted
                     # If False, only valence shell (e.g. Mn 3d, O 2s+2p) is plotted
                     # When ORTH_METHOD=='iao', forced to be False
############################################

popul, nl_labels = get_popul(INPUT_PATH, ORB_IDX, ORTH_METHOD)
popul_ctt, nl_labels_ctt = contract_data(popul, nl_labels, ATOMS_TO_PLOT, get_diffuse=FIRST_DIFFUSE)

fig = horizontal_bar_plot(popul_ctt, nl_labels_ctt, ylabels=ORB_IDX)
plt.savefig('result.png')