import numpy as np
import matplotlib.pyplot as plt
from orbband import get_popul, contract_data, horizontal_bar_plot

################## INPUT ###################
INPUT_PATH = 'casscf.chk' # Must be chk or molden. SCF and MCSCF are supported.
ORB_IDX = 'auto' # list of indices or orbitals to plot (0-indexed)
                              # or 'auto' for all active space (for MCSCF)
ORTH_METHOD = 'lowdin' # Supported: 'iao', 'lowdin', 'meta-lowdin'

ATOMS_TO_PLOT = ['Mn1', 'O2', 'C3', 'H4'] # list of atoms with their indices (1-indexed)
FIRST_DIFFUSE = True # If True, first diffuse shells (e.g. Mn 4s+4d, O 3s+3p) are plotted
                     # If False, only valence shells (e.g. Mn 3d, O 2s+2p) are plotted
                     # When ORTH_METHOD=='iao', forced to be False
############################################

popul, nl_labels, orb_idx = get_popul(INPUT_PATH, ORB_IDX, ORTH_METHOD)
popul_ctt, nl_labels_ctt = contract_data(popul, nl_labels, ATOMS_TO_PLOT, get_diffuse=FIRST_DIFFUSE)

# Plot AO contribution for individual orbital
fig = horizontal_bar_plot(popul_ctt, nl_labels_ctt, ylabels=[f'Orb {i+1}' for i in orb_idx])
plt.savefig('result_indiv.png')

# Plot summed AO contribution for the entire active space
fig = horizontal_bar_plot(popul_ctt.sum(axis=0).reshape(1,-1), nl_labels_ctt, ylabels=['Overall'])
plt.savefig('result_summed.png')