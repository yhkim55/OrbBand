from importlib.resources import files
import re
import h5py
import numpy as np
import pandas as pd
from pyscf.tools import molden
from pyscf.lib import chkfile
import matplotlib.pyplot as plt
from colormath.color_conversions import convert_color
from colormath.color_objects import LCHabColor, sRGBColor
from .utils import iao_popul, lowdin_popul
from .periodic_table import ATOMIC_NUMBER

L_VAL = 74
L_DIFF = 81
C_VAL = 44
C_DIFF = 32

def get_color(label:str):
    if label.lower().startswith('ot'):
        return '#C9C9C9'

    atm, nl = label.split()
    atm = re.sub(r'\d', '', atm)
    nl = nl.split('+')[0]

    colorbook_fn = files('orbband').joinpath('jmol_rgb_lch.csv')
    colors_df = pd.read_csv(colorbook_fn)
    hue = colors_df.loc[colors_df['Element']==atm, 'H'].item()
    if nl in get_valence_shell(atm):
        ligh, chr = L_VAL, C_VAL
    else:
        ligh, chr = L_DIFF, C_DIFF
    lch = LCHabColor(ligh, chr, hue)
    rgb = convert_color(lch, sRGBColor)
    rgb.rgb_r = min(rgb.rgb_r, 1.0)
    rgb.rgb_g = min(rgb.rgb_g, 1.0)
    rgb.rgb_b = min(rgb.rgb_b, 1.0)
    return rgb.get_rgb_hex()

def lch2hex(lch):
    lch = LCHabColor(*lch)
    rgb = convert_color(lch,sRGBColor)
    rgb.rgb_r = min(rgb.rgb_r, 1.0)
    rgb.rgb_g = min(rgb.rgb_g, 1.0)
    rgb.rgb_b = min(rgb.rgb_b, 1.0)
    return rgb.get_rgb_hex()

def horizontal_bar_plot(data:np.ndarray, barlabels:list, ylabels=None, 
                          figsize:tuple=None, figtitle:str=None):
    '''
    data: np.ndarray to plot
        for AO population data, data.size=(N_data, N_label)
        N_label is the number of contracted shells to plot, including Others
    barlabels: list of barlabels, length=N_label
    ylabels: list of y labels, np.arange(N_data) if None
    '''
    ndata, nlabel = data.shape
    assert nlabel == len(barlabels)
    
    # Set color dictionary
    colors = [get_color(label) for label in barlabels]
    
    # Set y barlabels
    if ylabels is None:
        ylabels = np.arange(ndata)
    else:
        assert len(ylabels) == ndata
    
    ypos = np.arange(ndata)

    # Create plot
    if figsize is None:
        figsize = (7, ndata * 0.45 + 0.9)
    fig, ax = plt.subplots(figsize=figsize)
    left = np.zeros(ndata)

    # Plot by each component one by one
    for i, (bar_label, c)  in enumerate(zip(barlabels, colors)):
        bar_label_unn = re.sub(r'^\S*?\d+\S*', lambda m: re.sub(r'\d', '', m.group()), bar_label)
        ax.barh(ypos, data[:, i], label=bar_label_unn, height=0.5, color=c, left=left)
        left += data[:, i]
    
    # ylabels
    ax.set_yticks(ypos)
    ax.set_yticklabels(ylabels)
    
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=8)
    ax.set_xlabel('Orbital count')
    if figtitle is not None:   
        ax.set_title(figtitle)
    plt.tight_layout()
    return fig
    
def get_valence_shell(elm):
    # support <= Kr only
    atomnum = ATOMIC_NUMBER[elm]
    if atomnum <= 2:
        return ['1s']
    elif atomnum <= 4:
        return ['2s']
    elif atomnum <= 10:
        return ['2s', '2p']
    elif atomnum <= 12:
        return ['3s']
    elif atomnum <= 18:
        return ['3s', '3p']
    elif atomnum <= 20:
        return ['4s']
    elif atomnum <= 30:
        return ['3d']
    elif atomnum <= 36:
        return ['4s', '4p']
    else:
        return []
    
def get_1st_diffuse_shell(elm):
    # support <= Kr only
    atomnum = ATOMIC_NUMBER[elm]
    if atomnum <= 2:
        return ['2s']
    elif atomnum <= 4:
        return ['2p']
    elif atomnum <= 10:
        return ['3s', '3p']
    elif atomnum <= 12:
        return ['3p']
    elif atomnum <= 18:
        return ['4s', '4p']
    elif atomnum <= 20:
        return ['4p']
    elif atomnum <= 30:
        return ['4d']
    elif atomnum <= 36:
        return ['5s', '5p']
    else:
        return []

def contract_data(popul, nl_labels, atoms_to_plot, get_diffuse=False)->tuple[np.ndarray,list]:
    popul_ctt = []
    nl_labels_ctt = []
    label_pairing = {}
    
    for i, nl_label in enumerate(nl_labels):
        atm, nl = nl_label.split()
        if atm in atoms_to_plot:
            atm_unn = re.sub(r'\d', '', atm) # 'Mn 1' -> 'Mn'
            
            val_shell = get_valence_shell(atm_unn)
            if nl in val_shell:
                label_ctt = atm + ' ' + '+'.join(val_shell)
                if label_ctt in label_pairing:
                    popul_ctt[label_pairing[label_ctt]] += popul[:, i]
                else:
                    label_pairing[label_ctt] = len(popul_ctt)
                    popul_ctt.append(popul[:, i])
                    nl_labels_ctt.append(label_ctt)
            
            if get_diffuse:
                diff_shell = get_1st_diffuse_shell(atm_unn)
                if nl in diff_shell:
                    label_ctt = atm + ' ' + '+'.join(diff_shell)
                    if label_ctt in label_pairing:
                        popul_ctt[label_pairing[label_ctt]] += popul[:, i]
                    else:
                        label_pairing[label_ctt] = len(popul_ctt)
                        popul_ctt.append(popul[:, i])
                        nl_labels_ctt.append(label_ctt)
    
    popul_ctt = np.vstack(popul_ctt).T
    # Add column of Others
    popul_others = np.ones(popul_ctt.shape[0]) - popul_ctt.sum(axis=1)
    popul_ctt = np.hstack([popul_ctt, popul_others.reshape(-1, 1)])
    nl_labels_ctt.append('Others')
    return popul_ctt, nl_labels_ctt

def get_popul(filename, orb_idx, orth_method):
    # Read file
    if filename.endswith('chk'):
        mol = chkfile.load_mol(filename)
        if isinstance(mol.atom, str): # imported xyz
            mol.atom = mol._atom
        with h5py.File(filename) as f:
            if 'scf' in f.keys():
                mode = 'scf'
            elif 'mcscf' in f.keys():
                mode = 'mcscf'
            else:
                raise Exception(f'{filename} is not from scf or mcscf')
            mo_coeff = f[mode]['mo_coeff'][:]
            mo_occ = f[mode]['mo_occ'][:]
    elif filename.endswith('molden'):
        mol, _, mo_coeff, mo_occ, _, _ = molden.load(filename)
        if np.any([occ.is_integer() for occ in mo_occ]):
            mode = 'mcscf'
        else: mode = 'scf'
    else:
        raise Exception(f'{filename} is not a chk or molden file')

    # Process orb_idx
    if isinstance(orb_idx, str): # Auto
        assert mode == 'mcscf'
        orb_idx = np.where((mo_occ > 0) & (mo_occ < 2))[0]
    else:
        orb_idx = np.asarray(orb_idx)

    # Calculate population
    if orth_method.lower() == 'iao':
        pop, nl_labels = iao_popul(mol, mo_coeff, contract_by='nl')
    elif orth_method.lower().startswith('meta'):
        pop, nl_labels = lowdin_popul(mol, mo_coeff, is_meta=True, contract_by='nl')
    else: # Lowdin
        pop, nl_labels = lowdin_popul(mol, mo_coeff, is_meta=False, contract_by='nl')
    # nl label is AO label contracted by two quantum numbers n and l
    # e.g., 2px, 2py, 2pz orbitals of one atom is contracted into 2p

    # Process nl barlabels
    is_numbered = re.search(r'\d', nl_labels[0].split()[1])
    if not is_numbered: # '0 Mn 1s' -> '0 Mn1 1s'
        for i, label in enumerate(nl_labels[:]):
            a, b, c = label.split()
            nl_labels[i] = f'{a} {b}{int(a)+1} {c}'
    nl_labels = [re.sub(r'^\d+', '', l).lstrip() for l in nl_labels] # '0 V1 1s' -> 'V1 1s'
    
    pop = pop[orb_idx, :]
    return pop, nl_labels
