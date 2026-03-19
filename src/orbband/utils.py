import numpy as np
from pyscf import gto, scf, lo

def ao_contract_by_nl(ao_labels):
    '''
    Contract AO labels with n and l and generate conversion matrix
    
    Example
    Input:
        ao_labels = ['1 O 1s', '1 O 2s', '1 O 2px', '1 O 2py', '1 O 2pz', '2 H 1s']
    Output:
        ao_labels_contracted = ['1 O 1s', '1 O 2s', '1 O 2p', '2 H 1s']
        conversion_matr = [[1 0 0 0 0 0],
                            [0 1 0 0 0 0],
                            [0 0 1 1 1 0],
                            [0 0 0 0 0 1]]
    '''
    import re
    ao_labels_contracted = []
    conversion_matr = []
    for ao in ao_labels:
        ao_cont = re.sub(r'(\S+\s+\S+\s+\S{2})\S*', r'\1', ao).strip()
        if ao_cont in ao_labels_contracted: continue
        ao_labels_contracted.append(ao_cont)
        row = [1 if s.startswith(ao_cont) else 0 for s in ao_labels]
        conversion_matr.append(row.copy())
    return np.array(conversion_matr), ao_labels_contracted

def ao_contract_by_atom(ao_labels):
    '''
    Contract AO labels with atom and generate conversion matrix
    
    Example
    Input:
        ao_labels = ['1 O 1s', '1 O 2s', '1 O 2px', '1 O 2py', '1 O 2pz', '2 H 1s']
    Output:
        ao_labels_contracted = ['1 O', '2 H']
        conversion_matr = [[1 1 1 1 1 0],
                           [0 0 0 0 0 1]]
    '''
    import re
    ao_labels_contracted = []
    conversion_matr = []
    for ao in ao_labels:
        ao_cont = re.sub(r'^\s*(\S+\s+\S+)\s+\S+\s*$', r'\1', ao)
        if ao_cont in ao_labels_contracted: continue
        ao_labels_contracted.append(ao_cont)
        row = [1 if s.startswith(ao_cont) else 0 for s in ao_labels]
        conversion_matr.append(row.copy())
    return np.array(conversion_matr), ao_labels_contracted

def lowdin_popul(mf_or_mol, mo_coeff=None, is_meta=False, contract_by='atom'):
    if isinstance(mf_or_mol, scf.hf.SCF):
        mol = mf_or_mol.mol
        mo_coeff = mo_coeff or mf_or_mol.mo_coeff
    elif isinstance(mf_or_mol, gto.Mole):
        assert mo_coeff is not None
        mol = mf_or_mol
    ao_labels = mol.ao_labels()
    s = mol.intor_symmetric('int1e_ovlp')
    
    if is_meta:
        ao2mlowdin = lo.orth_ao(mol, 'meta-lowdin', s=s)
    else:
        ao2mlowdin = lo.orth_ao(mol, 'lowdin', s=s)

    c_orth = ao2mlowdin.T @ s @ mo_coeff

    # Orthogonality check
    assert np.allclose(mo_coeff.T @ s @ mo_coeff, np.eye(mol.nao))
    assert np.allclose(c_orth.T @ c_orth, np.eye(mol.nao))

    if contract_by is None:
        pop = c_orth ** 2
        contracted_labels = ao_labels
    elif contract_by == 'nl':
        conver_matr, contracted_labels = ao_contract_by_nl(ao_labels)
        pop = conver_matr @ (c_orth ** 2)
    else:
        conver_matr, contracted_labels = ao_contract_by_atom(ao_labels)
        pop = conver_matr @ (c_orth ** 2)
    
    return pop.T, contracted_labels

def iao_popul(mf_or_mol, mo_coeff=None, minao='minao', contract_by='atom'):
    if isinstance(mf_or_mol, scf.hf.SCF):
        mol = mf_or_mol.mol
        mo_coeff = mo_coeff or mf_or_mol.mo_coeff
    elif isinstance(mf_or_mol, gto.Mole):
        assert mo_coeff is not None
        mol = mf_or_mol

    s = mol.intor_symmetric('int1e_ovlp')
    
    a = lo.iao.iao(mol, mo_coeff, minao=minao)
    a = lo.vec_lowdin(a, s)

    iao_orbs = a.T @ s @ mo_coeff
    
    pmol = lo.iao.reference_mol(mol, minao)
    minao_labels = pmol.ao_labels()

    if contract_by is None:
        pop = iao_orbs ** 2
        contracted_labels = minao_labels
    elif contract_by == 'nl':
        conver_matr, contracted_labels = ao_contract_by_nl(minao_labels)
    else:
        conver_matr, contracted_labels = ao_contract_by_atom(minao_labels)
    pop = conver_matr @ (iao_orbs ** 2)
    return pop.T, contracted_labels
