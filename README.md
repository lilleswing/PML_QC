# PML<sub>QC</sub> datasets
## CCSD(T) dataset (4503 molecules, QM9 indices 1 to 6000, with omissions)
* **training set** (3611 molecules)
  * QM9 indices of molecules ([space-separated](ccsdt_dataset/list_of_space_separated_qm9_indices_ccsdt_train.dat), [comma-separated](ccsdt_dataset/list_of_comma_separated_qm9_indices_ccsdt_train.dat))
  * geometries ([tar.gz](ccsdt_dataset/ccsdt_train_coords.tar.gz), [zip](ccsdt_dataset/ccsdt_train_coords.zip))
  * energies ([space-separated](ccsdt_dataset/energies_ccsdt_train.dat))
* **validation set** (463 molecules)
  * QM9 indices of molecules ([space-separated](ccsdt_dataset/list_of_space_separated_qm9_indices_ccsdt_validation.dat), [comma-separated](ccsdt_dataset/list_of_comma_separated_qm9_indices_ccsdt_validation.dat))
  * geometries ([tar.gz](ccsdt_dataset/ccsdt_validation_coords.tar.gz), [zip](ccsdt_dataset/ccsdt_validation_coords.zip))
  * energies ([space-separated](ccsdt_dataset/energies_ccsdt_validation.dat))
* **test set** (430 molecules)
  * QM9 indices of molecules ([space-separated](ccsdt_dataset/list_of_space_separated_qm9_indices_ccsdt_test.dat), [comma-separated](ccsdt_dataset/list_of_comma_separated_qm9_indices_ccsdt_test.dat))
  * geometries ([tar.gz](ccsdt_dataset/ccsdt_test_coords.tar.gz), [zip](ccsdt_dataset/ccsdt_test_coords.zip))
  * energies ([space-separated](ccsdt_dataset/energies_ccsdt_test.dat))

See also: [energies for all first 6000 molecules](ccsdt_dataset/energies_QM9_indices_1to5999.dat) (with nans); [energies of isolated atoms](ccsdt_dataset/energies_atoms.dat).

In the files with energies, "dE_*QC_method*" stands for (*E* - &Sigma;<sub>*a*</sub>*n*<sub>*a*</sub>*E*<sub>*a*</sub>), where *E* is the electronic energy of the molecule computed with a certain quantum chemical *QC_method*, *n*<sub>*a*</sub> is the number of atoms of element *a* in the molecule, and *E*<sub>*a*</sub> is the energy of an isolated atom *a* computed with the same *QC_method* [compare to Eq. (S22) in the Supplemetary Information]. Note that "dE" differs from the atomization energy by the zero point vibrational energy. Values for the following *QC_method* are given in the energy files above: HF/cc-pVDZ, QM9 [that is, B3LYP/6-31G(2df,p))], and "CCSD(T)/aug-cc-pVQZ" (see SI, Section S2.2). "dE_0DNN_ls_onQM9" is the value of dE that would be predicted with the zero output of the DNN [from HF/cc-pVDZ and the linear correction alone, see Eq. (S23)]; "PML_QC_DFT_correction" and "PML_QC_CCSD_correction" are the predictions of the PML-QC<sub>DFT</sub> and PML-QC<sub>CCSD</sub> models, respectively [&Delta;*E*&prime;, in terms of Eq. (S22)]; and "dE_PML_QC_DFT" and "dE_PML_QC_CCSD" are the values of dE predicted by the PML-QC<sub>DFT</sub> and PML-QC<sub>CCSD</sub> models, respectively. The errors of the PML-QC<sub>DFT</sub> and PML-QC<sub>CCSD</sub> models can be computed for each molecule as the differences (dE_PML_QC_DFT - dE_CCSD(T)/aug-cc-pVQZ_additive_scheme) and (dE_PML_QC_CCSD - dE_CCSD(T)/aug-cc-pVQZ_additive_scheme), respectively. All energies are given in the Hartree units; the conversion factor to kcal/mol is: 1 hartree = 627.509 kcal/mol.
