# PML<sub>QC</sub> scripts
The python script for training and predictions (using tensorflow) is in [scripts/pml_qc.py](scripts/pml_qc.py)

First, npy files with minibatches were prepared by running a slurm script with the following commands:
```
module load py-tensorflow/1.9.0_py36
python3 pml_qc.py --mode generate_minibatches
```

The PML-QC<sub>DFT</sub> model was trained with the following commands in slurm scripts:
```
module load py-tensorflow/1.9.0_py36
for iepoch in `seq 1 10`; do
  nMLQM=$(expr $iepoch + 0)
  if [ ! -s checkpoint1/PML_QC-${nMLQM}.meta ]; then
    output_file="run.1.Epoch${nMLQM}.out"
    echo "Starting epoch $iepoch" > $output_file
    python3 pml_qc.py --mode train --dataset "../../QM9_Stanford_data/minibatches_split_based_on_3rd_digit_batchsize16" --checkpoint_dir checkpoint1 --epochs 1 --weight_E 0 --weight_wE 0 >> $output_file
  fi
done

wwE_increase_per_epoch=10
cp -r checkpoint1 checkpoint23
nPMLQC_final=$(expr 12 + 24)
while [ ! -s checkpoint23/PML_QC-${nPMLQC_final}.meta ]; do
  iepoch=1
  nPMLQC=$(expr $iepoch + 12)
  while [ -s checkpoint23/PML_QC-${nPMLQC}.meta ]; do
    iepoch=$(expr $iepoch + 1)
    nPMLQC=$(expr $iepoch + 12)
  done
  output_file="run.23.Epoch${nPMLQC}.out"
  wwE=$(echo "scale=2; $iepoch * $wwE_increase_per_epoch" | bc)
  if [ $iepoch -le 12 ]; then
      lr=0.0002
  else
      lr=0.00002
  fi
  echo "Starting epoch $iepoch" > $output_file
  python3 pml_qc.py --mode train --checkpoint_dir checkpoint23 --epochs 1 --weight_E 1e4 --weight_wE $wwE --learning_rate $lr >> $output_file
done

cp -r checkpoint23 checkpoint4
iepoch=25
nPMLQC=$(expr $iepoch + 12)
if [ ! -s checkpoint4/PML_QC-${nPMLQC}.meta ]; then
  output_file="run.4.Epoch${nPMLQC}.out"
  wwE=$(echo "scale=2; $iepoch * $wwE_increase_per_epoch" | bc)
  lr=0.000002
  echo "Starting epoch $iepoch" > $output_file
  python3 pml_qc.py --mode train --checkpoint_dir checkpoint4v6_3rd_digit_9ha.d${wwE_increase_per_epoch} --epochs 1 --weight_E 1e4 --weight_wE $wwE --learning_rate $lr >> $output_file
fi
```
(Different output files were recorded because training was run on pre-emptible nodes, and otherwise they would erase previous output files in the case of restart.)

The PML-QC<sub>CCSD</sub> model was trained with the following commands in slurm scripts:
```
module load py-tensorflow/1.9.0_py36
cp -r checkpoint4 checkpoint_transfer_learning
python3 pml_qc.py --mode transfer_learning --epochs 100 --checkpoint_dir checkpoint_transfer_learning --miniepochs_for_rho_per_epoch 100 --learning_rate 0.0002
python3 pml_qc.py --mode transfer_learning --epochs 100 --checkpoint_dir checkpoint_transfer_learning --miniepochs_for_rho_per_epoch 100 --learning_rate 0.0001
python3 pml_qc.py --mode transfer_learning --epochs 100 --checkpoint_dir checkpoint_transfer_learning --miniepochs_for_rho_per_epoch 100 --learning_rate 0.00002
```

The resulting checkpoint files mentioned in the paper:
  * for PML-QC<sub>DFT</sub> ([tar.gz](checkpoint_files/checkpoint_PML_QC_DFT.tar.gz))
  * for PML-QC<sub>CCSD</sub> ([tar.gz](checkpoint_files/checkpoint_PML_QC_CCSD.tar.gz))

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

In the files with energies, "dE_*QC_method*" stands for (*E* - &Sigma;<sub>*a*</sub>*n*<sub>*a*</sub>*E*<sub>*a*</sub>), where *E* is the electronic energy of the molecule computed with a certain quantum chemical *QC_method*, *n*<sub>*a*</sub> is the number of atoms of element *a* in the molecule, and *E*<sub>*a*</sub> is the energy of an isolated atom *a* computed with the same *QC_method* [compare to Eq. (S22) in the Supplemetary Information]. Note that "dE" differs from the atomization energy by the zero point vibrational energy. Values for the following *QC_method* are given in the energy files above: HF/cc-pVDZ, QM9 [that is, B3LYP/6-31G(2df,p))], and "CCSD(T)/aug-cc-pVQZ" (see SI, Section S2.2). "dE_0DNN_ls_onQM9" is the value of dE that would be predicted with the zero output of the DNN [from HF/cc-pVDZ and the linear correction alone, see Eq. (S23)]; "PML_QC_DFT_correction" and "PML_QC_CCSD_correction" are the predictions of the PML-QC<sub>DFT</sub> and PML-QC<sub>CCSD</sub> models, respectively [&Delta;*E*&prime;, in terms of Eq. (S22)]; and "dE_PML_QC_DFT" and "dE_PML_QC_CCSD" are the values of dE predicted by the PML-QC<sub>DFT</sub> and PML-QC<sub>CCSD</sub> models, respectively. The errors of the PML-QC<sub>DFT</sub> and PML-QC<sub>CCSD</sub> models can be computed for each molecule as the differences (dE_PML_QC_DFT - dE_CCSD(T)/aug-cc-pVQZ_additive_scheme) and (dE_PML_QC_CCSD - dE_CCSD(T)/aug-cc-pVQZ_additive_scheme), respectively. All energies are given in the Hartree units; the conversion factor to kcal/mol is: 1 hartree = 627.509474 kcal/mol.

Electron densities for 52 molecules (indices 1 to 100, with omissions) computed with RI-CCSD/aug-cc-pVQZ, in the format of 64\*64\*64 cube files, are given here: [tar.gz](ccsdt_dataset/RI-CCSD_aug-cc-pVQZ_centered.64x64x64.tar.gz). Other files with the electron densities are too large to be placed on github; they will be added to [Stanford Digital Repository](https://searchworks.stanford.edu/view/kf921gd3855).

## PBE0/pcS-3 dataset (133780 molecules from the QM9 dataset, indices 1 to 133885, with omissions)
This dataset was divided into (see section S4.3 in the Supplementary Information of the manuscript for details):

* **training set** (89432 molecules)
* **validation set** (11191 molecules)
* **test set 1** (11194 molecules)
* **test set 2** (21963 molecules)

The results of PBE0/pcS-3 and HF/cc-pVDZ quantum chemical computations for these molecules (fchk files) and the corresponding minibatches for training, validation and testing (npy files) are too large to be placed on github; they will be added to [Stanford Digital Repository](https://searchworks.stanford.edu/view/kf921gd3855).

# DFT errors
The results of quantum chemical computations (fchk and log files) for the first six molecules in the QM9 dataset, which were used to estimate errors of DFT and other methods in Section S2.1: [tar.gz](high_theory_level_large_basis_set_results/CCSD_with_large_basis_set_results.tar.gz).

Full versions of Tables S3 and S4 (errors of various DFT methods with various basis sets relative to "CCSD(T)/aug-cc-pVQZ" energies and RI-CCSD/aug-cc-pVQZ electron densities) are given here:
* errors in energies: computed in [Q-Chem](DFT_errors/DFT_energy_errors.1to6000.qchem.out) and in [Gaussian](DFT_errors/DFT_energy_errors.1to6000.gaussian.out);
* errors in electron densities: computed in [Q-Chem](DFT_errors/DFT_rho_L1_errors.1to6000.qchem.out) and in [Gaussian](DFT_errors/DFT_rho_L1_errors.1to6000.gaussian.out).
