Here is the code of my semester project. The report is avaible in the same folder.
On the git only the code files are providied.
On the drive the results are also available: https://drive.google.com/drive/folders/1lL8rq8Rftv-nyLJjJRDN8mQ22H8E-qch?usp=sharing
The folders are organised as follows:

-- alexNet
 |--- alexNet.py: train and evaluate alexNet
 |--- plot_perfs.py: plot the achieved perf
 |--- folders: stock the result of alexNet.py
-- resNet 
 |--- resNet_run.py: train and evaluate resNet
 |--- resnet.py: defines the resNet structure
 |--- plot_perfs.py: plot the achieved perf
 |--- folders: stock the result of resNet_run.py
-- HDNN (Hamiltonian Deep Neural Networks)
 |--- run.py: train and evaluate a specific HDNN
 |--- run_grid.py: grid search over the HDNN parameters
 |--- plot.py: plot the achieved perf of run.py or run_grid.py
 |--- plot_compare.py: compare different perf of run.py
 |--- others .py files: refer to the "code organisation" in the appendix of the report
 |--- folders: stock the results
   |--- TwoLayersPaper: two layers Hamiltonian defined as Chang et al.
   |--- TwoLayersIDS: our enhanced version of the Chang et al. two layers
   |--- J2: J2 HDNN
   |--- J1
     |- ReLU: J2 using ReLU
     |- Tanh: J2 using Tanh

Note: for the folders of the HDNN:
   - when not precised the dataset used is CIFAR10. 
   - _xit at the end indicates how many times the network has been repeated to see stats (most of time 1it)
   - grid_ at the beginning indicates that grid search has been applied to the parameter
   - vanilla indicates that it refers to the network described in the report
     

