# Fixed-Distance Hamiltonian Monte Carlo

This repository is the anonymous implementation of the 
Fixed-Distance Hamiltonian Monte Carlo algorithm.

#### Requirements
Our code only requires the general libraries 
such as numpy, scipy and tqdm which are installed on most machines. 

To run the code: 

I. make sure that "fixed_distance_hmc" is your working directory:

    cd where/the/code/is/unzipped/fixed_distance_hmc 
    

II. To execute the paper's experiments 
    you should run <code>MAIN_experiments.py</code> which 
   takes two parameters: 
   <code>--model</code> 
   that can take a value in ['MVN', 'FNNL', 'SPECT', 'GrCr', 'AusCr']
   and  <code>--dim</code> that takes an integer value 
    and is only used for the two first models i.e. 'MVN' and 'FNNL'.
    
#### Examples

    python3 MAIN_experiments.py --model FNNL --dim 10 
       
runs the paper's experiment on a 10-dimensional Neal's Funnel model. 

    python3 MAIN_experiments.py --model SPECT
    
 runs the paper's experiment on the Bayesian logistic regression model with SPECT data set.


#### Input data and output (samples)
The input data are in <code>./data/</code> and the generated samples are saved in 
<code>./results/AusCr</code>, <code>./results/FNNL</code>, 
<code>./results/GrCr</code>, <code>./results/MVN</code> and
<code>./results/SPECT</code> depending on the executed model. 
If these directories do not exist, you should build them. 