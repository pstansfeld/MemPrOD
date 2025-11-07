# MemPrOD
MemPrOD: Membrane Protein Orientation and Distortions is a program the predicts the distortions of a lipid bilayer around a membrane protein. The paper associated with this code is available [here]. 

Parameters from the following were used to help write this code:

**Matini forcefield:**
Siewert J ; Marrink et al. “The MARTINI force field The MARTINI Force Field: Coarse Grained Model for Biomolecular Simulations”. In: (2007). [Paper](https://pubs.acs.org/doi/10.1021/jp071097f)

## Installation
To install MemPrOD run the following:
>pip install memprod

This will install MemPrO, Insane4MemPrO and MemPrOD.

Otherwise, clone the GitHub repository. Python 3.11.5 or better is required, and the following packages need to be installed
* Jax 0.4.30 (As JAX is in constant development at the time of writing do not use any other versions, as MemPrOD may no longer work as intended)
>pip install jax\["cpu"\]==0.4.30
* Matplotlib 3.8.4
>pip install matplotlib>=3.8.4
* Scipy 1.11.3
>pip install scipy>=1.11.3

MemPrO is required. Follows the instructions [here](https://github.com/ShufflerBardOnTheEdge/MemPrO) to install MemPrO. MemPrOD should be placed in the same folder as MemPrO.py.

A Martini3 forcefield is also required, this can be downloaded from [here](https://cgmartini.nl/docs/downloads/force-field-parameters/martini3/particle-definitions.html) 

Before running MemPrOD the follwoing environment variable must be set, for Linux users the following lines can be run.

>export PATH_TO_MARTINI=PATH/TO/MARTINI3

## Outputs

MemPrOD is intended to run on either CG or atomostic PDB files and will output the following data:

* Membrane_Data - A folder that contains additional data, mainly .npz files for building CG systems with distortions using Insane4MemPrO.
  
    * All_beads.pdb - A pdb files containing all beads used in calculations. Mostly for debugging.
      
    * Surface.pdb - A pdn file showing the surface beads calculated. Again, mostly for debugging
      
* AcylTail_contacts.pdb - A pdb file of the original input with b-factors as acyl tail contacts. This is still WIP.
  
* HeadGroup_contacts.pdb - As above but with b-factors as head group contacts. This is still WIP.
  
* Deformations.pdb - A pdb file containing a reprensentation of the deformations only.
  
* Deformations_Lower.png - A height map of the deformations in the lower leaflet.
  
* Deformations_Upper.png - As above but for the upper leaflet.
  
* Charge_Lower.png - A graph of the charge in the lower leaflet, calculated after deformations.
  
* Charge_Upper.png - As above but for the upper leaflet.
  
* Potential_Lower.png - A graph of the potential of each membrane segment in the lower leaflet.
  
* Potential_Upper.png - As above but fot the upper leaflet.
  
* Run_command.txt - A text file containg the full run command for later reference.

## Flags
MemPrOD will takes the following flags as input:

-h, --help : This will display a help message with all possible flags and some information on their use.

-f, --file_name : This is the name of the input file for the protein that you wish to predict deformations for. It must be a .pdb. In future support for .gro files may be added. The protein can be either atomistic or coarse grained. The code should detect which it is and ignore unknown atom/bead types, however in the case of an error please let me know and send me the file which caused it. It is recommended that the PDB files had no missing atoms as this can cause the predictions to be of lower quality.

-o, --output : This is the name of the ouput directory. The code will default to a folder called Deformations in the directory where the code is run.

-ni, --iters : Number of minimisation iterations. (Default: 75)

-ng, --grid_density : The spacing between membrane segments. It is highly recommended to use values above 2 as low values can quickly become very expensive. (Default: 3)

-ncav, --no_cavity_surface : This toggles the use of a more expensive but more accurate surface finder. By default the more expensive algorithm is used but this flag will turn it off.

-itp, --itp_file : Path to a Martini3 forcefield. By default this is the enviroment variable PATH_TO_MARTINI

-cut, --cutoff : This is the cutoff for interactions between membrane segments and protein beads. It is not reccommended to change this. (Default: 12)

-k1, --k1_const : This scales the strength of membrane-membrane interactions. It is not reccommended to change this. (Default: 0.25)

-k2, --k2_const : This scales the potential associated with compression and extension of the membrane. This is not generally required and should not be change unless you know what you're doing. (Default: 0)

-zshift, --zshift : This changes the starting Z position of the protein.

-mt, --membrane_thickness : This indicates what the undeformed membrane thickness should be in angstroms. (Default: 38)

-wb, --write_bfactors : Toggles the writing of charge at each membrane segment rather than thinning/thickening.

-res, --additional_residues : A comma seperated list of additional residues present in the input file (eg POPE,POPG)

-res_itp, --additional_residues_itp_file : An itp file containing descriptions of each additional residue added.

-res_cg, --residue_cg_file : A folder containing files called RES.pdb where RES is the name of each additional residue added. RES.pdb should contain coarse graining information, example can be found on the GitHub repo for [CG2AT](https://github.com/owenvickery/cg2at/tree/master/database/fragments/martini_3-0_slipids/non_protein). This is only required for atomistic inputs with additional residues.

## An example

MemPrOD is a very simply code to use, here we will go through a quick example. The first step will be to download an exmaple integral membrane protein, let us choose 4BWZ from the protein data bank. We will be downloading this from MemProtMD rather than the protein data bank. Go to [this page](https://memprotmd.bioch.ox.ac.uk/_ref/PDB/4bwz/_sim/4bwz_default_dppc/) and download "4bwz_default_dppc-head-contacts.pdb" rename this to "4bwz.pdb". Create a folder called "MemPrOD_Example" and place "4bwz.pdb" in this folder. **NOTE: If installing with pip than "MemPrO" and "MemPrOD" can be used to run the code instead of the full path as shown in the example and tutorials below.**

Before we can predict deforamtions we will first need to orient the protein. If you are unfiamilar with MemPrO then take a look at the [MemPrO tutorials](https://github.com/ShufflerBardOnTheEdge/MemPrO/blob/main/MemPrO_tutorials.md). We can orient by running the following command in "MemPrOD_Example".

>python PATH/TO/MemPrO_Script.py -f 4bwz.pdb

This will create a folder called "Orient". To get the oriented protein run the following:

> sed '/DUM/d' ./Orient/Rank_1/oriented_rank_1.pdb > 4bwz-oriented.pdb

This will create a file called "4bwz-oriented.pdb". We can now run the following command:

>python PATH/TO/MemPrOD.py -f 4bwz-oriented.pdb

This will take a few minutes to run and will create a folder called "Deformations". In this folder we can find "Deformations.pdb" Load "4bwz_oriented.pdb" and "Deformations.pdb" in PyMOL. We can improve the clarity of the deformations by running the following command in PyMOL:

>spectrum b, red_white_blue,Deformations,miniumum=-10,maximum=10

We can compare this to the deformations shown on MemProtMD, and we see good agreement in general, with MemPrOD over prediciting thinning on the lower leaflet slightly.

![Alt text](Tutorial_Pics/Fig1.svg)

## FAQ
There are currently no frequently asked questions. If you do have any questions or encounter errors that you cannot fix please contact me via my email m.parrag@warwick.ac.uk and I will do my best to provide help.




