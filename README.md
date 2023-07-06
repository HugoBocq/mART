# mART
Open source parallel C++ code for computing transition states of classical magnetic systems.

1. INITIAL CONFIGURATION: In the same folder as the code to compile (mART.cpp) must appear the initial configuation file (named "config.ssv"). The format of the latter is space separated variables (.ssv) and corresponds to the text-based XYZ file format of Ovito (https://www.ovito.org/manual/reference/file_formats/input/xyz.html#file-formats-input-xyz). The first line of the header contains the number of particles and the second "Properties=pos:R:3:force:R:3:type:1", which enable later reading by Ovito. The rest of the file corresponds to the x-y-z coordinates of the spins followed by their Sx-Sy-Sz orientation (Sx^2+Sy^2+Sz^2=1) and their type that can be set to 0.

2. SYSTEM PARAMETERS:  In the global variables of the mART.cpp, the number of spins (N) as well as their symmetry (D=2: XY and D=3: Heisenberg) must be given. The Hamiltonian is definined by some attributes of the class "MEL". One can define an Hamiltonian with: an external uniform magnetic field, different quadratic anisotropy constants, a quartic anisotropy constant corresponding to the Oh point group, an exchange interaction, an SO(2) Dzyaloshinskii–Moriya interaction (DMI) and a dipolar interaction. The range of these interactions is set by a cut-off (drCritic). The units are defined by fixing 1 Angstroem, the reduced Planck constant, the mass of an electron and the electron charge to 1. Consequently, one unit of energy is equal to 7.621eV, which is useful to give consistent values for the anisotropy constants, the exchange and the DMI constants for instance.

4. ALGORITHM PARAMETERS: Also in the attributes of the class MEL, one can define the main algorithm parameters: a running time allowing for terminating the algorithm after a preset time while backing up the last configuration explored, the tolerance in the transverse effective field to identify the local minima and saddle points, a parameter definining the amoount of relaxation in the search direction (gammamART) and the tolerance in the trust ratio, wich sets the speed of the exploration.

5. PERTURBATIONS: In the main function of the code, the different types of perturbation are commented out. One can activate mART by selecting a uniform perturbation, a perturbation on one spin, a perturbation along one of the harmonic modes or a random direction of the global tangent space. 
    
6. COMPILING: the code is most efficiently compile with mpiCC in -DNDEBUG mode and can then be executed in parallel, for instance by mpiexec.

7. READING OUTPUTS: if the tolerance in the transverse effective field is reached in the activation and in the relaxation before the preset running time and no error message pops up, the algorithm has terminated successfully. The working folder should now contain two log files for the activation and relaxation steps and two new additional configuration files: the saddle point and the final local minimum. These files have the same format has the initial configuration file and can be read by Ovito. 

Miscellaneous notes:
variables containing _sphBasis are described according to the current spherical basis spanning the tangent space. Other variables are in the D*N Euclidian space. 
The code requires the Eigen3 library from https://eigen.tuxfamily.org/index.php?title=Main_Page.
Commented parts of the code enable for periodic boundary conditions in 1D and 2D.  
