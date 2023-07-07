# mART
<header>
    
_Open source parallel C++ code for computing transition states of classical magnetic systems_
</header>

### Prerequisites
Beyond the standard libraries used in C++, the codes relies on the the Eigen library (https://eigen.tuxfamily.org/index.php?title=Main_Page) for some of the algebra and uses the mpich library (https://www.mpich.org) for the communication between the processes.


### Initial configuration
In the same folder as the code to compile (mART.cpp) must appear the initial configuation file under the name "config.ssv". The format of the latter is space separated variables (.ssv) and corresponds to the text-based XYZ file format of Ovito (https://www.ovito.org/manual/reference/file_formats/input/xyz.html#file-formats-input-xyz). The first line of the header contains the number of particles and the second "Properties=pos:R:3:force:R:3:type:1", which enable later reading by Ovito. The rest of the file corresponds to the x-y-z coordinates of the spins followed by their Sx-Sy-Sz orientation (Sx^2+Sy^2+Sz^2=1) and their type that can be set to 0.


### System parameters (Magnetic Energy Landscape MEL definition)
In the global variables of the mART.cpp, the number of spins (N) as well as their symmetry (D=2: XY and D=3: Heisenberg) must be given. The Hamiltonian is definined by some attributes of the class "MEL". One can define an Hamiltonian with:
- an external uniform magnetic field $\vec{B}$ : $\mathcal{\overline{H}}=-\mu_B g \sum_i \vec{S}_i \cdot \vec{B}$,
- different quadratic anisotropy constants such as $K_x$ (easy-plane when positive) : $\mathcal{\overline{H}}=K_x \sum_i (S_i^x)^2$,
- a quartic anisotropy constant corresponding to the Oh point group $K_{Oh}$ : $\mathcal{\overline{H}}=K_{Oh} \sum_i (S_i^x)^2 (S_i^y)^2+(S_i^x)^2 (S_i^z)^2+(S_i^y)^2 (S_i^z)^2$,
- an exchange interaction constant $J$ (ferromagnetic when positive) : $\mathcal{\overline{H}}=-J \sum_{ \langle i,j \rangle} \vec{S}_i \cdot \vec{S}_j$,
- an SO(2) Dzyaloshinskii–Moriya interaction (DMI) constant D : $\mathcal{\overline{H}}= D \sum_{\langle i,j \rangle} \vec{r}_{ij} \cdot (\vec{S}_i \times \vec{S}_j)$,
- a dipolar interaction constant $\mu_0 \mu_B^2 g^2$ :  $\mathcal{\overline{H}}=-\frac{\mu_0 \mu_B^2 g^2}{4\pi} \sum_{\langle i,j \rangle}\frac{3(\vec{S_i} \cdot \vec{r_{ij}})(\vec{S_{j}} \cdot \vec{r_{ij}})-\vec{S_i} \cdot \vec{S_j}}{|\vec{r_{ij}}|^3}$.
  
The range of these interactions is set by a cut-off (drCritic).

### Algorithm parameters
Also among the attributes of the class MEL, one can define the main algorithm parameters:
- a running time (availableTime) allowing for terminating the algorithm after a preset time while saving the last configuration explored,
- the tolerance in the transverse effective field (tol) to identify the local minima and saddle points,
- the parameter of mART (gammamART) definining the amount of relaxation in the search direction and which should be set close to unity,
- the tolerance in the trust ratio (epsilonMEL) which sets the speed of the exploration and should be small compared to unity,
- the dimension of the Krylov basis for the Lanczos algorithm (krylovDimension) which sets the accuracy in the computation of the lowest eigenmode and eigenvector.

### Perturbation
In the main function of the code, the different types of perturbation to activate mART are commented out. One can select a uniform perturbation, a perturbation on one spin, a perturbation along one of the harmonic modes or a random direction of the global tangent space. 

    
### Compiling
Compiling with mpiCC in -DNDEBUG mode produces the most efficient executable file, that can then be executed in parallel, for instance by mpiexec. 


### Reading outputs
If the tolerance in the transverse effective field is reached in the activation and in the relaxation before the preset running time and no error message pops up, the algorithm has terminated successfully. The working folder should now contain two log files for the activation and relaxation steps and two new additional configuration files: the saddle point and the final local minimum. These files have the same format has the initial configuration file and can be read by Ovito. 



#### Miscellaneous notes:
- Variables containing _sphBasis are described according to the current spherical basis spanning the tangent space. Other variables are described in the D*N Euclidian space. 
- Commented parts of the code enable for periodic boundary conditions in 1D and 2D.  
