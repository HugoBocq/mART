#include <iostream> //to run cout and cin
#include <math.h> 
#include <fstream> //module to open and read file
#include <iomanip> //to set equal spacing in log file
#include <mpi.h> //for MPI
#include <limits>
#include <Eigen/Dense> //library for linear algebra
#include <Eigen/Sparse>
//#include <Eigen/unsupported/Eigen/SparseExtra> //to enable .mtx export
#include <complex>
using namespace Eigen;
using namespace std; 
typedef Eigen::Triplet<double> Trip; //to form vector of triplets to fill in sparse matrix

//global variable
int world_size, my_rank; 
int const N=128, D=3;
// for periodic boundary conditions
//int const Nx=0, Ny=0, Nz=0;


//prototypes of utility functions
double norm(double r[], int size);
void splitN2by2(int N, int &my_start, int &my_end);
void splitN(int N, int &my_start, int &my_end, int workloads[], int startInGlobal[]);
void randomOnSphere(double randArray[], int size);

//class defining a system of interacting spins 
class MEL{
	public: 
	//physical constants
	double const mu_0=1.257e-6/3.549e-3; //permeability of free space (in units of h_bar*a_0/e)
       	double const mu_B_times_g=1; //bohr magneton times g (in units of e*h_bar/m_e) (for effective spin contribution: g=2)

	//
	//parameters of the algorithm 
	//
	double availableTime=720; //total time in minutes
	double tol=1e-7; //tolerance in the transverse field to identify local minima and saddle points	
	double gammamART=1; //parameter that dictates the relaxation in mART	
	double epsilonMEL=0.1; //tolerance in the trust ratio 
	int krylovDimension=20; //dimension of the Krylov basis for the Lanczos algorithm

	//
	//parameters of the system MEL (single-ion terms and interaction terms)
	//
	double B[3]={0,0,0}; //external magnetic field (in units of h_bar/e/a_0^2)  
	double uniAnisotropyCstX=0;
	double uniAnisotropyCstY=0;  // easy-plane anisotropies (easy-axis when negative) 
    	double uniAnisotropyCstZ=0;
	double trigonalQuadraticAnisotropyCst=-76.1e-6/7.621; //trigonal quadratic anisotropy for environment set to 1 (leading order anisotropy of D_3h point group)
   	double cubicQuarticAnisotropyCst=-4.97e-6/7.621; //cubic quartic anisotropy for environments set to 0 (leading order anisotropy of O_h point group)
	double Jinter=37.5e-3/7.621; //inter-atomic exchange coupling (in units of h_bar^2/m_e/a_0^2)
	double dmi=0; //Dzyaloshinskii-Moriya strength
 	double const mu_0_times_mu_B2_times_g2=0; //mu_0; //effective constant in dipolar interaction (set to mu_0 to activate dipolar interaction or 0 otherwise)
    	double drCritic=3; // cut-off distance for all the interactions
    
	
	// attributes for storing metadata
	double startTime=0;
	string file_header; //to store the headers of the open file and output the same format
     	
	// attributes for describing the current state of the exploration
       	int environment[N]; //local crystallographic environment (0 is octahedral 0_h, 1 is dihedral D_3h)
       	double alpha, alpha_perturb, hTransverseNorm, hTransverseNormalized_dot_qMinEigen, lambdaMin, lambda2Min, deltaEModel, energy; //adaptative stepsize, constant stepsize during perturbation, norm of the transverse effective field, cosine between the transverse effective field and the softest quadratic mode, softest quadratic mode, second lowest quadratic mode, energy change predicted from effective field, energy
	double b[D*N],h[D*N], position[3*N], spin[D*N], spinRelax[D*N], theta[2*N], sphBasis[9*N], qConvexBasin[D*N], qMinEigen[D*N]; //local external field, effective magnetic field, position, current spins orientations, last relaxed spin configuation, current spins orientations in spherical coordinates, cosines of current spherical basis with e_x,e_y,e_z basis, perturbation in Euclidian space, sofest eigenmode
	SparseMatrix<double,RowMajor> H{D*N, D*N}; //matrix for pairwise interaction in Euclidian space
	SparseMatrix<double,RowMajor> H_accessFormat{N, D*D*N}; //matrix H written with a mapping of elements (l,m)->(l/3, (m/3)*9+(l%3)*3+m%3) allowing for ordered access when creating hessian
	int NNZperRow_H_accessFormat[N]; //number of non zero element in every row of H_accessFormat
	
	VectorXd qMinEigen_sphBasis=VectorXd::Random((D-1)*N); //if random, it needs to be shared among the processes
	SparseMatrix<double,RowMajor> hessian{SparseMatrix<double, RowMajor>((D-1)*N, (D-1)*N)}; // Riemannian Hessian  
       	
	// constructor: load configuration from file and define the Euclidian interaction Matrix
	MEL(string inputFileName) : b(), qMinEigen() //zero the members of the arrays by initialization (not assignment)
     	{
		startTime=MPI_Wtime(); 
		MPI_Bcast(&startTime, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD); //for all the ranks to get the same initial time
 		
		//read in config
		readConfig(inputFileName);
		
		//initialize the softest mode eigenvector witg norm=1 and share it among the processes (if random)
		qMinEigen_sphBasis=qMinEigen_sphBasis/qMinEigen_sphBasis.norm();
		MPI_Bcast(qMinEigen_sphBasis.data(), (D-1)*N, MPI_DOUBLE, 0, MPI_COMM_WORLD); //for all the ranks to get the same random vector
		
		//compute the Euclidian interaction matrix, given the hamiltonian defines from the attributes
		get_H();
	}

	void readConfig(string fileName){
		/*
		read a configuration of the system in XYZ format 
		caution: if not called from the constructor, the interactions/positions must remain unchanged from the initial configuration
		*/
		int Nconfig=0; 
		double trash=0;
		//read input file
		ifstream file; file.open(fileName);
		file >> Nconfig;
		if(N!=Nconfig){
			cout << "FATAL ERROR: the number of atoms in .cpp does not match the number of atoms in the input file";
			exit(1);
		}
		file >> file_header;
		for (int j = 0; j<Nconfig; j++ ){
			file >> position[j*3+0]; 
			file >> position[j*3+1]; 
			file >> position[j*3+2]; 
			file >> spin[j*D+0];
			file >> spin[j*D+1];
			if(D==3) file >> spin[j*D+2];
			if(D==2) file >> trash;
			file >> environment[j];
		}
		file.close();
	}

	void writeConfig(string fileName){
		/*
		output the configuration in the standard XYZ format with the desired fileName
		*/
		ofstream file(fileName); 
		file << N << "\n"; 
		file << file_header << "\n"; 	  	
		for (int i=0; i<N ; i++ ){
			for (int d=0; d<3; d++) file << position[i*3+d] << " ";
			file << spin[i*D] << " ";
			file << spin[i*D+1] << " ";
			if(D==3) file << spin[i*D+2] << " ";
			if(D==2) file << "0" << " ";
			file << environment[i] << " ";
			file << "\n";                
		}
		file.close();
	}

	void writeHessian(string fileName){
		/*
		ouptut the sparse Riemannian hessian in standard format 
		*/
		ofstream file(fileName); 
		file << hessian;
		file.close();	
	}

//	void writeHessian_mtx(string fileName){
//		/*
//		output the sparse Riemannian hessian in .mtx format
//		caution: it requires the eigen3/unsupported library
//		*/
//		Eigen::saveMarket(hessian, fileName); 
//	}

	void writeH(string fileName){
		/*
		output the Euclidian interaction matrix in standard format
		*/
		ofstream file(fileName); 
		file << H;
		file.close();	
	}

	void get_H(){
		/*
		Get the external field b and the Euclidian interaction matrix H such that: E = b*S + 0.5*S*H*S + higher order interactions and anistropies
		*/
		vector<Trip> H_tripletList; //create a vector of triplets
		int my_start, my_end; 
		splitN2by2(N, my_start, my_end);
		
		for(int i=my_start; i<my_end; i++){ 
			//external field
			if(mu_B_times_g!=0){		
				for(int d=0; d<D;d++){
					b[i*D+d]-=mu_B_times_g*B[d];
				}
			}

			//uniaxial anisotropy energy (2 to account for the definition of H in the energy)
			if(uniAnisotropyCstX!=0) H_tripletList.push_back(Trip(i*D, i*D,2*uniAnisotropyCstX));
			if(uniAnisotropyCstY!=0) H_tripletList.push_back(Trip(i*D+1, i*D+1,2*uniAnisotropyCstY));
			if(uniAnisotropyCstZ!=0) H_tripletList.push_back(Trip(i*D+2, i*D+2,2*uniAnisotropyCstZ));

			//quadratic anisotropy contribution for dihedral environment (environment=1) when hexagonal lattice vectors are in 111 planes
			if(environment[i]==1){
				H_tripletList.push_back(Trip(i*D,i*D+1,2.0/3.0*trigonalQuadraticAnisotropyCst));
				H_tripletList.push_back(Trip(i*D+1,i*D,2.0/3.0*trigonalQuadraticAnisotropyCst));
				H_tripletList.push_back(Trip(i*D,i*D+2,2.0/3.0*trigonalQuadraticAnisotropyCst));
				H_tripletList.push_back(Trip(i*D+2,i*D,2.0/3.0*trigonalQuadraticAnisotropyCst));
				H_tripletList.push_back(Trip(i*D+1,i*D+2,2.0/3.0*trigonalQuadraticAnisotropyCst));
				H_tripletList.push_back(Trip(i*D+2,i*D+1,2.0/3.0*trigonalQuadraticAnisotropyCst));
			}
				
			//interaction terms
			for(int j=0; j<i; j++){
				double dr[3]={};
				for(int d=0; d<D;d++) dr[d]=position[j*3+d]-position[i*3+d];
				double dr_norm=norm(dr, 3);
			
				// exchange interactions
				if(Jinter!=0){
					if(dr_norm<drCritic){
						for(int d=0; d<D; d++){
							H_tripletList.push_back(Trip(i*D+d,j*D+d,-Jinter));
							H_tripletList.push_back(Trip(j*D+d,i*D+d,-Jinter));
						}
					}
				}

				// DMI: dzyaloschinskii-Moriya interactions for SO(2) skyrmions (Bloch skyrmions)
				if(dr_norm<drCritic){
					double dmi_vector[3];
					for(int d=0; d<D; d++) dmi_vector[d]=dmi*dr[d]/dr_norm; //if towards -x then sign of dmi_vector is inverted w.r.t. +x, because the cross product is flipped
					if(dmi_vector[2]!=0) H_tripletList.push_back(Trip(i*D,j*D+1,dmi_vector[2]));
					if(dmi_vector[1]!=0) H_tripletList.push_back(Trip(i*D,j*D+2,-dmi_vector[1]));
					if(dmi_vector[2]!=0) H_tripletList.push_back(Trip(i*D+1,j*D,-dmi_vector[2]));
					if(dmi_vector[0]!=0) H_tripletList.push_back(Trip(i*D+1,j*D+2,dmi_vector[0]));
					if(dmi_vector[1]!=0) H_tripletList.push_back(Trip(i*D+2,j*D,dmi_vector[1]));
					if(dmi_vector[0]!=0) H_tripletList.push_back(Trip(i*D+2,j*D+1,-dmi_vector[0]));

					if(dmi_vector[2]!=0) H_tripletList.push_back(Trip(j*D+1,i*D,dmi_vector[2]));
					if(dmi_vector[1]!=0) H_tripletList.push_back(Trip(j*D+2, i*D,-dmi_vector[1]));//symetric w.r.t. i,j and d1 and d2
					if(dmi_vector[2]!=0) H_tripletList.push_back(Trip(j*D, i*D+1,-dmi_vector[2]));
					if(dmi_vector[0]!=0) H_tripletList.push_back(Trip(j*D+2, i*D+1,dmi_vector[0]));
					if(dmi_vector[1]!=0) H_tripletList.push_back(Trip(j*D, i*D+2,dmi_vector[1]));
					if(dmi_vector[0]!=0) H_tripletList.push_back(Trip(j*D+1, i*D+2,-dmi_vector[0]));

				}

				// dipolar interactions
				if(mu_0_times_mu_B2_times_g2!=0){
					// modified dr to take into account periodic boundary conditions in 2D when Nx and Ny are defined
				//	if(Nx-fabs(position[i*3]-position[j*3])<fabs(position[j*3]-position[i*3])){
				//		dr[0]=position[j*3]-position[i*3]-(position[j*3]-position[i*3])/fabs(position[j*3]-position[i*3])*Nx;
				//	}else{
				//		dr[0]=position[j*3]-position[i*3];
				//	}
				//	if(Ny-fabs(position[i*3+1]-position[j*3+1])<fabs(position[j*3+1]-position[i*3+1])){
				//		dr[1]=position[j*3+1]-position[i*3+1]-(position[j*3+1]-position[i*3+1])/fabs(position[j*3+1]-position[i*3+1])*Ny;
				//	}else{
				//		dr[1]=position[j*3+1]-position[i*3+1];
				//	}
				//	double dr_norm=norm(dr,3);
					
					if(dr_norm<drCritic){
						for(int d1=0; d1<D; d1++){
							H_tripletList.push_back(Trip(i*D+d1, j*D+d1,mu_0_times_mu_B2_times_g2*0.25/M_PI/pow(dr_norm,3)));
							H_tripletList.push_back(Trip(j*D+d1, i*D+d1,mu_0_times_mu_B2_times_g2*0.25/M_PI/pow(dr_norm,3)));
							for(int d2=0; d2<D; d2++){
								H_tripletList.push_back(Trip(i*D+d1,j*D+d2,-mu_0_times_mu_B2_times_g2*0.25/M_PI/pow(dr_norm,3)*3/pow(dr_norm,2)*dr[d1]*dr[d2]));
								H_tripletList.push_back(Trip(j*D+d1,i*D+d2,-mu_0_times_mu_B2_times_g2*0.25/M_PI/pow(dr_norm,3)*3/pow(dr_norm,2)*dr[d1]*dr[d2])); //symmetric w.r.t. i,j
						 	}
						} 
					}
				}
			}	
		}

		//introducing PBC on exchange in 1D
//		if(my_rank==0){
//			for(int d=0; d<D; d++){
//				H_tripletList.push_back(Trip(0*D+d,(N-1)*D+d,-Jinter));
//				H_tripletList.push_back(Trip((N-1)*D+d,0*D+d,-Jinter));
//			}	
//		}
//
		//PBC on exchange and DIM in 2D when Nx and Ny are defined
//		if(my_rank==0){ 
//			for(int k=0; k<Nx; k++){
//				for(int d=0; d<D; d++){
//					H_tripletList.push_back(Trip(k*D+d,(N-Nx+k)*D+d,-Jinter));
//					H_tripletList.push_back(Trip((N-Nx+k)*D+d,k*D+d,-Jinter));
//				}
//				double dmi_vector[3]={};
//				dmi_vector[2]=-dmi; //minus because j is the spin at the top of the lattice so that dr=rj-ri is pointing down
//				if(dmi_vector[2]!=0) H_tripletList.push_back(Trip(k*D,(N-Nx+k)*D+1,dmi_vector[2]));
//				if(dmi_vector[1]!=0) H_tripletList.push_back(Trip(k*D,(N-Nx+k)*D+2,-dmi_vector[1]));
//				if(dmi_vector[2]!=0) H_tripletList.push_back(Trip(k*D+1,(N-Nx+k)*D,-dmi_vector[2]));
//				if(dmi_vector[0]!=0) H_tripletList.push_back(Trip(k*D+1,(N-Nx+k)*D+2,dmi_vector[0]));
//				if(dmi_vector[1]!=0) H_tripletList.push_back(Trip(k*D+2,(N-Nx+k)*D,dmi_vector[1]));
//				if(dmi_vector[0]!=0) H_tripletList.push_back(Trip(k*D+2,(N-Nx+k)*D+1,-dmi_vector[0]));
//				//symmetric w.r.t. i,j and d1 and d2 
//				if(dmi_vector[2]!=0) H_tripletList.push_back(Trip((N-Nx+k)*D+1,k*D,dmi_vector[2]));
//				if(dmi_vector[1]!=0) H_tripletList.push_back(Trip((N-Nx+k)*D+2,k*D,-dmi_vector[1])); 
//				if(dmi_vector[2]!=0) H_tripletList.push_back(Trip((N-Nx+k)*D,k*D+1,-dmi_vector[2]));
//				if(dmi_vector[0]!=0) H_tripletList.push_back(Trip((N-Nx+k)*D+2,k*D+1,dmi_vector[0]));
//				if(dmi_vector[1]!=0) H_tripletList.push_back(Trip((N-Nx+k)*D,k*D+2, dmi_vector[1]));
//				if(dmi_vector[0]!=0) H_tripletList.push_back(Trip((N-Nx+k)*D+1,k*D+2,-dmi_vector[0]));
//
//			}
//			for(int k=0; k<Nz; k++){
//				for(int d=0; d<D; d++){
//					H_tripletList.push_back(Trip((k*Nx)*D+d,(k*Nx+Nx-1)*D+d,-Jinter)); 
//					H_tripletList.push_back(Trip((k*Nx+Nx-1)*D+d,(k*Nx)*D+d,-Jinter)); 
//				}
//				double dmi_vector[3]={};
//				dmi_vector[0]=-dmi; //minus because j is the spin at the righthand side of the lattice so that dr=rj-ri is pointing left
//				if(dmi_vector[2]!=0) H_tripletList.push_back(Trip((k*Nx)*D,(k*Nx+Nx-1)*D+1,dmi_vector[2]));
//				if(dmi_vector[1]!=0) H_tripletList.push_back(Trip((k*Nx)*D,(k*Nx+Nx-1)*D+2,-dmi_vector[1]));
//				if(dmi_vector[2]!=0) H_tripletList.push_back(Trip((k*Nx)*D+1,(k*Nx+Nx-1)*D,-dmi_vector[2]));
//				if(dmi_vector[0]!=0) H_tripletList.push_back(Trip((k*Nx)*D+1,(k*Nx+Nx-1)*D+2,dmi_vector[0]));
//				if(dmi_vector[1]!=0) H_tripletList.push_back(Trip((k*Nx)*D+2,(k*Nx+Nx-1)*D,dmi_vector[1]));
//				if(dmi_vector[0]!=0) H_tripletList.push_back(Trip((k*Nx)*D+2,(k*Nx+Nx-1)*D+1,-dmi_vector[0]));
//				//symmetric w.r.t. i,j and d1 and d2 
//				if(dmi_vector[2]!=0) H_tripletList.push_back(Trip((k*Nx+Nx-1)*D+1,(k*Nx)*D,dmi_vector[2]));
//				if(dmi_vector[1]!=0) H_tripletList.push_back(Trip((k*Nx+Nx-1)*D+2,(k*Nx)*D,-dmi_vector[1]));
//				if(dmi_vector[2]!=0) H_tripletList.push_back(Trip((k*Nx+Nx-1)*D,(k*Nx)*D+1,-dmi_vector[2]));
//				if(dmi_vector[0]!=0) H_tripletList.push_back(Trip((k*Nx+Nx-1)*D+2,(k*Nx)*D+1,dmi_vector[0]));
//				if(dmi_vector[1]!=0) H_tripletList.push_back(Trip((k*Nx+Nx-1)*D,(k*Nx)*D+2,dmi_vector[1]));
//				if(dmi_vector[0]!=0) H_tripletList.push_back(Trip((k*Nx+Nx-1)*D+1,(k*Nx)*D+2,-dmi_vector[0]));
//			}
//		}
//
		//commicate and create H
		//communicate the zeeman term of hamiltonian
		MPI_Allreduce(MPI_IN_PLACE,&b, N*D, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		
		//compute a shared array with the size of every triplet list for every core and compute the total number of triplets
		int H_tripletList_size=H_tripletList.size(); 
		int H_tripletList_sizes[world_size];	
		MPI_Allgather(&H_tripletList_size,1,MPI_INT, &H_tripletList_sizes[0], 1, MPI_INT, MPI_COMM_WORLD);
		int H_tripletListShared_size=H_tripletList_sizes[0]; 
		int displ[world_size]={};
		for(int i=1; i<world_size; i++){
			displ[i]=displ[i-1]+H_tripletList_sizes[i-1];
			H_tripletListShared_size+=H_tripletList_sizes[i];
		}

		//split the triplet in each process into three local arrays
		int indices_i[H_tripletList_size];
		int indices_j[H_tripletList_size];
		double values_ij[H_tripletList_size];
		for(int i=0; i<H_tripletList_size; i++){
			indices_i[i]=H_tripletList[i].row(); 
			indices_j[i]=H_tripletList[i].col();
			values_ij[i]=H_tripletList[i].value(); 
		}
		
		//communicate the split triplets in arrays form to obtain shared arrays
		int indicesShared_i[H_tripletListShared_size];
		int indicesShared_j[H_tripletListShared_size];
		double valuesShared_ij[H_tripletListShared_size];
		
		MPI_Allgatherv(&indices_i[0], H_tripletList_sizes[my_rank], MPI_INT, &indicesShared_i[0], &H_tripletList_sizes[0], &displ[0] , MPI_INT, MPI_COMM_WORLD); 
		MPI_Allgatherv(&indices_j[0], H_tripletList_sizes[my_rank], MPI_INT, &indicesShared_j[0], &H_tripletList_sizes[0], &displ[0] , MPI_INT, MPI_COMM_WORLD); 
		MPI_Allgatherv(&values_ij[0], H_tripletList_sizes[my_rank], MPI_DOUBLE, &valuesShared_ij[0], &H_tripletList_sizes[0], &displ[0] , MPI_DOUBLE, MPI_COMM_WORLD);			
		//put back into triplets format for H and H_accessFormat
		vector<Trip> H_tripletListShared;
		vector<Trip> H_accessFormat_tripletListShared; 
		H_tripletListShared.reserve(H_tripletListShared_size);
		H_accessFormat_tripletListShared.reserve(H_tripletListShared_size); 
		for(int i=0; i<H_tripletListShared_size; i++){
			H_tripletListShared.push_back(Trip(indicesShared_i[i],indicesShared_j[i],valuesShared_ij[i]));
			int indexI_accessFormat=indicesShared_i[i]/D;
			int indexJ_accessFormat=(indicesShared_j[i]/D)*D*D+(indicesShared_i[i]%D)*D+(indicesShared_j[i]%D);
			H_accessFormat_tripletListShared.push_back(Trip(indexI_accessFormat, indexJ_accessFormat, valuesShared_ij[i]));

		}		

		//create final sparse matrices
		H.setFromTriplets(H_tripletListShared.begin(), H_tripletListShared.end());
		H_accessFormat.setFromTriplets(H_accessFormat_tripletListShared.begin(), H_accessFormat_tripletListShared.end());

		//counting the number of non-zero element per row in H_access Format
		int *ptrArrStarting=H_accessFormat.outerIndexPtr(); //pointer to array giving the sparse matrix index for the first non zero element of the row
		for(int i=0; i<N; i++){
			int startingNNZIdx=*ptrArrStarting; 
			ptrArrStarting++; 
			int startingNNZIdx_next=*ptrArrStarting;
			NNZperRow_H_accessFormat[i]=startingNNZIdx_next-startingNNZIdx;
		}
	}
    
	void get_sphCoorBasis(){
		/*
		Update the spherical coordinates (global variable: theta) to the one of the actual configuration
		Update the cosines between the spherical coordinate basis vectors {e_x_1, e_y_1, e_z_1, e_x_2,... } (global variable: sphBasis) to the one of the actual configuration
		*/
		for(int i=0; i<N; i++){
			if(D==3) theta[i*2]=acos(spin[i*D+2]); //theta_i
			if(D==2) theta[i*2]=M_PI/2.0; 
			theta[i*2+1]=atan2(spin[i*D+1],spin[i*D]); //phi_i
			//note for atan2: if spin[i*D+1]=spin[i*D]=0, then phi=0, theta=0 => e_theta=e_x, e_phi=e_y and they can be used as tangent vector for the hessian
		
			sphBasis[i*3*3]=cos(theta[i*2])*cos(theta[i*2+1]);//e_theta*e_x
			sphBasis[i*3*3+1]=cos(theta[i*2])*sin(theta[i*2+1]);//e_theta*e_y
			sphBasis[i*3*3+2]=-sin(theta[i*2]);//e_theta*e_z
			sphBasis[i*3*3+3]=-sin(theta[i*2+1]);//e_phi*e_x
			sphBasis[i*3*3+4]=cos(theta[i*2+1]);//e_phi*e_y
			sphBasis[i*3*3+5]=0; //e_phi*e_z 
			sphBasis[i*9+6]=sin(theta[i*2])*cos(theta[i*2+1]); //e_r*e_x
			sphBasis[i*9+7]=sin(theta[i*2])*sin(theta[i*2+1]); //e_r*e_y
			sphBasis[i*9+8]=cos(theta[i*2]); //e_r*e_z 
			
		} 
	}

	void from_Euclidian_to_sphBasis(double *r, double *r_sphBasis){
		/*
		transform the first parameter vector in the D*N Euclidian space into the current spherical basis that spans the tangent space (second parameter)
		Caution: does not update the spherical basis!
		*/
		for(int i=0; i<N; i++){
			if(D==3){
				r_sphBasis[2*i]=r[3*i]*sphBasis[i*9]+r[3*i+1]*sphBasis[i*9+1]+r[3*i+2]*sphBasis[i*9+2]; //F_x*e_theta*e_x+F_y*e_theta*e_y+F_z*e_theta*e_z
				r_sphBasis[2*i+1]=r[3*i]*sphBasis[i*9+3]+r[3*i+1]*sphBasis[i*9+4]; //F_x*e_phi*e_x+F_y*e_phi*e_y
			}else{
				r_sphBasis[i]=r[2*i]*sphBasis[i*9+3]+r[2*i+1]*sphBasis[i*9+4]; //F_x*e_phi*e_x+F_y*e_phi*e_y
			}
		}
	}

	void from_sphBasis_to_Euclidian(double *r_sphBasis, double *r){
		/*
		transform the first parameter vector written in the current spherical basis into a vector of the D*N Euclidian space (second parameter)
		Caution: does not update the spherical basis!
		*/
		for(int i=0; i<N; i++){
			if(D==3){
				r[i*D]=r_sphBasis[i*2]*sphBasis[i*9]+r_sphBasis[i*2+1]*sphBasis[i*9+3]; //F_theta*e_theta_i*e_x_i + F_phi*e_phi_i*e_x_i
				r[i*D+1]=r_sphBasis[i*2]*sphBasis[i*9+1]+r_sphBasis[i*2+1]*sphBasis[i*9+4]; //F_theta*e_theta_i*e_y_i + F_phi*e_phi_i*e_y_i 
				r[i*D+2]=r_sphBasis[i*2]*sphBasis[i*9+2]+r_sphBasis[i*2+1]*sphBasis[i*9+5]; //F_theta*e_theta_i*e_z_i + F_phi*e_phi_i*e_z_i 
			}else{
				r[i*D]=r_sphBasis[i]*sphBasis[i*9+3]; //F_phi*e_phi_i*e_x_i
				r[i*D+1]=r_sphBasis[i]*sphBasis[i*9+4]; //F_phi*e_phi_i*e_y_i 			
			}
		}

	}

	void get_field(){
		/*
		compute h, the effective field in the Euclidian space constituted of the local fields on every spin for the current configuration
		*/

		// split the for loop per process
		int my_start, my_end, workloads[world_size]={}, startInGlobal[world_size]={}; 
		splitN(N, my_start, my_end, workloads, startInGlobal);
		for(int i=0; i<world_size; i++){
			workloads[i]*=D; //to use in MPI_Allgatherv
			startInGlobal[i]*=D; // start position in global vector
		}

		// compute the effective field as the negative of the gradient
		for(int k=my_start*D; k<my_end*D; k++){
			h[k]=-b[k]; //contribution of the external magnetic field to the total magnetic field 
			for(SparseMatrix<double, RowMajor, int>::InnerIterator H_it(H,k); H_it; ++H_it) h[k]-=H_it.value()*spin[H_it.col()];
		}

		MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, h, workloads, startInGlobal, MPI_DOUBLE, MPI_COMM_WORLD);

		//quadratic octahedral anisotropy contribution to the force
		if(cubicQuarticAnisotropyCst!=0){
			for(int i=0; i<N; i++){
				if(environment[i]==0){
					h[3*i]+=-cubicQuarticAnisotropyCst*(2*spin[3*i]*pow(spin[3*i+1],2)+2*spin[3*i]*pow(spin[3*i+2],2));
					h[3*i+1]+=-cubicQuarticAnisotropyCst*(2*spin[3*i+1]*pow(spin[3*i],2)+2*spin[3*i+1]*pow(spin[3*i+2],2));
					h[3*i+2]+=-cubicQuarticAnisotropyCst*(2*spin[3*i+2]*pow(spin[3*i],2)+2*spin[3*i+2]*pow(spin[3*i+1],2));
				}
			}
		}

	}
 
	void fill_hessian_fromEuclidian(int i, int j, MatrixXd& Hij, vector<Trip>& hessian_tripletList){ 
		/*
		fill H_sph, i.e. the Riemannian hessian without the diagonal correction, from Hij, the Euclidian hessian components corresponding to interaction between i and j
		*/
		MatrixXd Ujj(3,2);
		MatrixXd Uii(3,2);
		Ujj << sphBasis[j*9], sphBasis[j*9+3], sphBasis[j*9+1], sphBasis[j*9+4], sphBasis[j*9+2], sphBasis[j*9+5];
		Uii << sphBasis[i*9], sphBasis[i*9+3], sphBasis[i*9+1], sphBasis[i*9+4], sphBasis[i*9+2], sphBasis[i*9+5];
		MatrixXd hessian_ij(2,2); 
		hessian_ij=Uii.transpose()*Hij*Ujj;
		if(D==3){
			hessian_tripletList.push_back(Trip(i*2, j*2, hessian_ij(0,0)));
			hessian_tripletList.push_back(Trip(i*2, j*2+1, hessian_ij(0,1)));
			hessian_tripletList.push_back(Trip(i*2+1, j*2, hessian_ij(1,0)));
			hessian_tripletList.push_back(Trip(i*2+1, j*2+1, hessian_ij(1,1)));
		}else{	
			hessian_tripletList.push_back(Trip(i, j, hessian_ij(1,1)));
		}

	}
 
	void get_hessian(){
		/*
		compute the Riemannian hessian 
		*/
		get_sphCoorBasis();
		
		//construct first H_sph, i.e. the Riemannian hessian without the diagonal correction, and then add diagonal correction
		vector<Trip> H_sph_tripletList; //create a vector of triplets
		int my_start, my_end, workloads[world_size]={}, startInGlobal[world_size]={}; 
		splitN(N, my_start, my_end, workloads, startInGlobal);	
		
		//construct H_sph (2Nx2N) in parallel from H_accessFormat
		for(int i=my_start; i<my_end; i++){

			int countNNZ_i=0; //to count the number of non zero elements already seen in row i
			MatrixXd Hij(3,3); //local matrix of interaction between spin i and j
			Hij.setZero(); 
			int j_buff=0;
			//iterate on the non zero element in row i
			for(SparseMatrix<double, RowMajor, int>::InnerIterator m(H_accessFormat,i); m; ++m){
				int j=m.col()/D/D;
				if((j==j_buff || countNNZ_i==0) && (countNNZ_i+1)!=NNZperRow_H_accessFormat[i]){ //same interaction, not the last non-zero element in the row
					Hij((m.col()%(D*D))/D,((m.col()%(D*D))%D))=m.value(); //update local matrix
				}

				if((j==j_buff || countNNZ_i==0) && (countNNZ_i+1)==NNZperRow_H_accessFormat[i]){ //same interaction, the last non-zero element in the row (the loop ends)
					Hij((m.col()%(D*D))/D,((m.col()%(D*D))%D))=m.value(); //update local matrix	
					fill_hessian_fromEuclidian(i,j,Hij,H_sph_tripletList); //fill H_sph
					
				}

				if(j!=j_buff && countNNZ_i!=0 &&  (countNNZ_i+1)!=NNZperRow_H_accessFormat[i]){ //new interaction pair, not the last non-zero element in the row
					fill_hessian_fromEuclidian(i,j_buff,Hij,H_sph_tripletList); //fill H_sph
					Hij.setZero(); 
					Hij((m.col()%(D*D))/D,((m.col()%(D*D))%D))=m.value(); //update local matrix

				}

				if(j!=j_buff && countNNZ_i!=0 &&  (countNNZ_i+1)==NNZperRow_H_accessFormat[i]){ //new interaction pair, the last non-zero term in the row (the loop ends)
					fill_hessian_fromEuclidian(i,j_buff,Hij,H_sph_tripletList); //fill H_sph
					Hij.setZero(); 	
					Hij((m.col()%(D*D))/D,((m.col()%(D*D))%D))=m.value(); // update local matrix
					fill_hessian_fromEuclidian(i,j,Hij,H_sph_tripletList); //fill H_sph
				}
				j_buff=j;
				countNNZ_i++; 
			}
		}
		
		//compute a shared array with the size of every triplet list for every core and compute the total number of triplets
		int H_sph_tripletList_size=H_sph_tripletList.size();
		int H_sph_tripletList_sizes[world_size];	
		MPI_Allgather(&H_sph_tripletList_size,1,MPI_INT, &H_sph_tripletList_sizes[0], 1, MPI_INT, MPI_COMM_WORLD);
		int H_sph_tripletListShared_size=H_sph_tripletList_sizes[0]; 
		int displ[world_size]={};
		for(int i=1; i<world_size; i++){
			displ[i]=displ[i-1]+H_sph_tripletList_sizes[i-1];
			H_sph_tripletListShared_size+=H_sph_tripletList_sizes[i];
		}

		//split the triplet in each process into three local arrays
		int indices_i[H_sph_tripletList_size];
		int indices_j[H_sph_tripletList_size];
		double values_ij[H_sph_tripletList_size];
		for(int i=0; i<H_sph_tripletList_size; i++){
			indices_i[i]=H_sph_tripletList[i].row(); 
			indices_j[i]=H_sph_tripletList[i].col();
			values_ij[i]=H_sph_tripletList[i].value(); 
		}
		
		//communicate the split triplets in arrays form to obtain shared arrays
		int indicesShared_i[H_sph_tripletListShared_size];
		int indicesShared_j[H_sph_tripletListShared_size];
		double valuesShared_ij[H_sph_tripletListShared_size];

		
		MPI_Allgatherv(&indices_i[0], H_sph_tripletList_sizes[my_rank], MPI_INT, &indicesShared_i[0], &H_sph_tripletList_sizes[0], &displ[0] , MPI_INT, MPI_COMM_WORLD); 
		MPI_Allgatherv(&indices_j[0], H_sph_tripletList_sizes[my_rank], MPI_INT, &indicesShared_j[0], &H_sph_tripletList_sizes[0], &displ[0] , MPI_INT, MPI_COMM_WORLD); 
		MPI_Allgatherv(&values_ij[0], H_sph_tripletList_sizes[my_rank], MPI_DOUBLE, &valuesShared_ij[0], &H_sph_tripletList_sizes[0], &displ[0] , MPI_DOUBLE, MPI_COMM_WORLD);
	
		//obtain the field projected along e_r to add the diagonal correction to the Riemannian hessian 
		get_field(); 
		double h_sphBasis_r[N];
		if(D==3){
			for(int i=0; i<N; i++) h_sphBasis_r[i]=h[i*3]*sphBasis[i*9+6]+h[i*3+1]*sphBasis[i*9+7]+h[i*3+2]*sphBasis[i*9+8];
		}else{
			for(int i=0; i<N; i++) h_sphBasis_r[i]=h[i*2]*sphBasis[i*9+6]+h[i*2+1]*sphBasis[i*9+7];
		}
		
		// construct hessian constituted of H_sph basis and the diagonal correction
		// add the correction for the octahedral environment (quartic anisotropy) if environment=0
		vector<Trip> hessian_tripletList;
		for(int i=0; i<N; i++){
			if(D==3){
				hessian_tripletList.push_back(Trip(i*2, i*2, h_sphBasis_r[i]));
				hessian_tripletList.push_back(Trip(i*2+1, i*2+1, h_sphBasis_r[i]));
	
				if(environment[i]==0 && cubicQuarticAnisotropyCst!=0){ //octahedral environment
					MatrixXd Ti(3,3); //local matrix of interaction between spin i and j
					for(int n=0; n<3; n++){
						for(int m=0; m<3; m++){
							if(m==n){
								Ti(n,m)=2*cubicQuarticAnisotropyCst*(pow(spin[(i*3+n+1)%3],2)+pow(spin[(i*3+n+2)%3],2));
							}else{
								Ti(n,m)=2*cubicQuarticAnisotropyCst*2*spin[i*3+n]*spin[i*3+m];
							}
						}
					}
					fill_hessian_fromEuclidian(i, i, Ti, hessian_tripletList);
				}
			}else{
				hessian_tripletList.push_back(Trip(i, i, h_sphBasis_r[i]));
			}
		}
		for(int i=0; i<H_sph_tripletListShared_size; i++){
			hessian_tripletList.push_back(Trip(indicesShared_i[i], indicesShared_j[i], valuesShared_ij[i]));
		}

		//create the sparse matrix containing the Riemannian hessian
		hessian.setFromTriplets(hessian_tripletList.begin(), hessian_tripletList.end()); //heavy command!	
	}

	void get_eigenmodes(double *eigenvalues, double eigenModes[D*N][(D-1)*N]){
		/*
 		find all eigenmodes of the Riemannian hessian and output them in Euclidian space
 		*/
		get_hessian();
		SelfAdjointEigenSolver<MatrixXd> es(hessian);
		for(int lambda=0; lambda<(D-1)*N; lambda++){
			eigenvalues[lambda]=(es.eigenvalues())(lambda); 
		}
			
		for(int lambda=0; lambda<(D-1)*N; lambda++){
			double q_sphBasis[(D-1)*N];
			for(int i=0; i<(D-1)*N; i++) q_sphBasis[i]=(es.eigenvectors().col(lambda))(i);
			double eigenMode[D*N];
			from_sphBasis_to_Euclidian(q_sphBasis, eigenMode); // the spherical coordinates are already updated by get_hessian() above
			for(int i=0; i<D*N; i++) eigenModes[i][lambda]=eigenMode[i];
		}
	}
	

	void get_softestModes(){
		/*
		find eigenvector of lowest eigenvalue and the two lowest eigenvalues of the current hessian
		Caution: does not update the hessian!
		*/

		//construct relevant quantities for Lanczos algorithm
		MatrixXd Q=MatrixXd::Zero((D-1)*N, krylovDimension);
		VectorXd alphas=VectorXd::Zero(krylovDimension);
		VectorXd betas=VectorXd::Zero(krylovDimension); 

		//initialize
		VectorXd v_krylov=qMinEigen_sphBasis; 
		Q.col(0)=(v_krylov)/(v_krylov).norm();// starting vector for lanczos (watch out if using previous eigenvector, the iteration converges quickly and the krylov suspace is of very small dimension - grade) 
		VectorXd r=hessian*Q.col(0); 
		alphas(0)=Q.col(0).transpose()*r;
		r=r-alphas(0)*Q.col(0);
		betas(0)=r.norm();

		//paralellize the force calculation in the loop
		int my_start, my_end, workloads[world_size]={}, startInGlobal[world_size]={}; 
		splitN((D-1)*N, my_start, my_end, workloads, startInGlobal);

		//main loop of Lanczos
		for(int j=1; j<krylovDimension;j++){
			Q.col(j)=r/betas(j-1);
			for(int k=my_start; k<my_end; k++){
				v_krylov(k)=0; 
				for(SparseMatrix<double, RowMajor, int>::InnerIterator hessian_it(hessian,k); hessian_it; ++hessian_it) v_krylov(k)+=hessian_it.value()*(Q.col(j))(hessian_it.col());
			}
			MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, v_krylov.data(), workloads, startInGlobal, MPI_DOUBLE, MPI_COMM_WORLD);
			r=v_krylov-betas(j-1)*Q.col(j-1); 
			alphas(j)=Q.col(j).transpose()*r;
			r=r-alphas(j)*Q.col(j);
		//	r=r-Q.block(0,0,2*N, j+1)*Q.block(0,0,2*N, j+1).transpose()*r; //reortogonalization
			betas(j)=r.norm(); 
			if(betas(j)<1e-10){ //if r.norm()==0, the new krylov vector is contained in the span of all previous ones
				cout << "Grade of Krylov space w.r.t. initial vector reached! " << endl; 
				break; 
			}		
		}
		//diagonalize triadiagonal matrix with elements alphas in the diag and betas in the sub-superdiag (QR algorithm)
		SelfAdjointEigenSolver<MatrixXd> es(krylovDimension);
		es.computeFromTridiagonal(alphas, betas.head(krylovDimension-1)); //constructor (head return the first m-1 values)
		lambdaMin=(es.eigenvalues())(0);
		lambda2Min=(es.eigenvalues())(1); //second lowest eigenvalue
		VectorXd qMinEigen_tridiagonal=es.eigenvectors().col(0); 

		//project eigenvector of tridiagonal to obtain eigenvector of hessian
		qMinEigen_sphBasis=Q*qMinEigen_tridiagonal;	
	
		//transform back into Euclidian space
		double qMinEigen_sphBasis_array[(D-1)*N];
		for(int i=0; i<(D-1)*N; i++) qMinEigen_sphBasis_array[i]=qMinEigen_sphBasis(i);
		from_sphBasis_to_Euclidian(qMinEigen_sphBasis_array, qMinEigen); //the spherical coordinates are already updated when the hessian is computed
	}



	void apply_progression(int mART){
		/*
		If mART=0: HB mode: progress in the direction of local softest mode or in the direction of a normal mode from the minimum
		If mART=1: mART mode: progress towards a combination of the force and the softest mode direction (remain at the bottom of an MEP valley)
		If mART=2: SP mode: progress a constant step away from the first minimum
		If mART=3: Relax mode: progress towards the force
		*/

		// split the for loop per process
		int my_start, my_end, workloads[world_size]={}, startInGlobal[world_size]={}; 
		splitN(N, my_start, my_end, workloads, startInGlobal);

		for(int i=0; i<world_size; i++){
			workloads[i]*=D; //to use in MPI_Allgatherv
			startInGlobal[i]*=D; // start position in global vector
		}

		double g[D*N]; //progression direction
		double hTransverse[D*N]; //transverse effective field
		
		// obtain the effective field 
		get_field(); 

		// obtain transverse field and compute norm
		for(int i=0; i<N; i++){
			double h_dot_s=0;
			for(int d=0; d<D; d++) h_dot_s+=h[i*D+d]*spin[i*D+d];
			for(int d=0; d<D; d++) hTransverse[i*D+d]=(h[i*D+d]-h_dot_s*spin[i*D+d]);
		}
		hTransverseNorm=norm(hTransverse,D*N); 
			
		// compute the hessian to later compute the softest mode and find the adaptative stepsize
		get_hessian();

		if(mART==3){
			// progression direction for minimization is given
			for(int i=0; i<D*N; i++) g[i]=hTransverse[i]; 
		}else{
			//update the softest mode
			get_softestModes();
			// compute dot product between normalize force and softest mode eigenvector (to compute progression direction and check paralellism later)
			hTransverseNormalized_dot_qMinEigen=0; 	
			for(int i=0; i<D*N; i++) hTransverseNormalized_dot_qMinEigen+=hTransverse[i]*qMinEigen[i]/hTransverseNorm;	
			
			if(mART==0){					
				// project perturbation direction on the current tangent space for qConvex in Eucldian space given
				for(int i=0; i<N; i++){
					double qConvexBasin_dot_s=0;
					for(int d=0; d<D; d++) qConvexBasin_dot_s+=qConvexBasin[i*D+d]*spin[i*D+d];
					for(int d=0; d<D; d++) qConvexBasin[i*D+d]=qConvexBasin[i*D+d]-qConvexBasin_dot_s*spin[i*D+d];
				}
				// progression along perturbation
				for(int i=0; i<D*N; i++)  g[i]=qConvexBasin[i];
			}

			if(mART==1){
				// compute progression direction as force combined with minimum eigenvalue direction
				for(int i=0; i<D*N; i++)  g[i]=hTransverse[i]-(gammamART+1)*(hTransverseNormalized_dot_qMinEigen*hTransverseNorm)*qMinEigen[i];
			}


			if(mART==2){
				// progress a constant step away from the last minimum: proportional to the angle travelled 
				for(int i=0; i<N; i++){
					double kick[D];
					double kick_dot_s=0; 
					double kick_sq=0;
					for(int d=0; d<D; d++) kick[d]=spin[i*D+d]-spinRelax[i*D+d];
					for(int d=0; d<D; d++) kick_sq+=pow(kick[d],2);
					double dangle=fabs(2*asin(0.5*pow(kick_sq,0.5)));
					kick_sq=0; 
					// project kick in the tanegent space and normalize by the value of the angle 
					for(int d=0; d<D; d++) kick_dot_s+=kick[d]*spin[i*D+d];
					for(int d=0; d<D; d++) kick[d]=kick[d]-kick_dot_s*spin[i*D+d];
					for(int d=0; d<D; d++) kick_sq+=pow(kick[d],2);
					for(int d=0; d<D; d++) g[i*D+d]=dangle*kick[d]/pow(kick_sq,0.5); 
				}
			}
		}
		// normalize the progression direction
		double maxPonSpin=0;
		double g_norm=norm(g, D*N);	
		for(int i=0; i<D*N; i++) g[i]=g[i]/g_norm;
	
		// compute step size
		if(mART==0 || mART==2){
			if(mART==0) alpha=alpha_perturb; //constant step size during the perturbation 
			if(mART==2) alpha=1e3*alpha_perturb; //to kick away from the saddle point
		}else{ 
			//adaptative step size during the activation
			double g_sphBasis[(D-1)*N]; //write the progression in the tangent space, to then apply Riemannian Hessian
			from_Euclidian_to_sphBasis(g, g_sphBasis); //the spherical coordinates are already updated when the hessian is computed above
			
			double g_dot_hessian_dot_g=0;
			for(int i=(D-1)*my_start; i<(D-1)*my_end; i++){
				for(SparseMatrix<double, RowMajor, int>::InnerIterator hessian_it(hessian, i); hessian_it; ++hessian_it){
					g_dot_hessian_dot_g+=g_sphBasis[i]*hessian_it.value()*g_sphBasis[hessian_it.col()];
				}
			}	
			MPI_Allreduce(MPI_IN_PLACE, &g_dot_hessian_dot_g, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
			alpha=min(abs(2*epsilonMEL*hTransverseNorm/g_dot_hessian_dot_g),0.1);
		}
		
		//add progression to spins
		for(int i=0; i<D*N; i++) spin[i]+=alpha*g[i];

		//renormalize spins
		for(int i=0; i<N; i++){
			double s_norm=0;
			for(int d=0; d<D; d++) s_norm+=spin[i*D+d]*spin[i*D+d];
			s_norm=pow(s_norm,0.5);
			for(int d=0; d<D; d++) spin[i*D+d]/=s_norm;   
		} 
		
		//compute the modeled energy change 	
		double hTransverse_dot_g=0; 
		for(int i=0; i<D*N; i++){
			hTransverse_dot_g+=hTransverse[i]*g[i];
		}

		deltaEModel=-alpha*hTransverse_dot_g;
	} 

	void get_energy(){
		/*
		compute the energy of the configuration 
		*/
		energy=0; 
		int my_start, my_end, workloads[world_size]={}, startInGlobal[world_size]={};
		splitN(N, my_start, my_end, workloads, startInGlobal);
		for(int i=D*my_start; i<D*my_end; i++){
			for(SparseMatrix<double, RowMajor,int>::InnerIterator H_it(H, i); H_it; ++H_it){
				energy+=0.5*spin[i]*H_it.value()*spin[H_it.col()];
			}
		}
		MPI_Allreduce(MPI_IN_PLACE,&energy,1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		for(int i=0; i<N; i++){
			for(int d=0; d<D; d++)	energy+=b[D*i+d]*spin[D*i+d]; //zeeman energy
			// higher order anisotropy energy
			if(environment[i]==0 && cubicQuarticAnisotropyCst!=0){ //quadratic cubic term from O_h point group
				energy+=cubicQuarticAnisotropyCst*(pow(spin[3*i],2)*pow(spin[3*i+1],2)+pow(spin[3*i+1],2)*pow(spin[3*i+2],2)+pow(spin[3*i],2)*pow(spin[3*i+2],2));
			}
		}
	}

	void relax(){
		/*
		steepest descent using the negative effective field (F=-b-H*S is generalized force) projected on the global tangent space and described in {e_x_i, e_y_i, e_z_i}
		*/
		if(my_rank==0){
			cout << "start relaxation to find a local minimum..." << endl;
		}

		// create output file for energy and dot product of q and force
		if(my_rank==0){
			ofstream output("logRelaxation.ssv");
			ofstream file;
			file.open("logRelaxation.ssv", ios_base::app);
			file << setw(20) << "iteration" << setw(20) << "energy" << setw(20) << "lambdaMin" << setw(20) << "trustRatio" << setw(20) << "alpha" << setw(20) << "hTransverseNorm" << endl; 
		}

		// main loop of steepest descent
		double maxIterations=1e6;
		int iteration=0;
		double E_buff=0;
		get_energy();
		
		get_hessian(); 
		get_softestModes(); 
		
		double startTime=MPI_Wtime(); 
		MPI_Bcast(&startTime, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD); //for all the ranks to get the same initial time
		while(true){
			// follow the direction of the force		
			apply_progression(3);  
		
			iteration+=1;

			// compute the energy and store it every 100 configs
			E_buff=energy; 
			get_energy();
		
			// compute trust region parameter to check if correct computation of alpha
			double trustRatio=deltaEModel/(energy-E_buff); 
			
	//		if(my_rank==0 && iteration%300==0) writeConfig("minimiz"+to_string(iteration)+".ssv");
			
			// fill up the log file
			if(my_rank==0){
				ofstream file; 
				file.open("logRelaxation.ssv", ios_base::app);
				file << setw(20) << iteration << setw(20) << setprecision(8) << energy << setw(40) << trustRatio << setw(20) << alpha << setw(20) << hTransverseNorm << setw(20) << endl; 
		
			}

			// check successful criterion
			if(hTransverseNorm<tol){
				get_hessian();
				get_softestModes(); 
				if(my_rank==0) cout << "Minimum found in " << iteration << " iterations. Energy: " << energy << ". Minimum eigenvalue: "<< lambdaMin << "." << endl << endl;	
				if(my_rank==0) writeConfig("configMinimum.ssv");
				break; 
			}

			double actualTime=MPI_Wtime(); 
			MPI_Bcast(&actualTime, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD); //coordinate actual time
			// terminate due to time of max number of iterations
			if(iteration==maxIterations || actualTime-startTime>60*availableTime*0.95){
				if(my_rank==0) cout << "did not reach the minimum (too little time or too few iterations)."  << endl << endl; 
				if(my_rank==0) writeConfig("finalConfig.ssv");
				break; 
			}
		}
	}

	void activate(bool& success, int attempt){
		/*
		mART activation procedure starting with perturbation as selected beforehand by setting qConvexBasin	
		*/
		if(my_rank==0) cout << attempt <<": start activation to find a saddle point..." << endl; 

		// create log output file
		if(my_rank==0){
			ofstream output("logActivation.ssv");
			ofstream file;
			file.open("logActivation.ssv", ios_base::app);
			file << setw(20) << "iteration" << setw(20) << "energy" << setw(20) << "lambdaMin" << setw(20) << "lambda2Min" << setw(20) << "trustRatio" << setw(20) << "alpha" << setw(20) << "hTransNorm" << setw(20) << "hTrans_dot_qMinEigen" << endl; 
		}

		// save the initial configuration to kick away from it at the saddle point
		for(int i=0; i<D*N; i++){
			spinRelax[i]=spin[i];
		}

		//main loop of mART
		int maxIterations=1e6;
		int iteration=0;
		double E_buff=0;
		get_energy(); 
		double E_init=energy; 
		double E_boundaryCB=0; 
		double hTransverseNorm_boundaryCB=0;
		bool convexBasinMode=true; //start in convex basin mode?
	
		get_hessian(); 
		get_softestModes();
		hTransverseNormalized_dot_qMinEigen=0; //reset the global variable
		
		while(true){
			if(lambdaMin<0 && convexBasinMode){
				convexBasinMode=false;
				E_boundaryCB=energy; 
				hTransverseNorm_boundaryCB=hTransverseNorm; 
			}
			 
			if(convexBasinMode){
				//progress towards the mART direction with some fixed stepsize
				apply_progression(0);
			}else{
				//progress towards mART direction with stepsize well define by trust region 
				apply_progression(1);
			}

			iteration+=1;
			
			//compute the energy and store it every 100 configs
			E_buff=energy; 
			get_energy();

			//write config
		//	if(my_rank==0 and iteration%100==0) writeConfig("configmART"+to_string(iteration)+".ssv");
			
			//compute trust region parameter to check if correct computation of alpha
			double trustRatio=deltaEModel/(energy-E_buff);  
			
			//fill up the log file
			if(my_rank==0){
				ofstream file; 
				file.open("logActivation.ssv", ios_base::app);
			file << setw(20) << iteration << setw(20) << setprecision(8) << energy << setw(20)  << lambdaMin << setw(20) << lambda2Min << setw(20) << trustRatio << setw(20) << alpha << setw(20) << hTransverseNorm << setw(20) << fabs(hTransverseNormalized_dot_qMinEigen) << endl; 
			}

			//check requirements for saddle point  
			if(lambdaMin<0 && lambda2Min>0 && hTransverseNorm<tol){
				if(my_rank==0) cout << "Saddle point found in " << iteration << " iterations. Energy: " << energy << ". Minimum eigenvalue: "<< lambdaMin << "." << endl << endl;
				if(my_rank==0) writeConfig("configSaddle"+to_string(attempt)+".ssv");			
				success=true;
				 //kick to the otherside of the saddle point away from the last minimum 
				apply_progression(2);
				break; 	
			}
			
			double actualTime=MPI_Wtime(); 
			MPI_Bcast(&actualTime, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD); //coordinate actual time
			if(actualTime-startTime>60*availableTime*0.95){
				if(my_rank==0) writeConfig("finalConfig.ssv");
				break;
			} 

			//check for loosing the valley
			if(!convexBasinMode && (lambda2Min<0)){
				if(my_rank==0) cout << "Attempt failed: lost the MPE! (lambda1: " << lambdaMin << " and lambda2: " << lambda2Min << ")" << endl << endl; 
				break; 
			}
	//		//check for coming back to the same stable state
	//		if(!convexBasinMode && (energy-E_init)<(E_boundaryHB-E_init)/2){
	//			if(my_rank==0) cout << "Attempt failed: change perturbation or relax less (increase gamma)! (lambda1: " << lambdaMin << " and lambda2: " << lambda2Min << ")" << endl << endl; 
	//			break; 

	//		}
		} 	
	}

	void find_saddlePoints(){
		/*
 		Repeated mART with different perturbations
		*/
		get_hessian();
		get_softestModes();
		double eigenvalues[(D-1)*N];
		double eigenModes[D*N][(D-1)*N];
		get_eigenmodes(eigenvalues, eigenModes);
		get_energy(); 
		double e0=energy; 		

		int numberOfSuccesses=0; 
		for(int attempt=0; attempt<2e6; attempt++){
			readConfig("config.ssv");	
	
			// single spin perturbation
			double qConvexBasin_sphBasis[(D-1)*N];
			for(int i=0; i<N; i++){	
      				qConvexBasin_sphBasis[i]=0; 
      				if(2*i==attempt) qConvexBasin_sphBasis[i]=1; 
      				if(2*i+1==attempt) qConvexBasin_sphBasis[i]=-1; 
      			}	
			get_sphCoorBasis(); 
			from_sphBasis_to_Euclidian(qConvexBasin_sphBasis, qConvexBasin);

			// harmonic mode perturbation (eigenModes are already given in Euclidian space)
//			for(int i=0; i<D*N; i++) qConvexBasin[i]=pow(-1, attempt)*eigenModes[i][attempt/2];
			
			// random perturbation in tangent space
	//		double qConvexBasin_sphBasis[(D-1)*N];
	//		randomOnSphere(qConvexBasin_sphBasis, (D-1)*N);
	//		MPI_Bcast(qConvexBasin_sphBasis, (D-1)*N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	//		get_sphCoorBasis(); 
	//		from_sphBasis_to_Euclidian(qConvexBasin_sphBasis, qConvexBasin);

			bool success=false;
			activate(success, numberOfSuccesses);
			if(success){
				numberOfSuccesses+=1;
				if(my_rank==0){
					ofstream file; 
					file.open("energyBarriers.ssv", ios_base::app);
					file << attempt <<  setw(15) << energy-e0 << endl; 
					file.close(); 
				}	
			}
		}
		if(my_rank==0) cout << "number of successes: " << numberOfSuccesses << endl << endl; 
	}


};

// random point on S^d sphere by using d+1 Gaussians and normalizing
void randomOnSphere(double *randArray, int d){
	double randArray_sq=0; 
	for(int n=0; n<ceil(d/2.0); n++){
		//Box-Muller to create Gaussians		
		double a=(double) rand()/RAND_MAX;
		double b=(double) rand()/RAND_MAX; 
		double phi=a*2*M_PI; 
		double r=pow(-2*log(b),0.5);
		randArray[2*n]=r*cos(phi);
		randArray_sq+=pow(randArray[2*n],2);
		if(2*n+1<d){
			randArray[2*n+1]=r*sin(phi); 	
			randArray_sq+=pow(randArray[2*n+1],2);	
		}	
	}
	for(int i; i<d; i++){
		randArray[i]=randArray[i]/pow(randArray_sq,0.5); 
	}

}

// compute the two-norm of input vector r of length size
double norm(double *r, int size){
	double nor=0;
	for(int i=0; i<size; i++) nor+=r[i]*r[i];
	return sqrt(nor);
}

// split evenly the computation of symmetric matrix by outputing the column index where to start and finish for each rank
void splitN2by2(int N, int &my_start, int &my_end){
	int starts[world_size];
	int ends[world_size];
	starts[0]=1;
	double workloadAvg=0; // workload average on the remaining processes
	for (int i=1; i<world_size; i++){
		workloadAvg=(N*(N-1.0)-starts[i-1]*(starts[i-1]-1.0))/(2.*(world_size-i+1.0));
		starts[i]=0.5+0.5*pow(1.0+4.*(2.*workloadAvg+(starts[i-1]-1.0)*starts[i-1]),0.5);
		ends[i-1]=starts[i];
	}
	ends[world_size-1]=N+1;
	my_start=starts[my_rank]-1; //minus 1 to account that indices start at 0
	my_end=ends[my_rank]-1;
}

// split evenly the computation of a vector among the ranks 
void splitN(int N, int &my_start, int &my_end, int *workloads, int *startInGlobal){
	for (int i=0; i<world_size; i++) {
		workloads[i] = N / world_size;
		if ( i < N % world_size ) workloads[i]++;
		if (i>0) startInGlobal[i]=startInGlobal[i-1]+workloads[i-1];
	}

	my_start=0; 
	for (int i=0; i<my_rank; i++) {
		my_start += workloads[i];
	}
	my_end = my_start + workloads[my_rank];
}


// main() contains the program functions to execute defined above
int main(int argc, char **argv) {  

	//initialize MPI 
	MPI_Init(&argc, &argv);

	//decorator for MPI
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);//get the number of processes that will be paralellized
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);//get the rank (identifier) of the process
	
	//set seed for random vectors using the current time 
	srand((unsigned int) time(0));
	
	//
	//create the system 
	//
	MEL state("config.ssv"); 	
	
	//
	// select the perturbation 
	//

	//uniform perturbation
	double qConvexBasin_sphBasis[(D-1)*N];
	for(int i=0; i<N; i++){	
		qConvexBasin_sphBasis[2*i]=0; //along e_theta_i
		qConvexBasin_sphBasis[2*i+1]=1; //along e_phi_i
	}	
	state.get_sphCoorBasis(); 
	state.from_sphBasis_to_Euclidian(qConvexBasin_sphBasis, state.qConvexBasin);
	state.alpha_perturb=1e-2*pow(N,0.5); 

	//single spin perturbation
//	double qConvexBasin_sphBasis[(D-1)*N];
//	for(int i=0; i<N; i++){	
//		qConvexBasin_sphBasis[2*i]=0; 
//		qConvexBasin_sphBasis[2*i+1]=0; 
//		if(i==5){
//			qConvexBasin_sphBasis[2*i]=1; //along e_theta_i
//			qConvexBasin_sphBasis[2*i+1]=0; //along e_phi_i
//		}
//	}	
//	MEL.get_sphCoorBasis(); 
//	state.from_sphBasis_to_Euclidian(qConvexBasin_sphBasis, state.qConvexBasin);
//	state.alpha_perturb=1e-2;

	//harmonic mode perturbation (already given in Euclidian space)
//	double eigenvalues[(D-1)*N];
//	double eigenModes[D*N][(D-1)*N];
//	state.get_hessian();
//	state.get_eigenmodes(eigenvalues, eigenModes);
//	for(int i=0; i<D*N; i++) state.qConvexBasin[i]=eigenModes[i][0];
//	state.alpha_perturb=1e-2*pow(N,0.5); 
	
	//random perturbation in tangent space
//	double qConvexBasin_sphBasis[(D-1)*N];
//	randomOnSphere(qConvexBasin_sphBasis, (D-1)*N);
//	MPI_Bcast(qConvexBasin_sphBasis, (D-1)*N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
//	state.get_sphCoorBasis(); 
//	state.from_sphBasis_to_Euclidian(qConvexBasin_sphBasis, state.qConvexBasin)
//	state.alpha_perturb=1e-2;

	//exploration
	bool success=false; 
	state.activate(success, 1);
	if(success) state.relax(); 

	//stop MPI
	MPI_Finalize();
	return 0;
}

