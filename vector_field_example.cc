#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
//Mesh related classes
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/tria_boundary_lib.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_in.h>
//Finite element implementation classes
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/manifold_lib.h>


using namespace dealii;
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <unistd.h>
#include <time.h>

//Define the order of the basis functions (Lagrange polynomials)
//and the order of the quadrature rule globally
const unsigned int order = 1;
const unsigned int quadRule = 2;

template <int dim>
class FEM
{
    public:
    //Class functions
        FEM (); // Class constructor
        ~FEM(); //Class destructor

        //Solution steps

        void generate_mesh(std::vector<unsigned int> numberOfElements);
        void define_boundary_conds(double time);
        void setup_system(); 
        void apply_initial_conditions();
        void assemble_system(double t);
        void solve(); 
        void output_results(unsigned int index); 
        void output_dat_file(unsigned int index);


    //Class objects

    Triangulation<dim> triangulation;                         // mesh
    FESystem<dim>      fe;                                    // FE element
    DoFHandler<dim>    dof_handler;                           // Connectivity matrices

    //NEH - deal.II quadrature
    QGauss<dim>     quadrature_formula; //Quadrature
    QGauss<dim-1>   face_quadrature_formula; //Face Quadrature

    //Data structures

    SparsityPattern sparsity_pattern; //Sparse natrix pattern

    SparseMatrix<double>          K, M, system_matrix; //Global stiffness natrix - Sparse matrix - used in the solver
    Vector<double>                F, RHS, D;  //Global vectors - Solution vector (D), Global force vector (F), 
                                                       // right hand side of the final system of equations (RHS)

    Table<2,double>               dofLocation; //Table of the coordinates of dofs by global dof nunber
    std::map<unsigned int,double> boundary_values; //Map of dirichlet boundary conditions

    //solution name array
    std::vector<std::string> nodal_solution_names;
    std::vector<DataComponentInterpretation::DataComponentInterpretation> nodal_data_component_interpretation;
 
    double x_min = 0., // rectangular domain
           x_max = 1.,
           y_min = 0.,
           y_max = 0.2,
           z_min = 0.,
           z_max = 1.5;

    double pi = std::atan(1.)*4.; // pi const
};

// Class constructor for a vector field
template <int dim>
FEM<dim>::FEM ()
    :
    fe (FE_Q<dim>(order), dim),
    dof_handler (triangulation),
    quadrature_formula(quadRule),
    face_quadrature_formula(quadRule)
    {

    //Nodal Solution names - this is for writttig the output file

    for (unsigned int i=0; i<dim; ++i){
        nodal_solution_names.push_back("u");
        nodal_data_component_interpretation.push_back(DataComponentInterpretation::component_is_part_of_vector);

    }
}

//class destructor
template <int dim>
FEM<dim>::~FEM (){dof_handler.clear ();}


//the problem domain and mesh deneration
template <int dim>
void FEM<dim>::generate_mesh(std::vector<unsigned int> numberOfElements){

    //the limits of our domain

    Point<dim,double> min(x_min,y_min,z_min),
                      max(x_max,y_max,z_max);

    GridGenerator::subdivided_hyper_rectangle(triangulation, numberOfElements, min ,max);
}

//Specify the Dirichlet boundary conditions
template <int dim>
void FEM<dim>::define_boundary_conds(double time){

    /*
    Since we now have more than 1 dof per node, it is possible to apply a different Dirichlet condition on
    each of the 3 dofs on the same node. Note that this is NOT the case for the current assignment. 
    But if you wanted to do it, you would need to check the nodal dof (0, 1, or 2) as well as the location. 
    For example, if you wanted to fix displacements only in the x-direction at x=0, you would have an
    condition such as this: (doflocation[globalDOF][8] == 0 && nodalDOF == 0) Note that you can get 
    the nodal dof from the global dof using the "modulo" operator, i.e nodalDOF = globalDOF%dim. 
    Here "%" gives you the remainder when dividing globaIDOF by dim.) / 
    Add the global dof number and the specified value (displacement in this problem) to the boundary values map, 
    something like this: 

    boundary_values[globalDOFIndex] = dirichletDisplacementValue 
    Note that "dofLocation" is now a Table of degree of degree of freedom locations, not just node locations,
    so, for example, dofs 0,1 and 2 correspond to the same node, so that have the same coordinates.
    The row index is hte global dof number; the column index refers to the x,y or z component (0,1, or 2 for 3D).
    e.g. dofLocation[7][2] is the z-coordinate of blobal dof 7 */

    const unsigned int totalDOFs = dof_handler.n_dofs(); // Total number of degrees of freedom

    for(unsigned int global_dof_index=0; global_dof_index < totalDOFs; global_dof_index++){

        if(dofLocation[global_dof_index][0] == x_min && global_dof_index%dim == 1){
            boundary_values[global_dof_index] = 0.;  
        }
        if(dofLocation[global_dof_index][0] == x_max && global_dof_index%dim == 1){
            boundary_values[global_dof_index] = 0; 
        }

        if(dofLocation[global_dof_index][2] == z_min && global_dof_index%dim == 0){
            boundary_values[global_dof_index] = 0.;  
        }

        if(dofLocation[global_dof_index][2] == z_max && global_dof_index%dim == 0){
            boundary_values[global_dof_index] = 0.;  
        }

        if(dofLocation[global_dof_index][1] == y_min && global_dof_index%dim == 2){
            boundary_values[global_dof_index] = 1.*std::exp(-3.*time);  
        }

    }
}


//Setup data structures (sparse matrix, vectors) 
template <int dim> 
void FEM<dim>::setup_system(){ 

    //Let deal.II organize degrees of freedom 
    dof_handler.distribute_dofs (fe); 

    //Get a vector of global degree-of-freedom x-coordinates 
    MappingQ1<dim,dim> mapping; 
    std::vector< Point<dim,double> > dof_coords(dof_handler.n_dofs()); 
    dofLocation.reinit(dof_handler.n_dofs(),dim); 
    DoFTools::map_dofs_to_support_points<dim,dim>(mapping,dof_handler,dof_coords); 
    for(unsigned int i=0; i<dof_coords.size(); i++){
        for(unsigned int j=0; j<dim; j++){ 
            dofLocation[i][j] = dof_coords[i][j]; 
        }
    }
 
    //Specify boundary condtions (call the function) 

    //Define the size of the global matrices and vectors 
    sparsity_pattern.reinit (dof_handler.n_dofs(), 
                             dof_handler.n_dofs(),
                             dof_handler.max_couplings_between_dofs());        
   
    DoFTools::make_sparsity_pattern (dof_handler, sparsity_pattern); 
    sparsity_pattern.compress(); 

    M.reinit (sparsity_pattern);
    K.reinit (sparsity_pattern); 
    system_matrix.reinit (sparsity_pattern);

    RHS.reinit(dof_handler.n_dofs());
    D.reinit(dof_handler.n_dofs()); 
    F.reinit(dof_handler.n_dofs()); 

    //Just some notes...

    std::cout << " Number of active elems: " << triangulation.n_active_cells() << std::endl;
    std::cout << " Number of degrees of greedom " << dof_handler.n_dofs() << std::endl;
}

//Form elmental vectors and matrices and assemble to the global vector (F) and matrix (K) 
template <int dim> 
void FEM<dim>::assemble_system(double time){

    /*NEW - deal.II basis functions, etc. The third input values (after fe and quadrature_formula) 
    specify what information we want to be updated. For fe_values, we need the basis function values, 
    basis function gradients, and det(Jacobian) times the quadrature weights (JxW). For fe_face_values, 
    we need the basis function values, the value of x at the quadrature points, and JxW.*/ 
 
    //For volume integration/quadrature points 
    FEValues<dim> fe_values (fe, 
                             quadrature_formula, 
                             update_values | 
                             update_quadrature_points |  
                             update_gradients |
                             update_JxW_values); 

    //For surface integration/quadrature points 
    FEFaceValues<dim> fe_face_values (fe, 
                                      face_quadrature_formula, 
                                      update_values | 
                                      update_quadrature_points |                 
                                      update_JxW_values); 
 
    K=0; F=0; M=0; 

    const unsigned int dofs_per_elem = fe.dofs_per_cell; //This gives you dofs per element
    const unsigned int nodes_per_elem = GeometryInfo<dim>::vertices_per_cell; 
    const unsigned int num_quad_pts = quadrature_formula.size(); //Total number of quad points in the element    
    const unsigned int num_face_quad_pts = face_quadrature_formula.size(); //Total number of quad points in the face
    const unsigned int faces_per_elem = GeometryInfo<dim>::faces_per_cell; 

    FullMatrix<double> Klocal (dofs_per_elem, dofs_per_elem); 
    FullMatrix<double> Mlocal (dofs_per_elem, dofs_per_elem); 
    Vector<double> Flocal (dofs_per_elem); 
    std::vector<unsigned int> local_dof_indices (dofs_per_elem); //This relates local dof numbering to global dof numbering

    //loop over elements 
    typename DoFHandler<dim>::active_cell_iterator elem = dof_handler.begin_active(), endc = dof_handler.end(); 

    for (;elem!=endc; ++elem){
        //Update fe_values for the current element 
        fe_values.reinit(elem); 

        //Retrieve the effective "connectivity matrix" for this element 
        elem->get_dof_indices (local_dof_indices); 

        /*Global Assembly -
        Get the current Flocal and Klocal from the functions you wrote above 
        and populate F_assembly and K_assembly using local_dof_indices to relate local and global DOFs.*/ 
        
        Flocal = 0.;   
        Klocal = 0.; 
        Mlocal = 0.; 

        //Loop over local DOFs and quadrature points to populate Klocal 
        //Note that all quadrature points are included in this single loop 
        for (unsigned int q=0; q<num_quad_pts; ++q){ 
            //evaluate elemental stiffness matrix, K^{AB}_{ik} \integral N^A_{,j}*C_{ijkl}*N^B_{,l} dV 
            for (unsigned int A=0; A<nodes_per_elem; A++){ //Loop over nodes 
                for(unsigned int i=0; i<dim; i++){ //Loop over nodal dots 
                    for (unsigned int B=0; B<nodes_per_elem; B++) { 
                        Mlocal[dim*A+i][dim*B+i] += fe_values.shape_value(dim*A+i,q)*fe_values.shape_value(dim*B+i,q)*fe_values.JxW(q);
                        for(unsigned int k=0; k<dim; k++){ 
                            /* Note that the indices of Klocal are the element dof numbers
                            which one can caluclate from the element node numbers (0 through 8) and the nodal dofs (0 through 2). 
                            You'll need the following information: 
                            basis gradient vector: fe_values.shape_grad(elementD0F,q), where elementDOF is dim*A+i or dim*B+k 
                            NOTE: this is the gradient with respect to the real domain (not the bi-unit domain) 
                            det(J) times the total quadrature weight: fe_values.JxW(q)*/ 

                            Klocal[dim*A+k][dim*B+k] += fe_values.shape_grad(dim*A+k,q)[i]*fe_values.shape_grad(dim*B+k,q)[i]*fe_values.JxW(q); 

                            // The next term is currently commented out because we by definition consider dirvergence-free cases
                            // Nonetheless, it may be interesting to see how it could have been written.
                            //Klocal[dim*A+k][dim*B+i] -= fe_values.shape_grad(dim*A+k,q)[k]*fe_values.shape_grad(dim*B+i,q)[i]*fe_values.JxW(q); 
                             
                        }
                    }
                }
            }   
        }  
        
        // Here we consider contributions to the Flocal due to the non-zero forcing vector f
        for (unsigned int q=0; q<num_quad_pts; ++q){ 
            double z = fe_values.quadrature_point(q)[2]; //z-coordinate at the current quad. point 
            double y = fe_values.quadrature_point(q)[1]; //y-coordinate at the current quad. point 

            double f_x = (-2-3*z*(z-1.5))*std::exp(-3.*time);
            double f_z = (-pi*pi-3)*std::exp(pi*y-3.*time);

            for (unsigned int A=0; A<nodes_per_elem; A++){ //Loop over nodes 
                Flocal[dim*A+0] += f_x*fe_values.shape_value(A*dim+0,q)*fe_values.JxW(q); // f
                Flocal[dim*A+2] += f_z*fe_values.shape_value(A*dim+2,q)*fe_values.JxW(q); // f
            }
        }
    
        
        //Loop over faces (for Neumann BCs), local DOFs and quadrature points to populate Flocal. 
        for (unsigned int f=0; f < faces_per_elem; f++){ 
            //Update fe_face_values from current element and face
            fe_face_values.reinit (elem, f); 
            /*elem->face(f)->center() gives a position vector (in the real domain) of the center point on face f 
            of the current element. We can use it to see if we are at the Neumann boundary, y = 0.2 */ 
            if(elem->face(f)->center()[1] == y_max){ 
                //To integrate over this face, loop over all face quadrature points with this single loop 
                for (unsigned int q=0; q<num_face_quad_pts; ++q){ 
                    for (unsigned int A=0; A<nodes_per_elem; A++){ //loop over all element nodes 
                        /* the indices of Flocal are the element dof numbers (0 through 23). 

                        Note that we are looping over all element dofs, not just those on the Neumann face. However, 
                        the face quadrature points are only on the Neumann face, so we are indeed doing a surface integral. 

                        For det(J) times the total quadrature weight: fe_face_values.JxW(q)*/ 
 
                        Flocal[dim*A+2] += pi*std::exp(pi*y_max)*std::exp(-3.*time)*fe_face_values.shape_value(A*dim+2,q)*fe_face_values.JxW(q);          
                    }
                }
            }
        }
        
       
        for (unsigned int i=0; i<dofs_per_elem; ++i){                         
            F[local_dof_indices[i]] += Flocal[i];                              
            for (unsigned int j=0; j<dofs_per_elem; ++j){                     
                K.add(local_dof_indices[i],local_dof_indices[j],Klocal[i][j]);
                M.add(local_dof_indices[i],local_dof_indices[j],Mlocal[i][j]);
            }
        }
    }
}


// Applying initial conditions
template <int dim>
void FEM<dim>::apply_initial_conditions(){

    const unsigned int totalDOFs = dof_handler.n_dofs(); // Total number of degrees of freedom

    for(unsigned int global_dof_index=0; global_dof_index < totalDOFs; global_dof_index = global_dof_index + 1){
        if(global_dof_index%dim == 0){
            D[global_dof_index] = dofLocation[global_dof_index][2]*(dofLocation[global_dof_index][2]-1.5);
        }
        if(global_dof_index%dim == 1){
            D[global_dof_index] = 0.;
        }
        if(global_dof_index%dim == 2){
            D[global_dof_index] = std::exp(pi * dofLocation[global_dof_index][1]);
        }
    }
}


template <int dim>
void FEM<dim>::solve(){
    
    double delta_t = 0.0005;                                                    // initial time step
    double t_step = 0.0, t_max = 1;                                            // initial time

    const unsigned int totalNodes = dof_handler.n_dofs();                     // Total number of nodes

    Vector<double> D_tilde(totalNodes);

    apply_initial_conditions();                                           // applying initial conditions at t_step = 0

    unsigned int snap_shot_counter = 0;

   
    while(t_step < t_max){                                                    // Loop over time 
        
        if(snap_shot_counter%20==0){
            output_results(snap_shot_counter/20);
            output_dat_file(snap_shot_counter/20); 
        }

        t_step += delta_t;                                                    // updating time
    
        define_boundary_conds(t_step);  
        assemble_system(t_step);
        
        // Solving a matrix equation d_{n+1} = (M + alpha*delta_t*K)^{-1}*(alpha*delta_t*F_{n+1} + M*d_tilde_{n+1}),
        // where d_tilde_{n+1} = d_{n} + delta_t*(1-alpha)*v_{n}

        D_tilde = D;
        system_matrix.copy_from(M);
        system_matrix.add(delta_t,K);

        M.vmult(RHS,D_tilde);
        RHS.add(delta_t,F);

        MatrixTools::apply_boundary_values(boundary_values, system_matrix, D, RHS, false);

        //SparseDirectUMFPACK  A;
        //A.initialize(system_matrix);
        //A.vmult (D, RHS);
 
           
        SolverControl solver_control(2000, 1e-22);
        SolverCG<>   cg(solver_control);
        cg.solve(system_matrix, D, RHS,
           PreconditionIdentity());

        if(snap_shot_counter%5==0){
            std::cout << solver_control.last_step()
            << " CG iterations."
            << std::endl;
            std::cout << "t = " << t_step << std::endl;
            std::cout << "snap_shot_counter = " << snap_shot_counter << std::endl;
            std::cout << " " << std::endl;
        }
   

        snap_shot_counter ++;
    }
}

template <int dim>
void FEM<dim>::output_results (unsigned int index){

    //Write to VTK

    char filename[100];

    snprintf(filename, 100, "./solution_trans_%d.vtk", index);
    std::ofstream output1 (filename);

    //std::ofstream output1 ("solution.vtk");
    DataOut<dim> data_out; data_out.attach_dof_handler (dof_handler);

    // Add data

    data_out.add_data_vector (D,
                         nodal_solution_names,
                         DataOut<dim>::type_dof_data,
                         nodal_data_component_interpretation);
    data_out.build_patches();
    data_out.write_vtk(output1);
    output1.close();
}


template <int dim>
void FEM<dim>::output_dat_file(unsigned int index){

    const unsigned int totalNodes = dof_handler.n_dofs(); //Total number of nodes

    char buffer[100];

    snprintf(buffer, 100, "./output%i.dat", index);

    FILE* temperature_profile;
    temperature_profile = std::fopen(buffer, "w" );

    for(unsigned  int i=0; i<totalNodes; i=i+3){
        std::fprintf(temperature_profile, "%14.7e %14.7e %14.7e %14.7e %14.7e %14.7e\n", dofLocation[i][0], dofLocation[i][1], dofLocation[i][2], D[i], D[i+1], D[i+2]);
    }
    std::fclose(temperature_profile);
}

// The main program, using the FEM class
int main (){
    try{

        deallog.depth_console (0);
        const int dimension = 3;

        //NOTE: This is where you define the nunber of elements the the mesh
        std::vector<unsigned int> num_of_elems(dimension);

        num_of_elems[0] = 5;
        num_of_elems[1] = 10;
        num_of_elems[2] = 10;

        FEM<dimension> problemobject;
        problemobject.generate_mesh(num_of_elems);
        problemobject.setup_system();
        //problemobject.assemble_system();
        problemobject.solve();
        //problemobject.output_results();
        //problemobject.output_dat_file();

    }
    catch (std::exception &exc){
        std::cerr << std::endl << std::endl
	<< "----------------------------------------------------"
	<< std::endl;
	std::cerr << "Exception on processing: " << std::endl
	<< exc.what() << std::endl
	<< "Aborting!" << std::endl
	<< "----------------------------------------------------"
	<< std::endl;

	return 1;
	}
    catch (...){
	std::cerr << std::endl << std::endl
	<< "----------------------------------------------------"
	<< std::endl;
	std::cerr << "Unknown exception" << std::endl
	<< "Aborting!" << std::endl
	<< "----------------------------------------------------"
	<< std::endl;
	return 1;
    }
    return 0;
}
   
    

