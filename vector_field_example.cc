#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/timer.h>

#include <deal.II/lac/generic_linear_algebra.h>

namespace LA
{
using namespace dealii::LinearAlgebraTrilinos;
}

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparsity_tools.h>

#include <deal.II/lac/petsc_parallel_sparse_matrix.h>
#include <deal.II/lac/petsc_parallel_vector.h>
#include <deal.II/lac/petsc_solver.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/manifold_lib.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_q.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>

#include <deal.II/base/utilities.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/index_set.h>

#include <deal.II/distributed/tria.h>
#include <deal.II/distributed/grid_refinement.h>

#include <fstream>
#include <iostream>

using namespace dealii;


//Define the order of the basis functions (Lagrange polynomials)
//and the order of the quadrature rule globally
const unsigned int order = 1;
const unsigned int quadRule = 4;

template <int dim>
class FEM
{
    public:
    //Class functions
        FEM (); // Class constructor
        ~FEM(); //Class destructor

        //Solution steps
        void generate_mesh();
        void apply_initial_conditions();
        int e(int i, int j, int k);  // The levi civita terson. It is recommended to rewrite it in a more "numerically optimized" form 
        void setup_system(); 
        void assemble_system();
        void solve(); 
        void output_results (const unsigned int cycle) const;

        // these three functions below are employed to compute the velocity field due to toroidal magnetic field (currently just once, in the beginning of the simulation)
        void assemble_vel();
        void solve_vel(); 
        void output_vel (const unsigned int cycle) const;

    // a vector for the initial toroidal magnetic field. We will specify its components inside the assembly_vel procedure at all quadrature points.
    std::vector<double> BT_INITIAL = {0.,0.,0.}; 

    double pi = 3.141592653589793238462;
    
    MPI_Comm                                  mpi_communicator;

    parallel::distributed::Triangulation<dim> triangulation;
 
    FESystem<dim>                             fe;
    DoFHandler<dim>                           dof_handler;
    std::map< types::global_dof_index, Point< dim > > dof_coords;

    QGauss<dim>   quadrature_formula;                       
    QGauss<dim-1> face_quadrature_formula;                

    IndexSet                                  locally_owned_dofs;
    IndexSet                                  locally_relevant_dofs;

    ConstraintMatrix                          constraints;

    // D is the solution vector (poloidal pertubation of magnetic field, initial configuration is along e_z)
    // VEL is the vecocity field that has to be calculated by computing VEL = 1/(4pi n) rot ( BT_INITIAL )
    // matricies K, M and a vector F are discussed in the .pdf notes
    // The other objects are initialized only for computational purposes
    LA::MPI::SparseMatrix                     K, M, system_matrix;
    LA::MPI::Vector                           D, F, VEL, RHS, D_tilde, completely_distributed_solution;
    std::vector<unsigned int> local_dofs;	                    

    ConditionalOStream                        pcout;
    TimerOutput                               computing_timer;
   
    // needed for constructing output data
    std::vector<std::string> nodal_solution_names;
    std::vector<DataComponentInterpretation::DataComponentInterpretation> nodal_data_component_interpretation;
};


// Class constructor for a vector field
template <int dim>
FEM<dim>::FEM ()
    :
    mpi_communicator (MPI_COMM_WORLD),
    triangulation (mpi_communicator),
    fe (FE_Q<dim>(order), dim),
    dof_handler (triangulation),
    quadrature_formula(quadRule),
    face_quadrature_formula(quadRule),
    pcout (std::cout,
           (Utilities::MPI::this_mpi_process(mpi_communicator)
            == 0)),
    computing_timer (mpi_communicator,
                     pcout,
                     TimerOutput::summary,
                     TimerOutput::wall_times)
    {
    for (unsigned int i=0; i<dim; ++i){
        nodal_solution_names.push_back("u");
        nodal_data_component_interpretation.push_back(DataComponentInterpretation::component_is_part_of_vector);
    }
}


//class destructor
template <int dim>
FEM<dim>::~FEM (){dof_handler.clear ();}


// The levi-civita terson
template <int dim> 
int FEM<dim>::e(int i, int j, int k){

    if (i==j && i==k && k==j){
        return 0;
    }
    else if (i==0 && j==1 && k==2){
       return 1;
    }
    else if (i==0 && j==2 && k==1){
       return -1;
    }
    else if (i==1 && j==2 && k==0){
       return 1;
    }
    else if (i==1 && j==0 && k==2){
       return -1;
    }
    else if (i==2 && j==0 && k==1){
       return 1;
    }
    else if (i==2 && j==1 && k==0){
       return -1;
    }
    else{
        return 0;
    }
}


//Define the problem domain and generate the mesh
template <int dim>
void FEM<dim>::generate_mesh(){ // setting up the mesh
     
    
    double  x_center = 0.,
            y_center = 0.,
            z_center = 0.;

    Point<dim,double> center(x_center,y_center,z_center);
    GridGenerator::hyper_shell (triangulation, center, 0.4, 1.0, 96, true);
    static const SphericalManifold<3> manifold_description(center);
    triangulation.set_all_manifold_ids(0);
    triangulation.set_manifold (0, manifold_description);
    triangulation.refine_global(2);
}


// Applying initial conditions
template <int dim>
void FEM<dim>::apply_initial_conditions(){

    // here we want to enter the initial values of the solution vector D (at t=0). 

    LA::MPI::Vector auxiliary_vector (locally_owned_dofs, mpi_communicator); // we need a vector without ghost elements to access and fill its components.

    for (unsigned int i=0; i<dof_handler.n_locally_owned_dofs(); i=i+3){
        auxiliary_vector[local_dofs[i+0]] = 0.0 ;  // xth component
        auxiliary_vector[local_dofs[i+1]] = 0.0 ;  // yth component
        auxiliary_vector[local_dofs[i+2]] = 0.001 ; // zth component
    }
    
    // After specifying components of the auxiliary vector we make it to be in the form as D (i.e. with ghost cells)
    constraints.distribute (auxiliary_vector);
    // Now we can copy the data from the auxiliary vector to D 
    D = auxiliary_vector;
}


//Setup data structures (sparse matrix, vectors) 
template <int dim> 
void FEM<dim>::setup_system(){ 

    // some initializing procedures 
    TimerOutput::Scope t(computing_timer, "setup");

    dof_handler.distribute_dofs (fe);

    locally_owned_dofs = dof_handler.locally_owned_dofs ();
    DoFTools::extract_locally_relevant_dofs (dof_handler,
                                             locally_relevant_dofs);

    MappingQ1<dim,dim> mapping;
    DoFTools::map_dofs_to_support_points<dim,dim>(mapping,dof_handler,dof_coords);

    D.reinit (locally_owned_dofs,locally_relevant_dofs, mpi_communicator) ;
    VEL.reinit (locally_owned_dofs,locally_relevant_dofs, mpi_communicator);

    D_tilde.reinit (locally_owned_dofs, mpi_communicator);
    RHS.reinit (locally_owned_dofs, mpi_communicator);
    F.reinit (locally_owned_dofs, mpi_communicator);
    completely_distributed_solution.reinit (locally_owned_dofs, mpi_communicator);

    locally_owned_dofs.fill_index_vector(local_dofs);

    pcout << "   Number of active elems:       " << triangulation.n_active_cells() << std::endl;
    pcout << "   Number of degrees of freedom: " << dof_handler.n_dofs() << std::endl;
    std::cout << "   Number of local degrees of freedom: " << dof_handler.n_locally_owned_dofs() << std::endl;

    constraints.clear ();
    constraints.reinit (locally_relevant_dofs);
    constraints.close ();

  
    DynamicSparsityPattern dsp (locally_relevant_dofs);

    DoFTools::make_sparsity_pattern (dof_handler, dsp,
                                     constraints, false);
    SparsityTools::distribute_sparsity_pattern (dsp,
                                                dof_handler.n_locally_owned_dofs_per_processor(),
                                                mpi_communicator,
                                                locally_relevant_dofs);

    system_matrix.reinit (locally_owned_dofs,
                          locally_owned_dofs,
                          dsp,
                          mpi_communicator);

    K.reinit (locally_owned_dofs,
                          locally_owned_dofs,
                          dsp,
                          mpi_communicator);
    M.reinit (locally_owned_dofs,
                          locally_owned_dofs,
                          dsp,
                          mpi_communicator);
}


// assembly function for computing velocities
template <int dim> 
void FEM<dim>::assemble_vel(){

    TimerOutput::Scope t(computing_timer, "Assembly");
    
    FEValues<dim> fe_values (fe, 
                             quadrature_formula, 
                             update_values | 
                             update_quadrature_points |  
                             update_gradients |
                             update_JxW_values); 

    FEFaceValues<dim> fe_face_values (fe, 
                                      face_quadrature_formula, 
                                      update_values | 
                                      update_quadrature_points |                 
                                      update_JxW_values |
                                      update_normal_vectors); 
 
    K=0; F=0; 

    const unsigned int dofs_per_elem = fe.dofs_per_cell; 
    const unsigned int num_quad_pts = quadrature_formula.size();    
    const unsigned int num_face_quad_pts = face_quadrature_formula.size(); 
    const unsigned int faces_per_elem = GeometryInfo<dim>::faces_per_cell; 

    FullMatrix<double> Klocal (dofs_per_elem, dofs_per_elem); 
    Vector<double> Flocal (dofs_per_elem); 
    std::vector<unsigned int> local_dof_indices (dofs_per_elem);  
    typename DoFHandler<dim>::active_cell_iterator elem = dof_handler.begin_active(), endc = dof_handler.end(); 

    for (;elem!=endc; ++elem){

        if (elem->is_locally_owned()){

            fe_values.reinit(elem); 
            elem->get_dof_indices (local_dof_indices); 

            Klocal = 0.; 
            Flocal = 0.;  

            for (unsigned int q=0; q<num_quad_pts; ++q){ 

                double x = fe_values.quadrature_point(q)[0]; 
                double y = fe_values.quadrature_point(q)[1]; 
                double z = fe_values.quadrature_point(q)[2]; 

                double R = std::sqrt( x*x + y*y );
                double r = std::sqrt( x*x + y*y + z*z );

                double n = 1-r*r;

                // initial toroidal component of the magnetic field, B_z = 0
                BT_INITIAL[0] = -y/R * pow(R,4) * n*n;
                BT_INITIAL[1] = x/R * pow(R,4) * n*n;

                for (unsigned int A=0; A<dofs_per_elem/3; A++){ 
                    for(unsigned int k=0; k<dim; k++){ 
                        for (unsigned int B=0; B<dofs_per_elem/3; B++) { 
                            Klocal[dim*A+k][dim*B+k] += fe_values.shape_value(dim*A+k,q)*fe_values.shape_value(dim*B+k,q)*fe_values.JxW(q); 
                        }
                        for(unsigned int i=0; i<dim; i++){ 
                            for(unsigned int j=0; j<dim; j++){ 
                                Flocal[dim*A+k] += BT_INITIAL[i]*e(i,j,k)*fe_values.shape_grad(A*dim+k,q)[j]*fe_values.JxW(q); // f
                            }
                        }
                    }
                }   
            }
            // performing the surface integral       
            for (unsigned int f=0; f < faces_per_elem; f++){ 
                fe_face_values.reinit (elem, f); 
                if(elem->face(f)->at_boundary()){
                    for (unsigned int q=0; q<num_face_quad_pts; ++q){ 

                        double x = fe_face_values.quadrature_point(q)[0]; 
                        double y = fe_face_values.quadrature_point(q)[1]; 
                        double z = fe_face_values.quadrature_point(q)[2]; 

                        double R = std::sqrt( x*x + y*y );
                        double r = std::sqrt( x*x + y*y + z*z );

                        double n = 1-r*r;

                         // initial toroidal component of the magnetic field, B_z = 0
                        BT_INITIAL[0] = -y/R * pow(R,4) * n*n;
                        BT_INITIAL[1] = x/R * pow(R,4) * n*n;

                        for (unsigned int A=0; A<dofs_per_elem/3; A++){ 
                            for(unsigned int i=0; i<dim; i++){ 
                                for(unsigned int j=0; j<dim; j++){ 
                                    for(unsigned int k=0; k<dim; k++){               
                                        Flocal[dim*A+j] -= fe_face_values.normal_vector(q)[i]*e(i,j,k)*fe_face_values.shape_value(A*dim+j,q)*BT_INITIAL[k]*fe_face_values.JxW(q); // f         
                                    }
                                }
                            }
                        }  
                    }
                }
            }       
            // adding up local contributions to the global matricies and vectors
            constraints.distribute_local_to_global (Flocal,
                                                   local_dof_indices,
                                                   F);

            constraints.distribute_local_to_global (Klocal,
                                                   local_dof_indices,
                                                   K);
        }
    }
    // the interative solution of the linear system begins only after all the processors are done with the assembly of their part of the manifold. 
    // Thus some comminication between the processors is required
    K.compress (VectorOperation::add);
    F.compress (VectorOperation::add);
}


// solve for velocities
template <int dim>
void FEM<dim>::solve_vel(){

    computing_timer.enter_subsection ("Solve"); 
                                            
    SolverControl solver_control(1000, 1e-12 * F.l2_norm());
    dealii::TrilinosWrappers::SolverCG cg(solver_control);
    dealii::TrilinosWrappers::PreconditionSSOR preconditioner;
    preconditioner.initialize(K, 1.0);
    cg.solve(K, completely_distributed_solution, F, preconditioner);
       
    // above we solve the following equation: A = curl ( B_toroidar ). Therefore, to obtain the velocities we also need to divide the above solution by 4pi n, since v = 1/(4pi n) A
    for (unsigned int i=0; i<dof_handler.n_locally_owned_dofs(); i++){
        double r = std::sqrt( pow(dof_coords[local_dofs[i]][0],2) + pow(dof_coords[local_dofs[i]][1],2) + pow(dof_coords[local_dofs[i]][2],2) );
        double n = 1 - r*r + 1e-5;
        completely_distributed_solution(local_dofs[i]) = completely_distributed_solution(local_dofs[i])/4./pi/n ;
    }

    // now we put the velocities to a vector, that contains ghost cell. It will become important in the following assembly section of the code, where we have to retrieve some 
    // information from the velocity field.
    constraints.distribute (completely_distributed_solution);

    VEL = completely_distributed_solution;
    pcout << solver_control.last_step() << " CG iterations." << std::endl;
    computing_timer.leave_subsection();
}


template <int dim> 
void FEM<dim>::assemble_system(){

    TimerOutput::Scope t(computing_timer, "Assembly");


    FEValues<dim> fe_values (fe, 
                             quadrature_formula, 
                             update_values | 
                             update_quadrature_points |  
                             update_gradients |
                             update_JxW_values); 

    FEFaceValues<dim> fe_face_values (fe, 
                                      face_quadrature_formula, 
                                      update_values | 
                                      update_quadrature_points |                 
                                      update_JxW_values |
                                      update_gradients |
                                      update_normal_vectors); 
 
    K=0; M=0; 

    const unsigned int dofs_per_elem = fe.dofs_per_cell; 
    const unsigned int num_quad_pts = quadrature_formula.size(); 
    const unsigned int num_face_quad_pts = face_quadrature_formula.size(); 
    const unsigned int faces_per_elem = GeometryInfo<dim>::faces_per_cell; 

    FullMatrix<double> Klocal (dofs_per_elem, dofs_per_elem); 
    FullMatrix<double> Mlocal (dofs_per_elem, dofs_per_elem); 
    Vector<double> Flocal (dofs_per_elem); 

    std::vector<unsigned int> local_dof_indices (dofs_per_elem); 

    typename DoFHandler<dim>::active_cell_iterator elem = dof_handler.begin_active(), endc = dof_handler.end(); 

    for (;elem!=endc; ++elem){

        if (elem->is_locally_owned()){
 
            fe_values.reinit(elem); 
            elem->get_dof_indices (local_dof_indices); 
        
            Klocal = 0.; 
            Mlocal = 0.;  
            Flocal = 0.;  

            for (unsigned int q=0; q<num_quad_pts; ++q){  
                Vector<double> u(dim);
                u = 0;
                for(unsigned int A=0; A<dofs_per_elem/3; A++){
                    for(unsigned int k=0; k<dim; k++){ 
                        u[k] += VEL[local_dof_indices[A*dim+k]]*fe_values.shape_value(A*dim+k,q);
                    }
                }
                for (unsigned int A=0; A<dofs_per_elem/3; A++){
                    for(unsigned int k=0; k<dim; k++){ 
                        for (unsigned int B=0; B<dofs_per_elem/3; B++) {  
                            Mlocal[dim*A+k][dim*B+k] += fe_values.shape_value(dim*A+k,q)*fe_values.shape_value(dim*B+k,q)*fe_values.JxW(q);
                            for(unsigned int i=0; i<dim; i++){
                                for(unsigned int j=0; j<dim; j++){
                                    for(unsigned int l=0; l<dim; l++){ 
                                        for(unsigned int m=0; m<dim; m++){ 
                                            Klocal[dim*A+k][dim*B+m] -= e(i,j,k)*fe_values.shape_grad(A*dim+k,q)[j]*e(i,l,m)*u[l]*fe_values.shape_value(B*dim+m,q)*fe_values.JxW(q); // f
                                        }
                                    }
                                }
                            }
                        } 
                    }
                }   
            }  

            // performing the surface integral
            for (unsigned int f=0; f < faces_per_elem; f++){         
                fe_face_values.reinit (elem, f); 
                for (unsigned int q=0; q<num_face_quad_pts; ++q){ 
                    if(elem->face(f)->at_boundary()){ 
                        // the velocity field was previously computed at all the nodes of the manifold, but here we need it at the quadrature points. Thus we do the interpolation.
                        Vector<double> u(dim);
                        u = 0.0;
                        for(unsigned int A=0; A<dofs_per_elem/3; A++){
                            for(unsigned int k=0; k<dim; k++){ 
                                u[k] += VEL[local_dof_indices[A*dim+k]]*fe_face_values.shape_value(A*dim+k,q);
                            }
                        }
                        for (unsigned int A=0; A<dofs_per_elem/3; A++){ 
                            for(unsigned int i=0; i<dim; i++){ 
                                for (unsigned int B=0; B<dofs_per_elem/3; B++) {
                                    for(unsigned int j=0; j<dim; j++){ 
                                        for(unsigned int k=0; k<dim; k++){ 
                                            for(unsigned int l=0; l<dim; l++){ 
                                                for(unsigned int m=0; m<dim; m++){ 
                                                    Klocal[dim*A+j][dim*B+m] += fe_face_values.normal_vector(q)[i]*e(i,j,k)*fe_face_values.shape_value(A*dim+j,q)*e(k,l,m)*u[l]*fe_values.shape_value(B*dim+m,q)*fe_face_values.JxW(q); 
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }  
                    }
                }
            }
            constraints.distribute_local_to_global (Mlocal,
                                                    local_dof_indices,
                                                    M);
            constraints.distribute_local_to_global (Klocal,
                                                    local_dof_indices,
                                                    K);
        }
    }

    M.compress (VectorOperation::add);
    K.compress (VectorOperation::add);
}



template <int dim>
void FEM<dim>::solve(){

    double delta_t = 0.00005;                                                   // initial
    double t_step = 0.0, t_max = 1;                                            // initial time


    apply_initial_conditions(); // apply the initial conditions to the poloidal perturbation field                                        

    assemble_vel(); // assembly for the velocity field
    solve_vel(); // compute the velocity field
     
    computing_timer.enter_subsection ("Output");  
    output_vel(0); 
    computing_timer.leave_subsection();
    
    unsigned int snap_shot_counter = 0;
    
    while(t_step < t_max){        

        computing_timer.enter_subsection ("Output");                               
        output_results(snap_shot_counter);
        computing_timer.leave_subsection();

        t_step += delta_t;                                                    // updating time
      
        assemble_system();

        computing_timer.enter_subsection ("Solve"); 
                                            
        D_tilde = D;
        system_matrix.copy_from(M);
        system_matrix.add(-delta_t,K);

        M.vmult(RHS,D_tilde);
        
        SolverControl solver_control(1000, 1e-8 * RHS.l2_norm());
        dealii::TrilinosWrappers::SolverCG cg(solver_control);
        dealii::TrilinosWrappers::PreconditionSSOR preconditioner;
        preconditioner.initialize(system_matrix, 1.0);
        cg.solve(system_matrix, completely_distributed_solution, RHS, preconditioner);
        
        constraints.distribute (completely_distributed_solution);

        D = completely_distributed_solution;

        computing_timer.leave_subsection();
         
        // some output
        pcout << solver_control.last_step() << " CG iterations." << std::endl;
        pcout << "t = " << t_step << std::endl;
        pcout << "snap_shot_counter = " << snap_shot_counter << std::endl;
        pcout << " " << std::endl;
        
        snap_shot_counter ++;
    }     
}


// function that generates .vtk-files for the velocity field
template <int dim>
void FEM<dim>::output_vel (const unsigned int cycle) const
{

    DataOut<dim> data_out;
    data_out.attach_dof_handler (dof_handler);
    data_out.add_data_vector (VEL,
                         nodal_solution_names,
                         DataOut<dim>::type_dof_data,
                         nodal_data_component_interpretation);

    Vector<float> subdomain (triangulation.n_active_cells());
    for (unsigned int i=0; i<subdomain.size(); ++i)
      subdomain(i) = triangulation.locally_owned_subdomain();
    data_out.add_data_vector (subdomain, "subdomain");

    data_out.build_patches ();

    const std::string filename = ("output/vel-" +
                                  Utilities::int_to_string (cycle, 3) +
                                  "." +
                                  Utilities::int_to_string
                                  (triangulation.locally_owned_subdomain(), 4));
    std::ofstream output ((filename + ".vtu").c_str());
    data_out.write_vtu (output);

    if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
    {
        std::vector<std::string> filenames;
        for (unsigned int i=0;
             i<Utilities::MPI::n_mpi_processes(mpi_communicator);
             ++i)
          filenames.push_back ("vel-" +
                               Utilities::int_to_string (cycle, 3) +
                               "." +
                               Utilities::int_to_string (i, 4) +
                               ".vtu");

        std::ofstream master_output (("output/vel-" +
                                      Utilities::int_to_string (cycle, 3) +
                                      ".pvtu").c_str());
        data_out.write_pvtu_record (master_output, filenames); 
    }
}


// function that generates .vtk-files for the poloidal perturbation
template <int dim>
void FEM<dim>::output_results (const unsigned int cycle) const
{

    DataOut<dim> data_out;
    data_out.attach_dof_handler (dof_handler);
    data_out.add_data_vector (D,  nodal_solution_names,
                         DataOut<dim>::type_dof_data,
                         nodal_data_component_interpretation);

    Vector<float> subdomain (triangulation.n_active_cells());
    for (unsigned int i=0; i<subdomain.size(); ++i)
      subdomain(i) = triangulation.locally_owned_subdomain();
    data_out.add_data_vector (subdomain, "subdomain");

    data_out.build_patches ();

    const std::string filename = ("output/solution-" +
                                  Utilities::int_to_string (cycle, 3) +
                                  "." +
                                  Utilities::int_to_string
                                  (triangulation.locally_owned_subdomain(), 4));
    std::ofstream output ((filename + ".vtu").c_str());
    data_out.write_vtu (output);

    if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
    {
        std::vector<std::string> filenames;
        for (unsigned int i=0;
             i<Utilities::MPI::n_mpi_processes(mpi_communicator);
             ++i)
          filenames.push_back ("solution-" +
                               Utilities::int_to_string (cycle, 3) +
                               "." +
                               Utilities::int_to_string (i, 4) +
                               ".vtu");

        std::ofstream master_output (("output/solution-" +
                                      Utilities::int_to_string (cycle, 3) +
                                      ".pvtu").c_str());
        data_out.write_pvtu_record (master_output, filenames);
    }
}



// The main program, using the FEM class
int main (int argc, char *argv[]){

    try{
        deallog.depth_console (0);

        using namespace dealii;

        Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

        const int dimension = 3;

        FEM<dimension> problemobject;
        problemobject.generate_mesh();
        problemobject.setup_system();
        problemobject.solve();

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
   
    

