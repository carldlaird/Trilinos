// @HEADER
// ************************************************************************
//
//               Rapid Optimization Library (ROL) Package
//                 Copyright (2014) Sandia Corporation
//
// Under terms of Contract DE-AC04-94AL85000, there is a non-exclusive
// license for use of this work by or on behalf of the U.S. Government.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact lead developers:
//              Drew Kouri   (dpkouri@sandia.gov) and
//              Denis Ridzal (dridzal@sandia.gov)
//
// ************************************************************************
// @HEADER

/* Implementation of the AMPL interface with ROL
 *
 */
#include "AmplOptimizationProblem.hpp"
#include <iostream>
#include "asl.h"
#include "getstub.h"
#include "asl_pfgh.h"

#include "ROL_OptimizationSolver.hpp"

#include "ROL_RandomVector.hpp"
#include "ROL_StdObjective.hpp"
#include "ROL_StdConstraint.hpp"
#include "ROL_Bounds.hpp"

#include "Teuchos_oblackholestream.hpp"
#include "Teuchos_GlobalMPISession.hpp"

const int DEBUG_PRINT_LEVEL = 0;
typedef Long fint;

// ToDo: change assert and exits to proper exceptions

void exit_on_ampl_error(fint* nerror, std::string func)
{
  if (nerror == NULL || *nerror != 0) {
    std::cout << "Error returned: " << *nerror << " calling ASL function " << func << ". Aborting" << std::endl;
    exit(1);
  }
}
  

AmplOptimizationProblem::AmplOptimizationProblem(char**& argv)
  :
  _asl(NULL),
  _x0(NULL),
  _lambda0(NULL),
  _n_nonzeros_h(0),
  _xl(NULL),
  _xu(NULL),
  _gl(NULL),
  _gu(NULL)
{
  // pointer needs to be called "asl" to correctly interact
  // with macros from the ASL header files
  ASL_pfgh* asl = _asl = (ASL_pfgh*)ASL_alloc(ASL_read_pfgh);

  char* stub = getstub(&argv, NULL);
  FILE* nl_file = jac0dim(stub, (fint)strlen(stub));
  std::cout << "stub = " << stub << std::endl;

  if (n_cc + nwv + nlnc + lnc > 0) {
    std::cout << "ROL AMPL interface does not support arc variables or network constraints" << std::endl;
    exit(1); // ToDo: Exceptions
  }

  if (nlvci + nlvoi + nlvbi + nbv + niv > 0) {
    std::cout << "ROL AMPL interface does not support discrete variables" << std::endl;
    exit(1); // ToDo: Exceptions
  }

  // read the rest of the nl file
  want_xpi0 = 1 | 2 | 4; // tell ASL we want initial values for variables and multipliers
  obj_no = 0; // we don't support multi-objective - just use the first one
  int status = pfgh_read(nl_file, ASL_return_read_err | ASL_findgroups);

  if (status != ASL_readerr_none) {
    std::cout << "There was an error reading the .nl file" << std::endl;
    exit(1); // ToDo: Exceptions
  }
  
  // set the hessian flags
  // ToDo: look this up
  hesset(1, 0, 0, 0, nlc);

  // single-objective only
  if (objtype[0] != 0) {
    std::cout << "ROL AMPL interface supports single objective problems only" << std::endl;
    exit(1); // ToDo: Exceptions
  }

  // set the flags for the hessian structure
  // ToDo: look this up
  // -1: ?
  // 1: we will supply a coefficient for the objective
  // 1: we will supply multipliers
  // 1: expecting an upper triangular symmetric representation
  _n_nonzeros_h = sphsetup(-1, 1, 1, 1);

  // allocate and retrieve the bounds
  _xl = new double[n_var];
  _xu = new double[n_var];
  _gl = new double[n_con];
  _gu = new double[n_con];

  for (int i=0; i<n_var; i++) {
    _xl[i] = LUv[2*i];
    _xu[i] = LUv[2*i+1];
  }

  for (int i=0; i<n_con; i++) {
    _gl[i] = LUrhs[2*i];
    _gu[i] = LUrhs[2*i+1];
  }
  
  // allocate and retrive the initial values for variables and multpliers
  _x0 = new double[n_var];
  _lambda0 = new double[n_con];

  for (int i=0; i<n_var; i++) {
    if (havex0[i]) {
      _x0[i] = X0[i];
    }
    else {
      // project 0.0 into the bounds
      _x0[i] = 0.0;
      if (_xl[i] > 0.0) {
	_x0[i] = _xl[i];
      }
      if (_xu[i] < 0.0) {
	_x0[i] = _xu[i];
      }
    }
  }

  for (int i=0; i<n_con; i++) {
    _lambda0[i] = 0.0;
    if (havepi0[i]) {
      _lambda0[i] = pi0[i];
    }
  }

  // ToDo: do we need constraint linearity/nonlinearity?

  // print problem stats
  std::cout << "ROL AMPL Interface" << std::endl;
  std::cout << "Number of variables = " << n_var << std::endl;
  std::cout << "Number of constraints = " << n_con << std::endl;
  std::cout << "Number of nonzeros in Jacobian = " << nzc << std::endl;
  std::cout << "Number nonzeros in the Hessian of the Lagrangian = " << _n_nonzeros_h << std::endl;

  if (DEBUG_PRINT_LEVEL > 0) {
    for (int i=0; i<n_var; i++) {
      std::cout << _xl[i] << " <= _x0[" << i << "]=" << _x0[i] << " <= " << _xu[i] << std::endl;
    }
  }
}

AmplOptimizationProblem::~AmplOptimizationProblem()
{
  delete [] _x0;
  _x0 = NULL;
  delete [] _lambda0;
  _lambda0 = NULL;
  
  delete [] _xl;
  _xl = NULL;
  delete [] _xu;
  _xu = NULL;
  delete [] _gl;
  _gl = NULL;
  delete [] _gu;
  _gu = NULL;

  ASL_free((ASL**)&_asl);
  _asl = NULL;
}


int AmplOptimizationProblem::n_variables()
{
  ASL* asl = (ASL*)_asl;
  return n_var;
}

int AmplOptimizationProblem::n_constraints()
{
  ASL* asl = (ASL*)_asl;
  return n_con;
}

bool AmplOptimizationProblem::compute_objective(double* x, double& value) {
  //std::cout << "In compute_objective" << std::endl;
  // ASL uses macros that rely on this variable
  ASL* asl = (ASL*)_asl;

  // very inefficient - for now, assume we have a new x with every call
  fint* nerror = new fint;
  *nerror = 0;
  xknowne(x, nerror);
  exit_on_ampl_error(nerror, "xknowne");

  value = objval(0, x, nerror);
  exit_on_ampl_error(nerror, "objval");

  return true;
}

bool AmplOptimizationProblem::compute_objective_gradient(double* x, double* values) {
  //std::cout << "In compute_objective_gradient" << std::endl;
  // ASL uses macros that rely on this variable
  ASL* asl = (ASL*)_asl;

  // very inefficient - for now, assume we have a new x with every call
  fint* nerror = new fint;
  *nerror = 0;
  xknowne(x, nerror);
  exit_on_ampl_error(nerror, "xknowne");

  objgrd(0, x, values, nerror);
  exit_on_ampl_error(nerror, "objgrd");

  return true;
}

bool AmplOptimizationProblem::compute_hessian_vector(double* hv, double* v, double* x) {
  //std::cout << "In compute_hessian_vector" << std::endl;
  // ASL uses macros that rely on this variable
  ASL* asl = (ASL*)_asl;

  // very inefficient - for now, assume we have a new x with every call
  fint* nerror = new fint;
  *nerror = 0;
  xknowne(x, nerror);
  exit_on_ampl_error(nerror, "xknowne");

  double dummy = 0.0;
  compute_objective(x, dummy);
  double* dummy_g = new double[n_con];
  conval(x, dummy_g, nerror);
  exit_on_ampl_error(nerror, "conval");
  delete [] dummy_g;
  dummy_g = NULL;

  // only compute with respect to the first objective, and ignore the constraints
  double OW = 1.0;
  double* lam = new double[n_con];
  for (int i=0; i<n_con; i++) {
    lam[i] = 0.0;
  }
  hvinit(-1, &OW, lam);
  hvcomp(hv, v, -1, &OW, lam);

  delete [] lam;
  lam = NULL;
  
  return true;
}

void AmplOptimizationProblem::write_solution_file(ROL::Ptr<std::vector<RealT> > x_ptr)
{
  ASL* asl = (ASL*)_asl;
  double* x_sol = new double[n_var];
  for (int i=0; i<n_var; i++) {
    x_sol[i] = (*x_ptr)[i];
  }
  write_sol("Optimal solution found", x_sol, NULL, NULL);
}


int main(int argc, char** argv)
{
  AmplOptimizationProblem* problem = new AmplOptimizationProblem(argv);

  /*
  double value = 0.0;
  double* x = problem->debug_get_x0();
  problem->compute_objective(x, value);
  std::cout << "f(x) = " << value << std::endl;

  int n = problem->n_variables();
  double* dfdx = new double[n];
  problem->compute_objective_gradient(x, dfdx);
  for (int i=0; i<n; i++) {
    std::cout << "dfdx[" << i << "]=" << dfdx[i] << std::endl;
  }
  delete [] dfdx;
  dfdx = NULL;

  double* v = new double[n];
  for (int i=0; i<n; i++) {
    v[i] = 0.0;
  }
  
  double* hv = new double[n];
  
  v[0] = 1.0;
  problem->compute_hessian_vector(hv, v, x);
  for (int i=0; i<n; i++) {
    std::cout << "hv[" << i << "]=" << hv[i] << std::endl;
  }

  v[0] = 0.0;
  v[1] = 1.0;
  problem->compute_hessian_vector(hv, v, x);
  for (int i=0; i<n; i++) {
    std::cout << "hv[" << i << "]=" << hv[i] << std::endl;
  }

  delete [] v;
  v = NULL;
  delete [] hv;
  hv = NULL;

  std::cout << "foo" << std::endl;
  */

  int n = problem->n_variables();
  
  // setup the ROL problem
  ROL::Ptr<std::vector<RealT> > rol_x_ptr  = ROL::makePtr<std::vector<RealT>>(n,2.0);

    ROL::Ptr<ROL::Vector<RealT> > rol_x  = ROL::makePtr<ROL::StdVector<RealT>>(rol_x_ptr);

  ROL::Ptr<ROL::Objective<RealT> > rol_obj = ROL::makePtr<ObjectiveAMPL<RealT>>(problem);

  ROL::OptimizationProblem<RealT> rol_problem( rol_obj, rol_x);

  // rol_problem.check(std::cout);

  Teuchos::ParameterList parlist;
  parlist.sublist("Step").set("Type","Augmented Lagrangian");

  ROL::OptimizationSolver<RealT> rol_solver( rol_problem, parlist );

  rol_solver.solve(std::cout); 

  /*
  std::cout << "x_opt = [";
  for(int i=0;i<n-1;++i) {
    std::cout << (*rol_x_ptr)[i] << ", " ;
  }
  std::cout << (*rol_x_ptr)[n-1] << "]" << std::endl;
  */
  problem->write_solution_file(rol_x_ptr);
  
  delete problem;
}

