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
#include "ROL_OptimizationSolver.hpp"
#include "ROL_StdObjective.hpp"

typedef double RealT;

// forward declaration of the ASL structures
struct ASL_pfgh;

class AmplOptimizationProblem
{
public:
  AmplOptimizationProblem(char**& argv);
  
  virtual ~AmplOptimizationProblem();

  int n_variables();
  int n_constraints();

  const double* x_lower() { return _xl; }
  const double* x_upper() { return _xu; }
  const double* g_lower() { return _gl; }
  const double* g_upper() { return _gu; }

  double* debug_get_x0() { return _x0; }
  
  bool compute_objective(double* x, double& value);
  bool compute_objective_gradient(double* x, double* values);
  bool compute_hessian_vector(double* hv, double* v, double* x);
  void write_solution_file(ROL::Ptr<std::vector<RealT> > x_ptr);
 
private:
  // ensure the default constructor and copy constructor
  // are not built by the compiler
  AmplOptimizationProblem();
  AmplOptimizationProblem(const AmplOptimizationProblem&);
  void operator=(const AmplOptimizationProblem&);

  // pointer to the asl object
  ASL_pfgh* _asl;

  // storage for the initial values and flag
  // indicating if they were set or not
  double* _x0;
  char* _x0_specified;
  double* _lambda0;
  char* _lambda0_specified;

  int _n_nonzeros_h; // number of nonzeros in the full hessian

  // bounds for variables and constraints
  double* _xl;
  double* _xu;
  double* _gl;
  double* _gu;
  
};

template<class Real>
class ObjectiveAMPL : public ROL::StdObjective<Real> {

public:
  ObjectiveAMPL(AmplOptimizationProblem* problem)
    :
    _problem(problem)
  {
  }

  virtual ~ObjectiveAMPL()
  {
    _problem = NULL;
  }

  Real value(const std::vector<Real> &x, Real &tol)
  {
  // ToDo: casting away constness
  // Is this necessary with the AMPL interface?
  // If so, move it inside the AmplOptimizationProblem
  double* xptr = (double*)x.data();
  double retvalue = 0;
  _problem->compute_objective(xptr, retvalue);
  return retvalue;
  }

  void gradient(std::vector<Real> &g, const std::vector<Real> &x, Real &tol)
  {
  double* ret_gradient = g.data();
  double* xptr = (double*)x.data();
  _problem->compute_objective_gradient(xptr, ret_gradient);
  }


  void hessVec( std::vector<Real> &hv, const std::vector<Real> &v, const std::vector<Real> &x, Real &tol )
  {
  // ToDo: see note about constness above.
  double* ret_hv = hv.data();
  double* vptr = (double*)v.data();
  double* xptr = (double*)x.data();
  _problem->compute_hessian_vector(ret_hv, vptr, xptr);
  }

private:
  AmplOptimizationProblem* _problem;
};
