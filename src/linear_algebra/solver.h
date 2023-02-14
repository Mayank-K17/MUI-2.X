/*****************************************************************************
* Multiscale Universal Interface Code Coupling Library                       *
*                                                                            *
* Copyright (C) 2023 W. Liu                                                  *
*                                                                            *
* This software is jointly licensed under the Apache License, Version 2.0    *
* and the GNU General Public License version 3, you may use it according     *
* to either.                                                                 *
*                                                                            *
* ** Apache License, version 2.0 **                                          *
*                                                                            *
* Licensed under the Apache License, Version 2.0 (the "License");            *
* you may not use this file except in compliance with the License.           *
* You may obtain a copy of the License at                                    *
*                                                                            *
* http://www.apache.org/licenses/LICENSE-2.0                                 *
*                                                                            *
* Unless required by applicable law or agreed to in writing, software        *
* distributed under the License is distributed on an "AS IS" BASIS,          *
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.   *
* See the License for the specific language governing permissions and        *
* limitations under the License.                                             *
*                                                                            *
* ** GNU General Public License, version 3 **                                *
*                                                                            *
* This program is free software: you can redistribute it and/or modify       *
* it under the terms of the GNU General Public License as published by       *
* the Free Software Foundation, either version 3 of the License, or          *
* (at your option) any later version.                                        *
*                                                                            *
* This program is distributed in the hope that it will be useful,            *
* but WITHOUT ANY WARRANTY; without even the implied warranty of             *
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the              *
* GNU General Public License for more details.                               *
*                                                                            *
* You should have received a copy of the GNU General Public License          *
* along with this program.  If not, see <http://www.gnu.org/licenses/>.      *
*****************************************************************************/

/**
 * @file conjugate_gradient.h
 * @author W. Liu
 * @date 03 February 2023
 * @brief Classes to solve problem A.x = b using different methods.
 * Solver implemented:
 *        Conjugate Gradient (iterative)
 *        Gaussian Elimination (direct)
 */

#ifndef MUI_SOLVER_H_
#define MUI_SOLVER_H_

#include <cmath>

#include "matrix.h"
#include "preconditioner.h"

namespace mui {
namespace linalg {

// Base linear equation solver class
template<typename ITYPE, typename VTYPE>
class solver {

    public:
        // Abstract function for solve
        virtual std::pair<ITYPE, VTYPE> solve(sparse_matrix<ITYPE,VTYPE> = sparse_matrix<ITYPE,VTYPE>()) = 0;
        // Abstract function to get the solution
        virtual sparse_matrix<ITYPE,VTYPE> getSolution() = 0;
};

// Class of one-dimensional Conjugate Gradient solver
template<typename ITYPE, typename VTYPE>
class conjugate_gradient_1d : public solver<ITYPE,VTYPE> {

    public:
        // Constructor
        conjugate_gradient_1d(sparse_matrix<ITYPE,VTYPE>, sparse_matrix<ITYPE,VTYPE>, VTYPE = 1e-6, ITYPE = 0, preconditioner<ITYPE,VTYPE>* = nullptr);
        // Destructor
        ~conjugate_gradient_1d();
        // Member function for solve
        std::pair<ITYPE, VTYPE> solve(sparse_matrix<ITYPE,VTYPE> = sparse_matrix<ITYPE,VTYPE>());
        // Member function to get the solution
        sparse_matrix<ITYPE,VTYPE> getSolution();

    private:
        // The coefficient matrix of the matrix equation
        sparse_matrix<ITYPE,VTYPE> A_;
        // The variable matrix of the matrix equation
        sparse_matrix<ITYPE,VTYPE> x_;
        // The constant matrix of the matrix equation
        sparse_matrix<ITYPE,VTYPE> b_;
        // The residual matrix of the CG solver
        sparse_matrix<ITYPE,VTYPE> r_;
        // The preconditioned residual matrix of the CG solver
        sparse_matrix<ITYPE,VTYPE> z_;
        // The direction matrix of the CG solver
        sparse_matrix<ITYPE,VTYPE> p_;
        // Preconditioner pointer
        preconditioner<ITYPE,VTYPE>* M_;
        // Tolerance of CG solver
        VTYPE cg_solve_tol_;
        // Maximum iteration of CG solver
        ITYPE cg_max_iter_;
};

// Class of multidimensional Conjugate Gradient solver
template<typename ITYPE, typename VTYPE>
class conjugate_gradient : public solver<ITYPE,VTYPE> {

    public:
        // Constructor
        conjugate_gradient(sparse_matrix<ITYPE,VTYPE>, sparse_matrix<ITYPE,VTYPE>, VTYPE = 1e-6, ITYPE = 0, preconditioner<ITYPE,VTYPE>* = nullptr);
        // Destructor
        ~conjugate_gradient();
        // Member function for solve
        std::pair<ITYPE, VTYPE> solve(sparse_matrix<ITYPE,VTYPE> = sparse_matrix<ITYPE,VTYPE>());
        // Member function to get the solution
        sparse_matrix<ITYPE,VTYPE> getSolution();

    private:
        // The coefficient matrix of the matrix equation
        sparse_matrix<ITYPE,VTYPE> A_;
        // The constant matrix of the matrix equation
        sparse_matrix<ITYPE,VTYPE> b_;
        // The column segments of the constant matrix of the matrix equation
        sparse_matrix<ITYPE,VTYPE> b_column_;
        // The column segments of the initial guess of the variable matrix of the matrix equation
        sparse_matrix<ITYPE,VTYPE> x_init_column_;
        // The variable matrix of the matrix equation
        sparse_matrix<ITYPE,VTYPE> x_;
        // Preconditioner pointer
        preconditioner<ITYPE,VTYPE>* M_;
        // Tolerance of CG solver
        VTYPE cg_solve_tol_;
        // Maximum iteration of CG solver
        ITYPE cg_max_iter_;
};

// Class of one-dimensional Gaussian Elimination solver
template<typename ITYPE, typename VTYPE>
class gaussian_elimination_1d : public solver<ITYPE,VTYPE> {

    public:
        // Constructor
        gaussian_elimination_1d(sparse_matrix<ITYPE,VTYPE>, sparse_matrix<ITYPE,VTYPE>);
        // Destructor
        ~gaussian_elimination_1d();
        // Member function for solve
        std::pair<ITYPE, VTYPE> solve(sparse_matrix<ITYPE,VTYPE> = sparse_matrix<ITYPE,VTYPE>());
        // Member function to get the solution
        sparse_matrix<ITYPE,VTYPE> getSolution();

    private:
        // The coefficient matrix of the matrix equation
        sparse_matrix<ITYPE,VTYPE> A_;
        // The constant matrix of the matrix equation
        sparse_matrix<ITYPE,VTYPE> b_;
        // The variable matrix of the matrix equation
        sparse_matrix<ITYPE,VTYPE> x_;

};

// Class of multidimensional Gaussian Elimination solver
template<typename ITYPE, typename VTYPE>
class gaussian_elimination : public solver<ITYPE,VTYPE> {

    public:
        // Constructor
        gaussian_elimination(sparse_matrix<ITYPE,VTYPE>, sparse_matrix<ITYPE,VTYPE>);
        // Destructor
        ~gaussian_elimination();
        // Member function for solve
        std::pair<ITYPE, VTYPE> solve(sparse_matrix<ITYPE,VTYPE> = sparse_matrix<ITYPE,VTYPE>());
        // Member function to get the solution
        sparse_matrix<ITYPE,VTYPE> getSolution();

    private:
        // The coefficient matrix of the matrix equation
        sparse_matrix<ITYPE,VTYPE> A_;
        // The constant matrix of the matrix equation
        sparse_matrix<ITYPE,VTYPE> b_;
        // The variable matrix of the matrix equation
        sparse_matrix<ITYPE,VTYPE> x_;
        // The column segments of the constant matrix of the matrix equation
        sparse_matrix<ITYPE,VTYPE> b_column_;

};

} // linalg
} // mui

// Include implementations
#include "../linear_algebra/solver_cg.h"
#include "../linear_algebra/solver_ge.h"

#endif /* MUI_SOLVER_H_ */
