/*****************************************************************************
* Multiscale Universal Interface Code Coupling Library                       *
*                                                                            *
* Copyright (C) 2021 S. M. Longshaw                                          *
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
 * @file sampler_gauss.cu
 * @author S. M. Longshaw
 * @date 18 February 2021
 * @brief CUDA kernel spatial sampler that provides a value at a point
 * using Gaussian interpolation.
 */

namespace mui {
	__global__
	void filter_cuda( point_type focus, point_type* points, double* values, size_t count, double r, double h ) {
		int index = threadIdx.x;
		int stride = blockDim.x;

		double nh = std::pow(2.0*3.1415926535897932385*h,-0.5*3);

		double wsum = 0;
		double vsum = 0;

		for( size_t i = index; i < count; i+=stride ) {
			auto d = normsq( focus - points[i] );
			if ( d < r*r ) {
				double w = nh * std::exp( (double(-0.5)/h) * d );
				vsum += values[i] * w;
				wsum += w;
			}
		}

		//if ( wsum ) return vsum / wsum;
		//else return REAL(0.);
	}

}
