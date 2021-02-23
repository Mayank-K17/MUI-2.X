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
******************************************************************************/

/**
 * @file mui_cuda.cpp
 * @author S. M. Longshaw
 * @date 19 February 2021
 * @brief Encapsulating methods for MUI CUDA kernels
 */

#include "mui_cuda.h"

// Constructor
template <typename TYPE, typename CONFIG>
mui::mui_cuda::mui_cuda() {}

// Destuctor
mui::mui_cuda::~mui_cuda() {}

bool mui::mui_cuda::initCUDA() {
	/*
	int deviceCount=0;
	cudaGetDeviceCount(&deviceCount);

	int gid0=-10;
	cudaGetDevice(&gid0);

	//-Driver information.
	int driverVersion = 0, runtimeVersion = 0;
	cudaDriverGetVersion(&driverVersion);
	cudaRuntimeGetVersion(&runtimeVersion);

	std::cout << "MUI [uniface.h]: CUDA driver/runtime version: " << driverVersion/1000 << "." << (driverVersion%100)/10 << "/"
			  << runtimeVersion/1000 << "." << (runtimeVersion%100)/10 << std::endl;

	if(!deviceCount) {
		std::cout << "MUI Error [uniface.h]: There are no available CUDA device(s)" << std::endl;
		return false;
	}
	else {
		std::cout << "MUI [uniface.h]: Detected CUDA device(s): " << deviceCount << std::endl;

		for(int dev = 0; dev < deviceCount; dev++) {
			cudaSetDevice(dev);

			cudaDeviceProp deviceProp;
			cudaGetDeviceProperties(&deviceProp, dev);

			std::cout << dev << ": " << deviceProp.name << std::endl;
			std::cout << "\tCUDA compute version: " << deviceProp.major << "." << deviceProp.minor << std::endl;
			std::cout << "\tGlobal memory: " << static_cast<float>(deviceProp.totalGlobalMem)/1048576.0f << "MB" << std::endl;
		}

		// Temporary - set first device in index (must be at least 1 but need to improve this for multi-GPU environments)
		cudaSetDevice(0);

		cudaMalloc(&points_cuda, 200*sizeof(point_type));
		cudaMalloc(&values_cuda, 200*sizeof(REAL));

		return true;
	}
	*/
	return true;
}

