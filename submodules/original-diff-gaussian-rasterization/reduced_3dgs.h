/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#ifndef REDUCED_3DGS_H_INCLUDED
#define REDUCED_3DGS_H_INCLUDED

#include <vector>
#include <functional>
#include <torch/extension.h>

namespace Reduced3DGS
{
	std::tuple<torch::Tensor, torch::Tensor>
	kmeans(
		const torch::Tensor &values,
		const torch::Tensor &centers,
		const float tol,
		const int max_iterations);
};

#endif