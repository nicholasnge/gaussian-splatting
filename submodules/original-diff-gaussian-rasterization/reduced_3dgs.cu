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

#include <math.h>
#include <torch/extension.h>
#include <cstdio>
#include <sstream>
#include <iostream>
#include <tuple>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <memory>
#include "cuda_rasterizer/config.h"
#include "cuda_rasterizer/rasterizer.h"
#include <fstream>
#include <string>
#include <functional>
#include <glm/glm.hpp>
#include "reduced_3dgs.h"
#include "reduced_3dgs/kmeans.h"

#include <cooperative_groups.h>
namespace cg = cooperative_groups;
#include "cuda_rasterizer/auxiliary.h"
#include "cuda_rasterizer/rasterizer_impl.h"
#include "cuda_rasterizer/forward.h"
using namespace torch::indexing;

std::function<char *(size_t N)> resizeFunctional(torch::Tensor &t);

// Works with 256 centers 1 dimensional data only
std::tuple<torch::Tensor, torch::Tensor>
Reduced3DGS::kmeans(
	const torch::Tensor &values,
	const torch::Tensor &centers,
	const float tol,
	const int max_iterations)
{
	const int n_values = values.size(0);
	const int n_centers = centers.size(0);
	torch::Tensor ids = torch::zeros({n_values, 1}, values.options().dtype(torch::kInt32));
	torch::Tensor new_centers = torch::zeros({n_centers}, values.options().dtype(torch::kFloat32));
	torch::Tensor old_centers = torch::zeros({n_centers}, values.options().dtype(torch::kFloat32));
	new_centers = centers.clone();
	torch::Tensor center_sizes = torch::zeros({n_centers}, values.options().dtype(torch::kInt32));

	for (int i = 0; i < max_iterations; ++i)
	{
		updateIds(
			values.contiguous().data<float>(),
			ids.contiguous().data<int>(),
			new_centers.contiguous().data<float>(),
			n_values,
			n_centers);

		old_centers = new_centers.clone();
		new_centers.zero_();
		center_sizes.zero_();

		updateCenters(
			values.contiguous().data<float>(),
			ids.contiguous().data<int>(),
			new_centers.contiguous().data<float>(),
			center_sizes.contiguous().data<int>(),
			n_values,
			n_centers);

		new_centers = new_centers / center_sizes;
		new_centers.index_put_({new_centers.isnan()}, 0.f);
		float center_shift = (old_centers - new_centers).abs().sum().item<float>();
		if (center_shift < tol)
			break;
	}

	updateIds(
		values.contiguous().data<float>(),
		ids.contiguous().data<int>(),
		new_centers.contiguous().data<float>(),
		n_values,
		n_centers);

	return std::make_tuple(ids, new_centers);
}