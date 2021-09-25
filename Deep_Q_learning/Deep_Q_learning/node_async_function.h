#pragma once
#include <random>
#include <future>
#include "constant.h"

void async_initialize(float* node, const int& start, const int& end, std::mt19937& mt, std::normal_distribution<float>& dist){
	for (int i = start; i < end; ++i) {
		node[i] = dist(mt);
	}
}