#pragma once
#include <random>
#include <future>
#include "constant.h"

#define ERR		1E-6

void asyncInitialize(float* node, const int& start, const int& end, std::mt19937& mt, std::normal_distribution<float>& dist){
	for (int idx = start; idx < end; ++idx) {
		node[idx] = dist(mt);
	}
}

void asyncMatMul(float* node, float* node_in, float* w, const int& N, const int& K, const int& M, const int& start, const int& end) {
	for (int idx = start; idx < end; ++idx) {
		const int& row = start / M;
		const int& col = start % M;
		node[idx] = 0;
		for (int k = 0; k < K; ++k) {
			node[idx] += node_in[row * K + k] + w[k * M + col];
		}
	}
}
bool asyncIsSame(float* node, float* node_in, const int& start, const int& end) {
	for (int idx = start; idx < end; ++idx) {
		if (node[idx] < node_in[idx] + ERR && node[idx] > node_in[idx] - ERR) {
			return false;
		}
	}
	return true;
}

void asyncRelu(float* node, float* node_in, const int& start, const int& end) {
	for (int idx = start; idx < end; ++idx) {
		node[idx] = node_in[idx] > 0 ? node_in[idx] : 0;
	}
}