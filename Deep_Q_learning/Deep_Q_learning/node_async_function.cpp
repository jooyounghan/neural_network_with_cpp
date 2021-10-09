#include "node_async_function.h"

void asyncInitialize(float* node, const int& start, const int& end, std::mt19937& mt, std::normal_distribution<float>& dist) {
	for (int idx = start; idx < end; ++idx) {
		node[idx] = dist(mt);
	}
	return;
}

void asyncMatMul(float* node, float* node_in, float* w, const int& N, const int& K, const int& M, const int& start, const int& end) {
	for (int idx = start; idx < end; ++idx) {
		const int& row = idx / M;
		const int& col = idx % M;
		node[idx] = 0;
		for (int k = 0; k < K; ++k) {
			node[idx] += node_in[row * K + k] * w[k * M + col];
		}
	}
	return;
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
	return;
}

void asyncSquare(float* node, const int& start, const int& end) {
	for (int idx = start; idx < end; ++idx) {
		node[idx] = node[idx] * node[idx];
	}
	return;
}

void asyncGetSquare(float* node, float* node_in, const int& start, const int& end) {
	for (int idx = start; idx < end; ++idx) {
		node[idx] = node_in[idx] * node_in[idx];
	}
	return;
}

void asyncGetSqrt(float* node, float* node_in, const int& start, const int& end) {
	for (int idx = start; idx < end; ++idx) {
		node[idx] = std::sqrtf(node_in[idx]);
	}
	return;
}

void asyncGetFraction(float* node, float* node_in, const int& start, const int& end) {
	for (int idx = start; idx < end; ++idx) {
		node[idx] = 1 / node_in[idx];
	}
	return;
}

void asyncGetNodeSubtract(float* node, float* node_from, float* node_sub, const int& start, const int& end) {
	for (int i = start; i < end; ++i) {
		node[i] = node_from[i] - node_sub[i];
	}
	return;
}

void asyncGetNodeMul(float* node, float* node_from, float* node_mul, const int& start, const int& end) {
	for (int i = start; i < end; ++i) {
		node[i] = node_from[i] * node_mul[i];
	}
	return;
}

void asycnGetConstantMul(float* node, float* node_in, const float& num, const int& start, const int& end) {
	for (int i = start; i < end; ++i) {
		node[i] = node_in[i] * num;
	}
	return;
}

void asyncTranspose(float* node, float* node_in, const int& N, const int& M, const int& start, const int& end) {
	for (int idx = start; idx < end; ++idx) {
		const int& row = idx / M;
		const int& col = idx % M;
		node[col * N + row] = node_in[row * M + col];
	}
	return;
}

void asyncNodeSubtract(float* node, float* node_from, float* node_sub, const int& start, const int& end) {
	for (int idx = start; idx < end; ++idx) {
		node[idx] = node_from[idx] - node_sub[idx];
	}
	return;
}

void asyncCopy(float* node, float* node_in, const int& start, const int& end) {
	for (int idx = start; idx < end; ++idx) {
		node[idx] = node_in[idx];
	}
	return;
}

void asyncConstantMul(float* node, const float& num, const int& start, const int& end) {
	for (int idx = start; idx < end; ++idx) {
		node[idx] = node[idx] * num;
	}
}

void asyncConstantMat(float* node, float* node_in, const float& num, const int& start, const int& end) {
	for (int idx = start; idx < end; ++idx) {
		node[idx] = node_in[idx] * num;
	}
	return;
}

void asyncConstantAdd(float* node, float* node_in, const float& num, const int& start, const int& end) {
	for (int idx = start; idx < end; ++idx) {
		node[idx] = node_in[idx] + num;
	}
}

void asyncNodeElementWiseAdd(float* node, float* node_in_1, float* node_in_2, const int& start, const int& end) {
	for (int idx = start; idx < end; ++idx) {
		node[idx] = node_in_1[idx] + node_in_2[idx];
	}
	return;
}
