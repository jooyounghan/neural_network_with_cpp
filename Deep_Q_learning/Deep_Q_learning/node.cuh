#include <iostream>
#include "node_function.cuh"

class CudaNode {

public:
	int row;
	int col;
	int size;
	float* device_node = nullptr;
	float* host_node = nullptr;

public:
	CudaNode(const int& row_in, const int& col_in)
		: row(row_in), col(col_in), size(row_in * col_in) {
		host_node = new float[size];
		cudaMalloc(&device_node, sizeof(float) * this->size);
	}

	void memSet() {
		cudaMemcpy(this->device_node, this->host_node, sizeof(float) * this->size, cudaMemcpyHostToDevice);
	}

	// 노드 수에 대한 초기화로 수정하여야 함
	void cudaInitialize() {
		dim3 block_dim(this->col, this->row);
		initialize << <1, block_dim >> > (this->device_node);
		cudaDeviceSynchronize();
		cudaMemcpy(this->host_node, this->device_node, sizeof(float) * this->size, cudaMemcpyDeviceToHost);
	}

	void cudaMatMul(CudaNode& node_in, CudaNode& w_in) {
		node_in.memSet();
		w_in.memSet();
		dim3 block_dim(this->col, this->row);

		matMul << <1, block_dim >> > (this->device_node, node_in.device_node, w_in.device_node, node_in.col, w_in.col);
		cudaDeviceSynchronize();
		cudaMemcpy(this->host_node, this->device_node, sizeof(float) * this->size, cudaMemcpyDeviceToHost);
	}

	void print() {
		for (int i = 0; i < row; ++i) {
			for (int j = 0; j < col; ++j) {
				std::cout << host_node[i * this->col + j] << " ";
			}
			std::cout << "\n";
		}
	}
};