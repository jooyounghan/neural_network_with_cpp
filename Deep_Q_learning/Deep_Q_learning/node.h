#pragma once
#include <iostream>
#include "node_async_function.h"

class Node {

public:
	int row;
	int col;
	int size;
	float* node = nullptr;

public:
	Node(const int& row_in, const int& col_in)
		: row(row_in), col(col_in), size(row_in* col_in) {
		node = new float[size];
	}

	void heInitialize(Node& before) {
		std::mt19937 mt(seed);
		std::normal_distribution<float> dist(0, (float)std::sqrt(2.0 / before.size));
		
		std::vector<std::future<void>> tasks;
		tasks.resize(cores);

		int task_allocation = size / cores;

		for (int i = 0; i < cores; ++i) {
			if (i == cores - 1) {
				tasks[i] = std::async(async_initialize, std::ref(node), task_allocation * i, size, std::ref(mt), std::ref(dist));
			}
			else {
				tasks[i] = std::async(async_initialize, std::ref(node), task_allocation * i, task_allocation * (i + 1), std::ref(mt), std::ref(dist));
			}
		}
	}

	void naive_heInitialize(Node& before) {
		std::mt19937 mt(seed);
		std::normal_distribution<float> dist(0, (float)std::sqrt(2.0 / before.size));

		for (int i = 0; i < size; ++i) {
			node[i] = dist(mt);
		}
	}
	//void heInitialize(Node& before, Node& after) {
	//	std::mt19937 mt(seed);
	//	std::normal_distribution<float> dist(0, (float)std::sqrt(2.0 / (before.size + after.size)));
	//	async_initialize();
	//}

	void print() {
		for (int i = 0; i < row; ++i) {
			for (int j = 0; j < col; ++j) {
				std::cout << node[i * col + j] << " ";
			}
			std::cout << std::endl;
		}
	}


	~Node() {
		if (node != nullptr) {
			delete node;
			node = nullptr;
		}
	}
};