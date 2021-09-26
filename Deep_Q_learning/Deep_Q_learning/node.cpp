#include "node_async_function.h"
#include "node.h"

#define SEED	100

Node::Node(const int& row_in, const int& col_in)
	: row(row_in), col(col_in), size(row_in* col_in) {
	node = new float[size];
}

void Node::heInitialize(Node& before) {
	std::mt19937 mt(SEED);
	std::normal_distribution<float> dist(0, (float)std::sqrt(2.0 / before.size));

	std::vector<std::future<void>> tasks;
	tasks.resize(cores);

	const int& task_allocation = size / cores;

	for (int i = 0; i < cores; ++i) {
		if (i == cores - 1) {
			tasks[i] = std::async(asyncInitialize, std::ref(node), task_allocation * i, size, std::ref(mt), std::ref(dist));
		}
		else {
			tasks[i] = std::async(asyncInitialize, std::ref(node), task_allocation * i, task_allocation * (i + 1), std::ref(mt), std::ref(dist));
		}
	}
}

void Node::naiveHeInitialize(Node& before) {
	std::mt19937 mt(SEED);
	std::normal_distribution<float> dist(0, (float)std::sqrt(2.0 / before.size));

	for (int i = 0; i < size; ++i) {
		node[i] = dist(mt);
	}
}

void Node::xavierInitialize(Node& before, Node& after) {
	std::mt19937 mt(SEED);
	std::normal_distribution<float> dist(0, (float)std::sqrt(2.0 / (before.size + after.size)));

	std::vector<std::future<void>> tasks;
	tasks.resize(cores);

	const int& task_allocation = size / cores;

	for (int i = 0; i < cores; ++i) {
		if (i == cores - 1) {
			tasks[i] = std::async(asyncInitialize, std::ref(node), task_allocation * i, size, std::ref(mt), std::ref(dist));
		}
		else {
			tasks[i] = std::async(asyncInitialize, std::ref(node), task_allocation * i, task_allocation * (i + 1), std::ref(mt), std::ref(dist));
		}
	}
}

void Node::naiveXavierInitialize(Node& before, Node& after) {
	std::mt19937 mt(SEED);
	std::normal_distribution<float> dist(0, (float)std::sqrt(2.0 / (before.size + after.size)));
	for (int i = 0; i < size; ++i) {
		node[i] = dist(mt);
	}
}

void Node::matMul(Node& node_in, Node& w_in) {
	if (node_in.row != this->row || w_in.col != this->col || node_in.col != w_in.row) {
		std::cout << "for matrix multification, (N x K) * (K x M) = (N x M) has to be met" << std::endl;
		assert(false);
		return;
	}

	std::vector<std::future<void>> tasks;
	tasks.resize(cores);

	const int& task_allocation = size / cores;

	for (int i = 0; i < cores; ++i) {
		if (i == cores - 1) {
			tasks[i] = std::async(asyncMatMul, std::ref(this->node), std::ref(node_in.node), std::ref(w_in.node), node_in.row, node_in.col, w_in.col, task_allocation * i, size);
		}
		else {
			tasks[i] = std::async(asyncMatMul, std::ref(this->node), std::ref(node_in.node), std::ref(w_in.node), node_in.row, node_in.col, w_in.col, task_allocation * i, task_allocation * (i + 1));
		}
	}
}

void Node::naiveMatMul(Node& node_in, Node& w_in) {
	if (node_in.row != this->row || w_in.col != this->col || node_in.col != w_in.row) {
		std::cout << "for matrix multification, (N x K) * (K x M) = (N x M) has to be met" << std::endl;
		assert(false);
		return;
	}

	for (int i = 0; i < this->row; ++i) {
		for (int j = 0; j < this->col; ++j) {
			float& temp_node = this->node[i * this->col + j];
			temp_node = 0;
			for (int k = 0; k < node_in.col; ++k) {
				temp_node += node_in.node[i * node_in.col + k] * w_in.node[k * w_in.col + j];
			}
		}
	}
}

bool Node::isSame(Node& node_in) {
	if (this->row != node_in.row || this->col != node_in.col) {
		std::cout << "Node sizes are different" << std::endl;
		assert(false);
		return false;
	}

	std::vector<std::future<bool>> tasks;
	tasks.resize(cores);

	const int& task_allocation = size / cores;

	for (int i = 0; i < cores; ++i) {
		if (i == cores - 1) {
			tasks[i] = std::async(asyncIsSame, std::ref(this->node), std::ref(node_in.node), task_allocation * i, size);
		}
		else {
			tasks[i] = std::async(asyncIsSame, std::ref(this->node), std::ref(node_in.node), task_allocation * i, task_allocation * (i + 1));
		}
	}

	bool result = true;
	for (int i = 0; i < tasks.size(); ++i) {
		result = result & tasks[i].get();
	}
	return result;
}

bool Node::naiveIsSame(Node& node_in) {
	if (this->row != node_in.row || this->col != node_in.col) {
		std::cout << "Node sizes are different" << std::endl;
		assert(false);
		return false;
	}

	for (int i = 0; i < this->row; ++i) {
		for (int j = 0; j < this->col; ++j) {
			if (this->node[i * this->col + j] < node_in.node[i * this->col + j] + ERR && this->node[i * this->col + j] > node_in.node[i * this->col + j] - ERR) {
				return false;
			}
		}
	}
	return true;
}

void Node::relu(Node& node_in) {
	if (this->size != node_in.size) {
		std::cout << "input and output sizes are different" << std::endl;
		assert(false);
		return;
	}

	std::vector<std::future<void>> tasks;
	tasks.resize(cores);

	const int& task_allocation = size / cores;

	for (int i = 0; i < cores; ++i) {
		if (i == cores - 1) {
			tasks[i] = std::async(asyncRelu, std::ref(this->node), std::ref(node_in.node), task_allocation * i, size);
		}
		else {
			tasks[i] = std::async(asyncRelu, std::ref(this->node), std::ref(node_in.node), task_allocation * i, task_allocation * (i + 1));
		}
	}
}

void Node::naiveRelu(Node& node_in) {
	if (this->size != node_in.size) {
		std::cout << "input and output sizes are different" << std::endl;
		assert(false);
		return;
	}
	for (int i = 0; i < this->size; ++i) {
		this->node[i] = this->node[i] > 0 ? this->node[i] : 0;
	}
}


