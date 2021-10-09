#include "node_async_function.h"
#include "node.h"

#define SEED	100

Node::Node() {
	this->row = 0;
	this->col = 0;
	this->size = 0;
}

Node::Node(const int& row_in, const int& col_in)
	: row(row_in), col(col_in), size(row_in* col_in) {
	this->node = new float[size];
}

Node::Node(const int& row_in, const int& col_in, float*& data)
	: row(row_in), col(col_in), size(row_in* col_in), node(data) {
	data = nullptr;
}

void Node::heInitialize(Node& before) {
	std::mt19937 mt(SEED);
	std::normal_distribution<float> dist(0, (float)std::sqrt(2.0 / before.size));

	std::vector<std::future<void>> tasks;
	tasks.resize(cores);

	const int& task_allocation = size / cores;

	for (int i = 0; i < cores; ++i) {
		if (i == cores - 1) {
			tasks[i] = std::async(asyncInitialize, std::ref(node), task_allocation * i, size,
				std::ref(mt), std::ref(dist));
		}
		else {
			tasks[i] = std::async(asyncInitialize, std::ref(node), task_allocation * i, task_allocation * (i + 1),
				std::ref(mt), std::ref(dist));
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

void Node::baseInitialize() {
	for (int i = 0; i < size; ++i) {
		node[i] = i;
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
			tasks[i] = std::async(asyncInitialize, std::ref(node), task_allocation * i, size,
				std::ref(mt), std::ref(dist));
		}
		else {
			tasks[i] = std::async(asyncInitialize, std::ref(node), task_allocation * i, task_allocation * (i + 1),
				std::ref(mt), std::ref(dist));
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

void Node::nodeMatMul(Node& node_in, Node& w_in) {
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
			tasks[i] = std::async(asyncMatMul, std::ref(this->node), std::ref(node_in.node), std::ref(w_in.node),
				node_in.row, node_in.col, w_in.col, task_allocation * i, size);
		}
		else {
			tasks[i] = std::async(asyncMatMul, std::ref(this->node), std::ref(node_in.node), std::ref(w_in.node),
				node_in.row, node_in.col, w_in.col, task_allocation * i, task_allocation * (i + 1));
		}
	}
}

void Node::naiveNodeMatMul(Node& node_in, Node& w_in) {
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
			tasks[i] = std::async(asyncIsSame, std::ref(this->node), std::ref(node_in.node),
				task_allocation * i, size);
		}
		else {
			tasks[i] = std::async(asyncIsSame, std::ref(this->node), std::ref(node_in.node),
				task_allocation * i, task_allocation * (i + 1));
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
			if (this->node[i * this->col + j] < node_in.node[i * this->col + j] + ERR &&
				this->node[i * this->col + j] > node_in.node[i * this->col + j] - ERR) {
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
			tasks[i] = std::async(asyncRelu, std::ref(this->node), std::ref(node_in.node),
				task_allocation * i, size);
		}
		else {
			tasks[i] = std::async(asyncRelu, std::ref(this->node), std::ref(node_in.node),
				task_allocation * i, task_allocation * (i + 1));
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

void Node::copySemantic(Node& node_in) {
	if (this->node != nullptr) {
		delete[] this->node;
		this->node = nullptr;
	}
	this->row = node_in.row;
	this->col = node_in.col;
	this->size = node_in.size;
	this->node = new float[size];

	std::vector<std::future<void>> tasks;
	tasks.resize(cores);

	const int& task_allocation = size / cores;

	for (int i = 0; i < cores; ++i) {
		if (i == cores - 1) {
			tasks[i] = std::async(asyncCopy, std::ref(this->node), std::ref(node_in.node),
				task_allocation * i, size);
		}
		else {
			tasks[i] = std::async(asyncCopy, std::ref(this->node), std::ref(node_in.node),
				task_allocation * i, task_allocation * (i + 1));
		}
	}
}

void Node::naiveCopySemantic(Node& node_in) {
	if (this->node != nullptr) {
		delete[] this->node;
		this->node = nullptr;
	}
	this->row = node_in.row;
	this->col = node_in.col;
	this->size = node_in.size;
	this->node = new float[size];
	for (int i = 0; i < size; ++i) {
		this->node[i] = node_in.node[i];
	}
}

void Node::moveSemantic(Node& node_in) {
	if (this->node != nullptr) {
		delete[] this->node;
		this->node = nullptr;
	}
	this->node = node_in.node;
	node_in.node = nullptr;
	return;
}

void Node::moveSemantic(Node&& node_in) {
	if (this->node != nullptr) {
		delete[] this->node;
		this->node = nullptr;
	}
	this->node = node_in.node;
	node_in.node = nullptr;
	return;
}

Node Node::transpose() {
	float* new_node = new float[this->size];
	std::vector<std::future<void>> tasks;
	tasks.resize(cores);

	const int& task_allocation = size / cores;

	for (int i = 0; i < cores; ++i) {
		if (i == cores - 1) {
			tasks[i] = std::async(asyncTranspose, std::ref(new_node), std::ref(this->node),
				this->row, this->col, task_allocation * i, size);
		}
		else {
			tasks[i] = std::async(asyncTranspose, std::ref(new_node), std::ref(this->node),
				this->row, this->col, task_allocation * i, task_allocation * (i + 1));
		}
	}
	for (int i = 0; i < cores; ++i) {
		tasks[i].wait();
	}
	return Node(this->col, this->row, new_node);
}

Node Node::naiveTranspose() {
	float* new_node = new float[this->size];
	for (int r = 0; r < row; ++r) {
		for (int c = 0; c < col; ++c) {
			new_node[c * this->row + r] = this->node[r * this->col + c];
		}
	}
	return Node(this->col, this->row, new_node);
}

void Node::nodeElementWiseSubtract(Node& node_from, Node& node_sub) {
	if (node_from.row != node_sub.row || node_from.col != node_sub.col
		|| this->row != node_from.row || this->col != node_from.col) {
		std::cout << "for matrix subtract, matrix sizes have to be same" << std::endl;
		assert(false);
		return;
	}

	std::vector<std::future<void>> tasks;
	tasks.resize(cores);

	const int& task_allocation = size / cores;

	for (int i = 0; i < cores; ++i) {
		if (i == cores - 1) {
			tasks[i] = std::async(asyncNodeSubtract, std::ref(this->node), std::ref(node_from.node),
				std::ref(node_sub.node), task_allocation * i, size);
		}
		else {
			tasks[i] = std::async(asyncNodeSubtract, std::ref(this->node), std::ref(node_from.node),
				std::ref(node_sub.node), task_allocation * i, task_allocation * (i + 1));
		}
	}

}

void Node::naiveNodeElementWiseSubtract(Node& node_from, Node& node_sub) {
	if (node_from.row != node_sub.row || node_from.col != node_sub.col ||
		this->row != node_from.row || this->col != node_from.col) {
		std::cout << "for matrix subtract, matrix sizes have to be same" << std::endl;
		assert(false);
		return;
	}
	for (int i = 0; i < this->size; ++i) {
		this->node[i] = node_from.node[i] - node_sub.node[i];
	}
	return;
}

void Node::constantMul(const float& num) {

	std::vector<std::future<void>> tasks;
	tasks.resize(cores);

	const int& task_allocation = size / cores;

	for (int i = 0; i < cores; ++i) {
		if (i == cores - 1) {
			tasks[i] = std::async(asyncConstantMul, std::ref(this->node),
				num, task_allocation * i, size);
		}
		else {
			tasks[i] = std::async(asyncConstantMul, std::ref(this->node),
				num, task_allocation * i, task_allocation * (i + 1));
		}
	}

}

void Node::naiveConstantMul(const float& num) {
	for (int i = 0; i < this->size; ++i) {
		this->node[i] = node[i] * num;
	}
	return;
}

Node Node::operator +(Node& node_in) {
	if (this->row != node_in.row || this->col != node_in.col) {
		std::cout << "for matrix addition, matrix sizes have to be same" << std::endl;
		assert(false);
		return;
	}

	float* result = new float[node_in.size];

	std::vector<std::future<void>> tasks;
	tasks.resize(cores);

	const int& task_allocation = node_in.size / cores;

	for (int i = 0; i < cores; ++i) {
		if (i == cores - 1) {
			tasks[i] = std::async(asyncNodeElementWiseAdd, std::ref(result), std::ref(this->node),
				std::ref(node_in.node), task_allocation * i, node_in.size);
		}
		else {
			tasks[i] = std::async(asyncNodeElementWiseAdd, std::ref(result), std::ref(this->node),
				std::ref(node_in.node), task_allocation * i, task_allocation * (i + 1));
		}
	}
	for (int i = 0; i < cores; ++i) {
		tasks[i].wait();
	}
	return Node(node_in.row, node_in.col, result);
}

Node Node::operator +(const Node& node_in) {
	if (this->row != node_in.row || this->col != node_in.col) {
		std::cout << "for matrix addition, matrix sizes have to be same" << std::endl;
		assert(false);
		return;
	}

	float* result = new float[node_in.size];

	std::vector<std::future<void>> tasks;
	tasks.resize(cores);

	const int& task_allocation = node_in.size / cores;

	for (int i = 0; i < cores; ++i) {
		if (i == cores - 1) {
			tasks[i] = std::async(asyncNodeElementWiseAdd, std::ref(result), std::ref(this->node),
				std::ref(node_in.node), task_allocation * i, node_in.size);
		}
		else {
			tasks[i] = std::async(asyncNodeElementWiseAdd, std::ref(result), std::ref(this->node),
				std::ref(node_in.node), task_allocation * i, task_allocation * (i + 1));
		}
	}
	for (int i = 0; i < cores; ++i) {
		tasks[i].wait();
	}
	return Node(node_in.row, node_in.col, result);
}

Node& Node::operator =(Node& node_in) {

	this->copySemantic(node_in);
	return *this;
}

Node& Node::operator =(Node&& node_in) {
	this->row = node_in.row;
	this->col = node_in.col;
	this->size = node_in.size;
	if (this->node != nullptr) {
		delete[] this->node;
		this->node = nullptr;
	}
	this->node = node_in.node;
	node_in.node = nullptr;
	return *this;
}


void Node::print() {
	for (int i = 0; i < row; ++i) {
		for (int j = 0; j < col; ++j) {
			std::cout << node[i * col + j] << " ";
		}
		std::cout << std::endl;
	}
}

Node::~Node() {
	if (node != nullptr) {
		delete node;
		node = nullptr;
	}
}

Node operator *(const float& num, Node& node_in) {
	float* result = new float[node_in.size];

	std::vector<std::future<void>> tasks;
	tasks.resize(cores);

	const int& task_allocation = node_in.size / cores;

	for (int i = 0; i < cores; ++i) {
		if (i == cores - 1) {
			tasks[i] = std::async(asyncConstantMat, std::ref(result), std::ref(node_in.node),
				num, task_allocation * i, node_in.size);
		}
		else {
			tasks[i] = std::async(asyncConstantMat, std::ref(result), std::ref(node_in.node),
				num, task_allocation * i, task_allocation * (i + 1));
		}
	}
	for (int i = 0; i < cores; ++i) {
		tasks[i].wait();
	}
	return Node(node_in.row, node_in.col, result);
}
