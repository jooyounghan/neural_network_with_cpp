#pragma once
#include <iostream>
#include <cassert>

class Node {

public:
	int row;
	int col;
	int size;
	float* node = nullptr;

public:
	Node(const int& row_in, const int& col_in);
	Node(const int& row_in, const int& col_in, float*& data);
	void heInitialize(Node& before);
	void naiveHeInitialize(Node& before);

	void baseInitialize();

	void xavierInitialize(Node& before, Node& after);
	void naiveXavierInitialize(Node& before, Node& after);

	void matMul(Node& node_in, Node& w_in);
	void naiveMatMul(Node& node_in, Node& w_in);

	bool isSame(Node& node_in);
	bool naiveIsSame(Node& node_in);

	void relu(Node& node_in);
	void naiveRelu(Node& node_in);

	Node transpose();
	Node naiveTranspose();

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