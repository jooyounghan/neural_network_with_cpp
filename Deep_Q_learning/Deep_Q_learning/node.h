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
	Node();
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

	void copySemantic(Node& node_in);
	void naiveCopySemantic(Node& node_in);

	void moveSemantic(Node& node_in);
	void moveSemantic(Node&& node_in);

	Node transpose();
	Node naiveTranspose();

	void elementSubtract(Node& node_from, Node& node_sub);
	void naiveElementSubtract(Node& node_from, Node& node_sub);

	Node& operator =(Node& node_in);
	Node& operator = (Node&& node_in);

	friend Node& operator *(const float& num, Node& node_in);
	friend Node operator +(Node& node_in_1, Node& node_in_2);

	void print();

	~Node();
};

