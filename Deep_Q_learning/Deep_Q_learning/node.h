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
	Node(const Node& node_in);
	Node(const int& row_in, const int& col_in);
	Node(const int& row_in, const int& col_in, float*& data);
	void heInitialize(Node& before);
	void naiveHeInitialize(Node& before);

	void baseInitialize();

	void xavierInitialize(Node& before, Node& after);
	void naiveXavierInitialize(Node& before, Node& after);

	void nodeMatMul(Node& node_in, Node& w_in);
	void naiveNodeMatMul(Node& node_in, Node& w_in);

	bool isSame(Node& node_in);
	bool naiveIsSame(Node& node_in);

	void relu(Node& node_in);
	void naiveRelu(Node& node_in);

	void square();
	void naiveSquare();

	Node getSquare();
	Node naiveGetSquare();
	Node getSqrt();
	Node naiveGetSqrt();
	Node getFraction();
	Node naiveGetFraction();

	void copySemantic(Node& node_in);
	void copySemantic(const Node& node_in);
	void naiveCopySemantic(Node& node_in);
	void naiveCopySemantic(const Node& node_in);

	void moveSemantic(Node& node_in);
	void moveSemantic(Node&& node_in);

	Node getTranspose();
	Node naiveGetTranspose();

	void nodeElementWiseSubtract(Node& node_from, Node& node_sub);
	void naiveNodeElementWiseSubtract(Node& node_from, Node& node_sub);

	Node getNodeElementWiseSubtract(Node& node_in);
	Node naiveGetNodeElementWiseSubtract(Node& node_in);
	Node getElementWiseMul(Node& node_in);
	Node getElementWiseMul(Node&& node_in);
	Node naiveGetElementWiseMul(Node& node_in);
	Node naiveGetElementWiseMul(Node&& node_in);

	void constantMul(const float& num);
	void naiveConstantMul(const float& num);
	
	Node getConstantMul(const float& num);
	Node naiveGetConstantMul(const float& num);

	Node operator +(Node& node_in_2);
	Node operator +(const Node& node_in_2);
	Node operator +(const float& num);

	Node& operator =(Node& node_in);
	Node& operator = (Node&& node_in);

	friend Node operator *(const float& num, Node& node_in);
	friend Node operator *(const float& num, Node&& node_in);

	void print();

	~Node();
};

