#pragma once
#include "node.h"
#include "optimizer.h"


class Layer {
public:
	Layer* rear = nullptr;
	Layer* front = nullptr;
	Node* input = nullptr;
	Node* output = nullptr;

public:
	virtual void forward() = 0;
	virtual void backward(const float& lr) = 0;
};

class HiddenLayer : public Layer {
public:
	Node& w;
	Optimizer* opt;

public:
	HiddenLayer(Node& w_in, Optimizer* opt_in) : w(w_in), opt(opt_in) {}

	void setInput(Node& input) {
		if (front == nullptr) {
			this->input = &input;
		}
		else {
			std::cout << "Can't set Input to this Layer" << "\n";
			return;
		}
	}

	virtual void forward() override {

	}


	virtual void backward(const float& lr) override {

	}
};


class Relu : public Layer {
public:
	virtual void forward() override {

	}

	virtual void backward(const float& lr) override {

	}
};