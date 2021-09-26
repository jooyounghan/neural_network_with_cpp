#pragma once
#include "node.h"
#include "optimizer.h"


class Layer {
public:
	Layer* front = nullptr;
	Layer* rear = nullptr;
	Node* input = nullptr;
	Node* output = nullptr;

public:
	virtual void setInput(Node& input) = 0;
	virtual void setOutput() = 0;
	virtual void forward() = 0;
	virtual void backward(const float& lr) = 0;

	void linkInputOutput();


	/*
	Layer destructor only delete output(Node*), because every input is linked with the output
	and only initial layer has the input which is not linked.
	and this input(that is not linked with the front layer's output) doesn't need to deleted by destructor
	because it has it's own destructor(it is now allocated dynamically)
	*/
	~Layer() {
		if (output != nullptr) {
			delete output;
			output = nullptr;
		}
	}
};

class HiddenLayer : public Layer {
public:
	Node& w;
	Optimizer* opt;

public:
	HiddenLayer(Node& w_in, Optimizer* opt_in);

	virtual void setInput(Node& input) override;
	virtual void setOutput() override;

	virtual void forward() override;
	virtual void backward(const float& lr) override {

	}
};


class Relu : public Layer {
public:
	virtual void setInput(Node& input) override {
		if (front == nullptr) {
			this->input = &input;
		}
		else {
			std::cout << "Can't set Input to this Layer" << "\n";
			return;
		}
	}

	virtual void setOutput() override {
		this->output = new Node(input->row, input->col);
		this->rear->input = output;
	}


	virtual void forward() override {
		output->relu(*input);
	}

	virtual void backward(const float& lr) override {

	}
};