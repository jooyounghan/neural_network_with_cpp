#pragma once
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
	virtual void setOptimizer(const int& mode, const float& lr, const float& constant1 = 0, const float& constant2 = 0, const float& constant3 = 0) = 0;
	virtual void forward() = 0;
	virtual void backward() = 0;
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
	HiddenLayer(Node& w_in);

	virtual void setInput(Node& input) override;
	virtual void setOutput() override;

	virtual void setOptimizer(const int& mode, const float& lr, const float& constant1 = 0, const float& constant2 = 0, const float& constant3 = 0) override;

	virtual void forward() override;
	virtual void backward() override {

	}
	~HiddenLayer() {
		if (opt != nullptr) {
			delete opt;
			opt = nullptr;
		}
	}
};


class Relu : public Layer {
public:
	virtual void setInput(Node& input) override;
	virtual void setOutput() override;

	virtual void setOptimizer(const int& mode, const float& lr, const float& constant1 = 0, const float& constant2 = 0, const float& constant3 = 0) override;

	virtual void forward() override;
	virtual void backward() override {

	}
};