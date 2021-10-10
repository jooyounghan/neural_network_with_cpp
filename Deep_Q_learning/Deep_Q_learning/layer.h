#pragma once
#include "optimizer.h"

#define HE_INITIALIZE		1
#define XAVIER_INITIALIZE	2

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
	virtual void naiveForward() = 0;
	virtual void naiveBackward() = 0;
	virtual void weightInitialize(const int& mode, Node& node_in) = 0;
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
	virtual void backward() override;
	virtual void naiveForward() override;
	virtual void naiveBackward() override;
	virtual void weightInitialize(const int& mode, Node& node_in);
	~HiddenLayer();
};


class Relu : public Layer {
public:
	virtual void setInput(Node& input) override;
	virtual void setOutput() override;

	virtual void setOptimizer(const int& mode, const float& lr, const float& constant1 = 0, const float& constant2 = 0, const float& constant3 = 0) override;

	virtual void forward() override;
	virtual void backward() override;
	virtual void naiveForward() override;
	virtual void naiveBackward() override;
	virtual void weightInitialize(const int& mode, Node& node_in);
};