#pragma once
#include "layer.h"
#include <vector>

class NeuralNetwork {

public:
	std::vector<Layer*> layers;

public:
	NeuralNetwork() {};
	
	void link(Layer& layer);
	void checkLayer();

	void setOptimizer(const int& mode);

	void forwardPropagate(Node& node_in);

	void backwardPropagate(Node& node_label);

	void getInput();
	void getOutput();
};