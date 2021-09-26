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

	void forwardPropagate(Node& Node_in);

	void getOutput() {
		layers[layers.size() - 1]->output->print();
	}
};