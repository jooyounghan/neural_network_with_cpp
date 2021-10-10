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

	void setInput(Node& node_in);
	void setOptimizer(const int& mode, const float& lr_in, const float& constant1 = 0, const float& constant2 = 0, const float& constant3 = 0);

	void weightInitialize(const int& mode);

	void forwardPropagate();
	void backwardPropagate(Node& node_label);
	void naiveForwardPropagate();
	void naiveBackwardPropagate(Node& node_label);

	void getInput();
	void getOutput();
};