#pragma once
#include "layer.h"
#include <vector>



class NeuralNetwork {

public:
	std::vector<Layer*> layers;

public:
	NeuralNetwork() {};
	
	void link(Layer& layer);

	void checkLayer() {
		for (Layer* layer : layers) {
			std::cout << layer->front << " / " << layer << " / " << layer->rear << std::endl;
		}
	}

};