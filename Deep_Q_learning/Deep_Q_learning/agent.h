#pragma once
#include <vector>
#include "neural_network.h"

class AI {
public:
	NeuralNetwork nn;

public:
	void setInput(const std::vector<int>& current_state) {
		// give current_state to nn as an input;
	}

	int getActions() {
		//nn.forward();
		//std::vector<float> q_values = nn.getOutput();

		//action = q_values.getMax();
	}

	void train(const float& reward) {
		//nn.train(reward);
	}
};