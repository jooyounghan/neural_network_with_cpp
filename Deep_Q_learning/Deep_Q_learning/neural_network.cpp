#include "neural_network.h"

void NeuralNetwork::link(Layer& layer) {
	if (layers.size()) {
		layer.front = layers[layers.size() - 1];
		layers[layers.size() - 1]->rear = &layer;
	}
	layers.push_back(&layer);
}
