#include "neural_network.h"

void NeuralNetwork::link(Layer& layer) {
	if (layers.size()) {
		layer.front = layers[layers.size() - 1];
		layers[layers.size() - 1]->rear = &layer;
	}
	layers.push_back(&layer);
}

void NeuralNetwork::checkLayer() {
	for (Layer* layer : layers) {
		std::cout << layer->front << " / " << layer << " / " << layer->rear << std::endl;
	}
}

void NeuralNetwork::forwardPropagate(Node& Node_in) {
	layers[0]->setInput(Node_in);
	if (layers[0]->output == nullptr) {
		for (int idx = 0; idx < layers.size(); ++idx) {
			layers[idx]->setOutput();
			if (idx != layers.size() - 1) {
				layers[idx]->linkInputOutput();
			}
		}
	}

	for (int idx = 0; idx < layers.size(); ++idx) {
		layers[idx]->forward();
	}
}
