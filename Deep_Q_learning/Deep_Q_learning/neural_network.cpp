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

void NeuralNetwork::setOptimizer(const int& mode, const float& constant1 = 0, const float& constant2 = 0) {
	for (Layer* layer : layers) {
		layer->setOptimizer(mode, constant1, constant2);
	}
}

void NeuralNetwork::forwardPropagate(Node& node_in) {
	layers[0]->setInput(node_in);
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

void NeuralNetwork::backwardPropagate(Node& node_label) {
	layers[layers.size() - 1]->output->elementSubtract(*layers[layers.size() - 1]->output, node_label);
	for (int idx = layers.size() - 1; idx >= 0; --idx) {
		layers[idx]->backward();
	}
}

void NeuralNetwork::getInput() {
	layers[0]->input->print();
}

void NeuralNetwork::getOutput() {
	layers[layers.size() - 1]->output->print();
}
