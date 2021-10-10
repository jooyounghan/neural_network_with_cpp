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

void NeuralNetwork::setOptimizer(const int& mode, const float& lr_in, const float& constant1, const float& constant2, const float& constant3) {
	for (Layer* layer : layers) {
		layer->setOptimizer(mode, lr_in, constant1, constant2, constant3);
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
	layers[layers.size() - 1]->output->nodeElementWiseSubtract(*layers[layers.size() - 1]->output, node_label);
	// receive parameter from output Node and send it to input Node at backpropagtion.
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
