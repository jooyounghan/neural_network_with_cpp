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

void NeuralNetwork::setInputSize(Node& node_in) {
	layers[0]->setInputSize(node_in);
	if (layers[0]->output == nullptr) {
		for (int idx = 0; idx < layers.size(); ++idx) {
			layers[idx]->setOutput();
			if (idx != layers.size() - 1) {
				layers[idx]->linkInputOutput();
			}
		}
	}
}

void NeuralNetwork::setInputSize(const int& data_dimension, const int& data_num) {
	layers[0]->input = new Node(data_dimension, data_num);
	if (layers[0]->output == nullptr) {
		for (int idx = 0; idx < layers.size(); ++idx) {
			layers[idx]->setOutput();
			if (idx != layers.size() - 1) {
				layers[idx]->linkInputOutput();
			}
		}
	}
}

void NeuralNetwork::setOptimizer(const int& mode, const float& lr_in, const float& constant1, const float& constant2, const float& constant3) {
	for (Layer* layer : layers) {
		layer->setOptimizer(mode, lr_in, constant1, constant2, constant3);
	}
}

void NeuralNetwork::weightInitialize(const int& mode) {
	for (Layer* layer : layers) {
		layer->weightInitialize(mode);
	}
}

void NeuralNetwork::train(const int& iteration, std::vector<float> input_data, std::vector<float> label_data, const int& batch_size) {
	std::vector<std::vector<float>> input_batch;
	std::vector<std::vector<float>> label_batch;
	int total_num = input_data.size();
	int batch_num = input_data.size() / batch_size + 1;
	for (int batch_idx = 0; batch_idx < batch_num; ++batch_idx) {
		input_batch[batch_idx].resize(batch_size);
		label_batch[batch_idx].resize(batch_size);
		for (int idx = 0; idx < batch_size; ++idx) {
			input_batch[batch_idx][idx] = input_data[(batch_size * batch_idx + idx) % total_num];
			label_batch[batch_idx][idx] = label_data[(batch_size * batch_idx + idx) % total_num];
		}
	}
	int cnt = 0;
	while (cnt < iteration) {
		if (cnt % 10) std::cout << "Train Progressing..." << '\n';
		for (int batch_turn = 0; batch_turn < batch_num; ++batch_turn) {
			Node input_node(input_batch[batch_turn]);
			Node label_node(label_batch[batch_turn]);
			this->layers[0]->input = &input_node;
			this->forwardPropagate();
			this->backwardPropagate(label_node);
		}
		cnt++;
	}
	std::cout << "Train Finished!" << '\n';
}

void NeuralNetwork::naiveTrain(const int& iteration, std::vector<float> input_data, std::vector<float> label_data, const int& batch_size) {
	std::vector<std::vector<float>> input_batch;
	std::vector<std::vector<float>> label_batch;
	int total_num = input_data.size();
	int batch_num = input_data.size() / batch_size + 1;
	for (int batch_idx = 0; batch_idx < batch_num; ++batch_idx) {
		input_batch[batch_idx].resize(batch_size);
		label_batch[batch_idx].resize(batch_size);
		for (int idx = 0; idx < batch_size; ++idx) {
			input_batch[batch_idx][idx] = input_data[(batch_size * batch_idx + idx) % total_num];
			label_batch[batch_idx][idx] = label_data[(batch_size * batch_idx + idx) % total_num];
		}
	}
	int cnt = 0;
	while (cnt < iteration) {
		if (cnt % 10) std::cout << "Train Progressing..." << '\n';
		for (int batch_turn = 0; batch_turn < batch_num; ++batch_turn) {
			Node input_node(input_batch[batch_turn]);
			Node label_node(label_batch[batch_turn]);
			this->layers[0]->input = &input_node;
			this->naiveForwardPropagate();
			this->naiveBackwardPropagate(label_node);
		}
		cnt++;
	}
	std::cout << "Train Finished!" << '\n';
}

void NeuralNetwork::forwardPropagate() {
	for (int idx = 0; idx < layers.size(); ++idx) {
		layers[idx]->forward();
	}
}

void NeuralNetwork::backwardPropagate(Node& node_label) {
	layers[layers.size() - 1]->output->nodeElementWiseSubtract(node_label, *layers[layers.size() - 1]->output);
	// receive parameter from output Node and send it to input Node at backpropagtion.
	for (int idx = layers.size() - 1; idx >= 0; --idx) {
		layers[idx]->backward();
	}
}

void NeuralNetwork::naiveForwardPropagate() {
	for (int idx = 0; idx < layers.size(); ++idx) {
		layers[idx]->naiveForward();
	}
}

void NeuralNetwork::naiveBackwardPropagate(Node& node_label) {
	layers[layers.size() - 1]->output->naiveNodeElementWiseSubtract(node_label, *layers[layers.size() - 1]->output);
	// receive parameter from output Node and send it to input Node at backpropagtion.
	for (int idx = layers.size() - 1; idx >= 0; --idx) {
		layers[idx]->naiveBackward();
	}
}

void NeuralNetwork::getInput() {
	layers[0]->input->print();
}

void NeuralNetwork::getOutput() {
	layers[layers.size() - 1]->output->print();
}
