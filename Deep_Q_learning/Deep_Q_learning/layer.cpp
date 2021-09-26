#include "layer.h"

void Layer::linkInputOutput() {
	this->rear->input = output;
}

HiddenLayer::HiddenLayer(Node& w_in, Optimizer* opt_in) : w(w_in), opt(opt_in) {}

void HiddenLayer::setInput(Node& input) {
	if (front == nullptr) {
		this->input = &input;
	}
	else {
		std::cout << "Can't set Input to this Layer" << "\n";
		return;
	}
}

void HiddenLayer::setOutput() {
	this->output = new Node(input->row, w.col);
}

void HiddenLayer::forward() {
	output->matMul(*input, w);
}

