#include "layer.h"

void Layer::linkInputOutput() {
	this->rear->input = output;
}

HiddenLayer::HiddenLayer(Node& w_in) : w(w_in), opt(nullptr) {}

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

void HiddenLayer::setOptimizer(const int& mode) {
	if (this->opt != nullptr) {
		delete this->opt;
		opt = nullptr;
	}

	switch (mode)
	{
		case GRADIENT_DESCENT:
			this->opt = new GradientDescent;
		case MOMENTUM:
			this->opt = new Momentum;
		case NAG:
			this->opt = new Nag;
		case ADAGRAD:
			this->opt = new Adagrad;
		case RMSPROP:
			this->opt = new Rmsprop;
		case ADAM:
			this->opt = new Adam;
		default:
			std::cout << "Inappropriate Mode" << std::endl;
			assert(false);
	}
	return;
}

void HiddenLayer::forward() {
	output->matMul(*input, w);
}




void Relu::setInput(Node& input) {
	if (front == nullptr) {
		this->input = &input;
	}
	else {
		std::cout << "Can't set Input to this Layer" << "\n";
		return;
	}
}

void Relu::setOutput() {
	this->output = new Node(input->row, input->col);
	this->rear->input = output;
}

void Relu::setOptimizer(const int& mode) {
	return;
}

void Relu::forward() {
	output->relu(*input);
}
