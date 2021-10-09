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

void HiddenLayer::setOptimizer(const int& mode, const float& lr, const float& constant1, const float& constant2, const float& constant3) {
	if (this->opt != nullptr) {
		delete this->opt;
		opt = nullptr;
	}
	switch (mode)
	{
		case GRADIENT_DESCENT:
			this->opt = new GradientDescent(lr);
			return;
		case MOMENTUM:
			this->opt = new Momentum(lr, constant1);
			return;
		case NAG:
			this->opt = new Nag(lr, constant1);
			return;
		case ADAGRAD:
			this->opt = new Adagrad(lr, constant1);
			return;
		case RMSPROP:
			this->opt = new Rmsprop(lr, constant1, constant2);
		case ADAM:
			this->opt = new Adam(lr, constant1, constant2, constant3);
		default:
			std::cout << "Inappropriate Mode" << std::endl;
			assert(false);
	}
	return;
}

void HiddenLayer::forward() {
	output->nodeMatMul(*input, w);
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

void Relu::setOptimizer(const int& mode, const float& lr, const float& constant1, const float& constant2, const float& constant3) {
	return;
}

void Relu::forward() {
	output->relu(*input);
}
