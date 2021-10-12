#include "layer.h"

void Layer::linkInputOutput() {
	this->rear->input = output;
}

HiddenLayer::HiddenLayer(Node& w_in) : w(w_in), opt(nullptr) {}

void HiddenLayer::setInputSize(Node& input) {
	if (front == nullptr) {
		this->input = &input;
	}
	else {
		std::cout << "Can't set Input to this Layer" << "\n";
		return;
	}
}

void HiddenLayer::setOutput() {
	this->output = new Node(w.row, input->col);
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
			return;
		case ADAM:
			this->opt = new Adam(lr, constant1, constant2, constant3);
			return;
		default:
			std::cout << "Inappropriate Mode" << std::endl;
			assert(false);
	}
	return;
}

void HiddenLayer::forward() {
	this->output->nodeMatMul(w, *input);
}

void HiddenLayer::backward() {
	Node present_gradient = this->output->getNodeMatMul(this->input->getTranspose());
	opt->optimize(present_gradient);
	w.nodeElementWiseAdd(w, opt->getGradient());
	this->input->moveSemantic(w.getTranspose().getNodeMatMul(*this->output));
}

void HiddenLayer::naiveForward() {
	this->output->naiveNodeMatMul(w, *input);
}

void HiddenLayer::naiveBackward() {
	Node present_gradient = this->output->naiveGetNodeMatMul(this->input->naiveGetTranspose());
	opt->naiveOptimize(present_gradient);
	w.naiveNodeElementWiseAdd(w, opt->getGradient());
	this->input->moveSemantic(w.naiveGetTranspose().naiveGetNodeMatMul(*this->output));
}

void HiddenLayer::weightInitialize(const int& mode) {
	switch (mode)
	{
	case HE_INITIALIZE:
		w.heInitialize(*input);
		return;
	case XAVIER_INITIALIZE:
		w.xavierInitialize(*input, *output);
		return;
	default:
		std::cout << "Inappropriate Mode" << std::endl;
		assert(false);
	}
	return;
}

HiddenLayer::~HiddenLayer() {
	if (opt != nullptr) {
		delete opt;
		opt = nullptr;
	}
}

void Relu::setInputSize(Node& input) {
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

void Relu::backward() {
	this->input->moveSemantic(this->input->getReluGradient().getElementWiseMul(*this->output));
}

void Relu::naiveForward() {
	output->naiveRelu(*input);
}

void Relu::naiveBackward() {
	this->input->moveSemantic(this->input->naiveGetReluGradient().naiveGetElementWiseMul(*this->output));
}

void Relu::weightInitialize(const int& mode) {
	return;
}
