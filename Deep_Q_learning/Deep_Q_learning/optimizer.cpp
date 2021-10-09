#include "optimizer.h"

void GradientDescent::optimize(Node& present_gradient) {
	this->past_gradient.moveSemantic(present_gradient);
	this->past_gradient.constantMul(-lr);
	return;
}

Node& GradientDescent::getGradient() {
	return past_gradient;
}

void Momentum::optimize(Node& present_gradient) {
	if (this->past_gradient.node == nullptr) {
		this->past_gradient.moveSemantic(present_gradient);
		this->past_gradient.constantMul(-lr);
		return;
	}

	present_gradient.constantMul(-lr);
	past_gradient = momentum_alpha * past_gradient + present_gradient;
	return;
}

Node& Momentum::getGradient() {
	return past_gradient;
}

void Nag::optimize(Node& present_gradient) {
	if (this->past_gradient.node == nullptr || this->gradient.node == nullptr) {
		if (this->past_gradient.node == nullptr) {
			this->past_gradient.moveSemantic(present_gradient);
			this->past_gradient.constantMul(-lr);
		}
		if (this->gradient.node == nullptr) {
			this->gradient.copySemantic(past_gradient);
			this->gradient.constantMul(-lr);
		}
		return;
	}
	present_gradient.constantMul(-lr);
	present_gradient = momentum_alpha * past_gradient + present_gradient;
	gradient = (momentum_alpha * momentum_alpha) * past_gradient + (1 + momentum_alpha) * present_gradient;
	past_gradient.moveSemantic(present_gradient);
	return;
}

Node& Nag::getGradient() {
	return gradient;
}
