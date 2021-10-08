#include "optimizer.h"

Node& Optimizer::getGradient() {
	return this->past_gradient;
}

void GradientDescent::optimize(Node& present_gradient) {
	this->past_gradient.moveSemantic(present_gradient);
	return;
}

void Momentum::optimize(Node& present_gradient) {
	if (this->past_gradient.node == nullptr) {
		this->past_gradient.moveSemantic(present_gradient);
		return;
	}
	past_gradient = momentum_alpha * past_gradient + (1 - momentum_alpha) * present_gradient;
	return;
}
