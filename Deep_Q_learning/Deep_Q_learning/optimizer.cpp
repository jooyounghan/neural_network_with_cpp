#include "optimizer.h"

void GradientDescent::optimize(Node& present_gradient) {
	this->past_gradient.moveSemantic(present_gradient);
	this->past_gradient.constantMul(-lr);
	return;
}

void GradientDescent::naiveOptimize(Node& present_gradient) {
	this->past_gradient.moveSemantic(present_gradient);
	this->past_gradient.naiveConstantMul(-lr);
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

void Momentum::naiveOptimize(Node& present_gradient) {
	if (this->past_gradient.node == nullptr) {
		this->past_gradient.moveSemantic(present_gradient);
		this->past_gradient.naiveConstantMul(-lr);
		return;
	}
	present_gradient.naiveConstantMul(-lr);
	past_gradient = momentum_alpha * past_gradient + present_gradient;
	return;
}

Node& Momentum::getGradient() {
	return past_gradient;
}

void Nag::optimize(Node& present_gradient) {
	if (this->past_gradient.node == nullptr || this->gradient.node == nullptr) {
		if (this->gradient.node == nullptr) {
			this->gradient.copySemantic(present_gradient);
			this->gradient.constantMul(-lr);
		}
		if (this->past_gradient.node == nullptr) {
			this->past_gradient.moveSemantic(present_gradient);
			this->past_gradient.constantMul(-lr);
		}
		return;
	}
	present_gradient.constantMul(-lr);
	present_gradient = momentum_alpha * past_gradient + present_gradient;
	gradient = (momentum_alpha * momentum_alpha) * past_gradient + (1 + momentum_alpha) * present_gradient;
	past_gradient.moveSemantic(present_gradient);
	return;
}

void Nag::naiveOptimize(Node& present_gradient) {
	if (this->past_gradient.node == nullptr || this->gradient.node == nullptr) {
		if (this->gradient.node == nullptr) {
			this->gradient.naiveCopySemantic(present_gradient);
			this->gradient.naiveConstantMul(-lr);
		}
		if (this->past_gradient.node == nullptr) {
			this->past_gradient.moveSemantic(present_gradient);
			this->past_gradient.naiveConstantMul(-lr);
		}
		return;
	}
	present_gradient.naiveConstantMul(-lr);
	present_gradient = momentum_alpha * past_gradient + present_gradient;
	gradient = (momentum_alpha * momentum_alpha) * past_gradient + (1 + momentum_alpha) * present_gradient;
	past_gradient.moveSemantic(present_gradient);
	return;
}

Node& Nag::getGradient() {
	return gradient;
}

void Adagrad::optimize(Node& present_gradient) {
	if (this->past_gradient.node == nullptr || this->g_node.node == nullptr) {
		if (this->g_node.node == nullptr) {
			this->g_node.copySemantic(present_gradient);
			this->g_node.square();
		}
		if (this->past_gradient.node == nullptr) {
			this->past_gradient = (g_node.getSqrt() + delta).getFraction().getElementWiseMul(present_gradient);
		}
		return;
	}
	g_node = g_node + present_gradient.getSquare();
	past_gradient = (g_node.getSqrt() + delta).getFraction().getElementWiseMul(present_gradient);
	past_gradient.constantMul(-lr);
	return;
}

void Adagrad::naiveOptimize(Node& present_gradient) {
	if (this->past_gradient.node == nullptr || this->g_node.node == nullptr) {
		if (this->g_node.node == nullptr) {
			this->g_node.naiveCopySemantic(present_gradient);
			this->g_node.naiveSquare();
		}
		if (this->past_gradient.node == nullptr) {
			this->past_gradient = (g_node.naiveGetSqrt() + delta).naiveGetFraction().naiveGetElementWiseMul(present_gradient);
		}
		return;
	}
	g_node = g_node + present_gradient.naiveGetSquare();
	past_gradient = (g_node.naiveGetSqrt() + delta).naiveGetFraction().naiveGetElementWiseMul(present_gradient);
	past_gradient.naiveConstantMul(-lr);
	return;
}

Node& Adagrad::getGradient() {
	return past_gradient;
}

void Rmsprop::optimize(Node& present_gradient) {
	if (this->past_gradient.node == nullptr || this->h_node.node == nullptr) {
		if (this->h_node.node == nullptr) {
			this->h_node.copySemantic(present_gradient);
			this->h_node.square();
		}
		if (this->past_gradient.node == nullptr) {
			this->past_gradient = (h_node.getSqrt() + delta).getFraction().getElementWiseMul(present_gradient);
		}
		return;
	}
	h_node = gamma * h_node + (1 - gamma) * present_gradient.getSquare();
	past_gradient = (h_node.getSqrt() + delta).getFraction().getElementWiseMul(present_gradient);
	past_gradient.constantMul(-lr);
	return;
}

void Rmsprop::naiveOptimize(Node& present_gradient) {
	if (this->past_gradient.node == nullptr || this->h_node.node == nullptr) {
		if (this->h_node.node == nullptr) {
			this->h_node.naiveCopySemantic(present_gradient);
			this->h_node.naiveSquare();
		}
		if (this->past_gradient.node == nullptr) {
			this->past_gradient = (h_node.naiveGetSqrt() + delta).naiveGetFraction().naiveGetElementWiseMul(present_gradient);
		}
		return;
	}
	h_node = gamma * h_node + (1 - gamma) * present_gradient.naiveGetSquare();
	past_gradient = (h_node.naiveGetSqrt() + delta).naiveGetFraction().naiveGetElementWiseMul(present_gradient);
	past_gradient.naiveConstantMul(-lr);
	return;
}

Node& Rmsprop::getGradient() {
	return past_gradient;
}

void Adam::optimize(Node& present_gradient) {
	if (this->past_gradient.node == nullptr || this->s_node.node == nullptr || this->r_node.node == nullptr) {
		if (this->s_node.node == nullptr) {
			this->s_node.copySemantic(present_gradient);
			this->s_node.constantMul(1 - rho1);
		}
		if (this->r_node.node == nullptr) {
			this->r_node.copySemantic(present_gradient);
			this->r_node.square();
			this->r_node.constantMul(1 - rho2);
		}
		if (this->past_gradient.node == nullptr) {
			this->past_gradient = (r_node.getConstantMul(1 / (1 - std::pow(rho2, t))).getSqrt() + delta).getFraction().getElementWiseMul(s_node.getConstantMul(1 / (1 - std::pow(rho1, t))));
			this->past_gradient.constantMul(-lr);
		}
		t++;
		return;
	}
	s_node = rho1 * s_node + (1 - rho1) * present_gradient;
	r_node = rho2 * r_node + (1 - rho2) * present_gradient.getSquare();
	this->past_gradient = (r_node.getConstantMul(1 / (1 - std::pow(rho2, t))).getSqrt() + delta).getFraction().getElementWiseMul(s_node.getConstantMul(1 / (1 - std::pow(rho1, t))));
	this->past_gradient.constantMul(-lr);
	t++;
	return;
}

void Adam::naiveOptimize(Node& present_gradient) {
	if (this->past_gradient.node == nullptr || this->s_node.node == nullptr || this->r_node.node == nullptr) {
		if (this->s_node.node == nullptr) {
			this->s_node.naiveCopySemantic(present_gradient);
			this->s_node.naiveConstantMul(1 - rho1);
		}
		if (this->r_node.node == nullptr) {
			this->r_node.naiveCopySemantic(present_gradient);
			this->r_node.naiveSquare();
			this->r_node.naiveConstantMul(1 - rho2);
		}
		if (this->past_gradient.node == nullptr) {
			this->past_gradient = (r_node.naiveGetConstantMul(1 / (1 - std::pow(rho2, t))).naiveGetSqrt() + delta).naiveGetFraction().naiveGetElementWiseMul(s_node.naiveGetConstantMul(1 / (1 - std::pow(rho1, t))));
			this->past_gradient.naiveConstantMul(-lr);
		}
		t++;
		return;
	}
	s_node = rho1 * s_node + (1 - rho1) * present_gradient;
	r_node = rho2 * r_node + (1 - rho2) * present_gradient.naiveGetSquare();
	this->past_gradient = (r_node.naiveGetConstantMul(1 / (1 - std::pow(rho2, t))).naiveGetSqrt() + delta).naiveGetFraction().naiveGetElementWiseMul(s_node.naiveGetConstantMul(1 / (1 - std::pow(rho1, t))));
	this->past_gradient.naiveConstantMul(-lr);
	t++;
	return;
}

Node& Adam::getGradient() {
	return past_gradient;
}
