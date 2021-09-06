#pragma once
#include "Variable.h"
class Optimizer
{
public:
	float lr;
	Variable gradient;
	/*
		Optimizer Class with 
		Momentum / NAG / AdaGrad / RMSprop / Adam
	*/

public:
	Optimizer(const float& lr_in) : lr(lr_in), gradient(Variable()) {}
	virtual Variable optimize(Variable& present_gradient) = 0;
};

class GradientDescent : public Optimizer
{
public:
	GradientDescent(const float& lr_in) : Optimizer(lr_in){}

	virtual Variable optimize(Variable& present_gradient) override {
		present_gradient = (-lr * present_gradient);
		gradient.move(present_gradient);
		return gradient;
	}

};

class Momentum : public Optimizer
{
public:
	float momentum_alpha;

public:
	Momentum(const float& ma_in, const float& lr_in) : momentum_alpha(ma_in), Optimizer(lr_in) {}

	virtual Variable optimize(Variable& present_gradient) override {
		present_gradient = -(lr * present_gradient);
		if (gradient.data == nullptr) {
			gradient.move(present_gradient);
			return gradient;
		}
		present_gradient = momentum_alpha * gradient + present_gradient;
		gradient.move(present_gradient);
		return gradient;
	}
};

class NAG : public Optimizer
{
public:
	float momentum_alpha;

public:
	NAG(const float& ma_in, const float& lr_in) : momentum_alpha(ma_in), Optimizer(lr_in) {}

	virtual Variable optimize(Variable& present_gradient) override {
		present_gradient = -(lr * present_gradient);
		if (gradient.data == nullptr) {
			gradient.move(present_gradient);
			return gradient;
		}
		present_gradient = momentum_alpha * gradient + present_gradient;
		Variable result = (momentum_alpha * momentum_alpha) * gradient + (1 + momentum_alpha) * present_gradient;
		gradient.move(present_gradient);		
		return result;
	}
};

class Adagrad : public Optimizer
{
public:
	float G;
	float delta;

public:
	Adagrad(const float& delta_in, const float& lr_in) : G(0), delta(delta_in), Optimizer(lr_in) {}

	virtual Variable optimize(Variable& present_gradient) override {
		G = G + present_gradient.element_mul(present_gradient).sum();
		Variable result = (-lr / (delta + std::sqrt(G))) * present_gradient;
		return result;
	}
};

class RMSprop : public Optimizer
{
public:
	float H;
	float delta;
	float gamma;

public:
	RMSprop(const float& delta_in, const float& gamma_in, const float& lr_in) : H(0), delta(delta_in), gamma(gamma_in), Optimizer(lr_in) {}

	virtual Variable optimize(Variable& present_gradient) override {
		H = gamma * H + (1 - gamma) * present_gradient.element_mul(present_gradient).sum();
		Variable result = (-lr / (delta + std::sqrt(H))) * present_gradient;
		return result;
	}
};


class Adam : public Optimizer
{
public:
	int t = 1;
	float delta;
	float beta1;
	float beta2;
	Variable s;
	Variable r;

public:
	Adam(const float& delta_in, const float& beta1_in, const float& beta2_in, const float& lr_in) :
		delta(delta_in), beta1(beta1_in), beta2(beta2_in), s(Variable()), r(Variable()), Optimizer(lr_in) {}

	virtual Variable optimize(Variable& present_gradient) override {
		if (t == 1) {
			s = (1 - beta1) * present_gradient;
			r = (1 - beta2) * present_gradient.element_mul(present_gradient);
		}
		else {
			s = beta1 * s + (1 - beta1) * present_gradient;
			r = beta2 * r + (1 - beta2) * present_gradient.element_mul(present_gradient);
		}
		Variable shat = s / (1 - std::pow(beta1, t));
		Variable rhat = r / (1 - std::pow(beta2, t));
		Variable result = (-lr / (delta + rhat.sqrt())).element_mul(shat);
		t++;
		return result;
	}
};