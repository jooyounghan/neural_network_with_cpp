#pragma once
#include "Variable.h"
class Optimizer
{
public:
	float lr;
	/*
		Optimizer Class with 
		Momentum / NAG / AdaGrad / RMSprop / AdaDelta / Adam
	*/
public:
	Optimizer(const float& lr_in) : lr(lr_in) {}
	virtual void optimize(Variable& gradient, Variable& present_gradient) = 0;
};

class GradientDescent : public Optimizer
{
public:
	GradientDescent(const float& lr_in) : Optimizer(lr_in){}

	virtual void optimize(Variable& gradient, Variable& present_gradient) override {
		present_gradient = (-lr * present_gradient);
		gradient.move(present_gradient);
	}
};

class Momentum : public Optimizer
{
public:
	float momentum_alpha;

public:
	Momentum(const float& ma_in, const float& lr_in) : momentum_alpha(ma_in), Optimizer(lr_in) {}

	virtual void optimize(Variable& gradient, Variable& present_gradient) override {
		present_gradient = -(lr * present_gradient);
		if (gradient.data == nullptr) {
			gradient.move(present_gradient);
			return;
		}
		present_gradient = present_gradient + momentum_alpha * gradient;
		gradient.move(present_gradient);
	}

};

class NAG : public Optimizer
{
public:
	float momentum_alpha;

public:
	NAG(const float& ma_in, const float& lr_in) : momentum_alpha(ma_in), Optimizer(lr_in) {}

	virtual void optimize(Variable& gradient, Variable& present_gradient) override {

	}
};

class Adagrad : public Optimizer
{
public:

};

class RMSprop : public Optimizer
{
public:

};

class AdaDelta : public Optimizer
{
public:

};

class Adam : public Optimizer
{
public:

};