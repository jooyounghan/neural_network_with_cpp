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
		present_gradient = -(lr * present_gradient);
		gradient.move(present_gradient);
	}
};

class Momentum : public Optimizer
	/*
	Momentum method uses the past gradient which is calculated at the past iteration
	which can make model escape from the local minimum poit
	*/
{
public:
	float momentum_alpha;

public:
	Momentum(const float& ma_in, const float& lr_in) : momentum_alpha(ma_in), Optimizer(lr_in) {}

	virtual void optimize(Variable& gradient, Variable& present_gradient) override {
		present_gradient = -(lr * present_gradient);
		if (gradient.row != 0 && gradient.col != 0) {
			gradient = momentum_alpha * gradient;
			present_gradient = present_gradient + gradient;
		}
		//gradient.reset();
		//gradient.move(present_gradient);
		gradient = -(lr * present_gradient);
	}

};

class NAG : public Optimizer
{
public:

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