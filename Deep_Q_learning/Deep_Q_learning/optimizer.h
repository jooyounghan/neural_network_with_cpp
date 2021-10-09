#pragma once
#include "node.h"

#define GRADIENT_DESCENT	1
#define MOMENTUM			2
#define NAG					3
#define ADAGRAD				4
#define RMSPROP				5
#define ADAM				6

class Optimizer {

public:
	float lr;
	Node past_gradient;

public:
	Optimizer(const float& lr_in) : lr(lr_in), past_gradient(Node()) {}
	virtual void optimize(Node& present_gradient) = 0;
	virtual Node& getGradient() = 0;
};

class GradientDescent : public Optimizer {
public:
	GradientDescent(const float& lr_in) : Optimizer(lr_in) {}
	virtual void optimize(Node& present_gradient) override;
	virtual Node& getGradient() override;
};

class Momentum : public Optimizer {
public:
	float momentum_alpha;
public:
	Momentum(const float& lr_in, const float& alpha) : Optimizer(lr_in), momentum_alpha(alpha) {}
	virtual void optimize(Node& present_gradient) override;
	virtual Node& getGradient() override;
};

class Nag : public Optimizer {
	float momentum_alpha;
	Node gradient;
public:
	Nag(const float& lr_in, const float& alpha) : Optimizer(lr_in), momentum_alpha(alpha), gradient(Node()) {}
	virtual void optimize(Node& present_gradient) override;
	virtual Node& getGradient() override;
};

class Adagrad : public Optimizer {

};

class Rmsprop : public Optimizer {

};

class Adam : public Optimizer {

};