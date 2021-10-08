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
	Node past_gradient;

public:
	Optimizer() : past_gradient(Node()) {}
	virtual void optimize(Node& present_gradient) = 0;
	Node& getGradient();
};

class GradientDescent : public Optimizer {
public:
	GradientDescent() : Optimizer() {}
	virtual void optimize(Node& present_gradient) override;
};

class Momentum : public Optimizer {
public:
	float momentum_alpha;
public:
	Momentum(const float& alpha) : momentum_alpha(alpha) {}
	virtual void optimize(Node& present_gradient) override;
};

class Nag : public Optimizer {

};

class Adagrad : public Optimizer {

};

class Rmsprop : public Optimizer {

};

class Adam : public Optimizer {

};