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
	virtual void naiveOptimize(Node& present_gradient) = 0;
	virtual Node& getGradient() = 0;
};

class GradientDescent : public Optimizer {
public:
	GradientDescent(const float& lr_in) : Optimizer(lr_in) {}
	virtual void optimize(Node& present_gradient) override;
	virtual void naiveOptimize(Node& present_gradient) override;
	virtual Node& getGradient() override;
};

class Momentum : public Optimizer {
public:
	float momentum_alpha;
public:
	Momentum(const float& lr_in, const float& alpha) : Optimizer(lr_in), momentum_alpha(alpha) {}

	virtual void optimize(Node& present_gradient) override;
	virtual void naiveOptimize(Node& present_gradient) override;
	virtual Node& getGradient() override;
};

class Nag : public Optimizer {
public:
	float momentum_alpha;
	Node gradient;
public:
	Nag(const float& lr_in, const float& alpha) : Optimizer(lr_in), momentum_alpha(alpha), gradient(Node()) {}
	virtual void optimize(Node& present_gradient) override;
	virtual void naiveOptimize(Node& present_gradient) override;
	virtual Node& getGradient() override;
};

class Adagrad : public Optimizer {
public:
	float delta;
	Node g_node;
public:
	Adagrad(const float& lr_in, const float& delta_in) : Optimizer(lr_in), g_node(Node()), delta(delta_in) {}
	virtual void optimize(Node& present_gradient) override;
	virtual void naiveOptimize(Node& present_gradient) override;
	virtual Node& getGradient() override;
};

class Rmsprop : public Optimizer {
public:
	float delta;
	float gamma;
	Node h_node;
public:
	Rmsprop(const float& lr_in, const float& delta_in, const float& gamma_in) : Optimizer(lr_in), h_node(Node()), delta(delta_in), gamma(gamma_in) {}
	virtual void optimize(Node& present_gradient) override;
	virtual void naiveOptimize(Node& present_gradient) override;
	virtual Node& getGradient() override;
};

class Adam : public Optimizer {
public:
	unsigned int t;
	float delta;
	float rho1;
	float rho2;
	Node s_node;
	Node r_node;
public:
	Adam(const float& lr_in, const float& delta_in, const float& rho1_in, const float& rho2_in)
		: Optimizer(lr_in), s_node(Node()), r_node(Node()), delta(delta_in), rho1(rho1_in), rho2(rho2_in), t(1) {}
	virtual void optimize(Node& present_gradient) override;
	virtual void naiveOptimize(Node& present_gradient) override;
	virtual Node& getGradient() override;
};