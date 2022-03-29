#pragma once
#include "Layer.h"
#include "Weight.h"
#include "RandomGenerator.h"

class NeuralNet : public std::vector<Layer2D>
{
public:
	NeuralNet();

public:
	void SetLayer(int32 layerCount, ...);
	void InitWithNormal();
	void SetInput(std::vector<Neuron> input);
	void ForwardProp();

private:
	std::vector<Weight> weights;
	std::shared_ptr<ActFunc> reluFunc;
	std::shared_ptr<ActFunc> sigmoidFunc;
};