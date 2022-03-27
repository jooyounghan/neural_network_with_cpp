#pragma once
#include "Layer.h"
#include "Weight.h"

class NeuralNet : public std::vector<Layer2D>
{
public:
	NeuralNet();

public:
	void SetLayer(int32 layerCount, ...);

public:
	std::vector<Weight> weights;

};