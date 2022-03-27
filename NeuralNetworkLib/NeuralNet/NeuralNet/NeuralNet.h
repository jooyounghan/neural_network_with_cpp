#pragma once
#include "Layer.h"
#include <stdarg.h>

class NeuralNet : public std::vector<Layer>
{
public:
	NeuralNet();

public:
	void SetLayer(int32 layerCount, ...);
	Layer GetLayer(int32 index);
};