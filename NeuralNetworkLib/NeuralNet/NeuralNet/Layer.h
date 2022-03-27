#pragma once
#include "Neuron.h"

class Layer : public std::vector<Neuron>, public NeuronWrapper<Layer>
{
public:
	void SetNodeNum(const int& num);
};

