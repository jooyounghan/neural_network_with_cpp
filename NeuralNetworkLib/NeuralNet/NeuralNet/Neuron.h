#pragma once
#include "NeuronWrapper.h"

class Neuron : public NeuronWrapper<Neuron>
{
public:
	Neuron();

public:
	std::vector<Neuron*>	in;
	std::vector<Neuron*>	out;
	int32					id;
};

