#include "pch.h"
#include "Neuron.h"

Neuron::Neuron() : value(0.0)
{
	static int32 neuronId = 1;
	id = neuronId++;
}
