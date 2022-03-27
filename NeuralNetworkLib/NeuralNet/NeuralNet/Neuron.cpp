#include "pch.h"
#include "Neuron.h"

Neuron::Neuron()
{
	static int32 neuronId = 1;
	id = neuronId++;
}
