#pragma once
#include "IncludePart.h"

class Neuron;

template<typename Self>
class NeuronWrapper
{
public:
	template<typename T>
	void ConnectTo(T& other)
	{
		for (Neuron& from : *static_cast<Self*>(this))
		{
			for (Neuron& to : other)
			{
				from.out.push_back(&to);
				to.in.push_back(&from);
			}
		}
	}
};


