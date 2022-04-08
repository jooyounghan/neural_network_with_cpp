#pragma once
#include "Neuron.h"

template<typename T>
class ActFunc
{
	virtual T* GetResult(const int& numOfElement, T* input) = 0;
	virtual T* GetDeriviate(const int& numOfElement, T* input) = 0;
};

template<typename T>
class Sigmoid : public ActFunc
{
	T* GetResult(T* input)
	{
		
	}
};

class Relu : public ActFunc
{
=
};

class Identity : public ActFunc
{

};

class Softmax : public ActFunc
{

};