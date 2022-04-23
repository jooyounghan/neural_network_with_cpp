#pragma once
#include "Layer.h"
#include "Timer.h"

class NeuralNet
{
private:
	std::vector<CLayer2D*> layers;

public:
	NeuralNet();
	~NeuralNet();

public:
	void SetLayers(const uint32& dimension, const uint32 numLayers, ...);
	void SetInputNum(const uint32& number);

public:
	void SetActivationFunc(...);
	void InitializeWeight(...);

public:
	void PushInput(CMatrix* inputMatrix);
	
	void ForwardPropagation();
	void BackwardPropagation();
};