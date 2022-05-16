#pragma once
#include "Matrix.h"

class CNeuralNetwork;
class CLayer2D;

using OPTIMIZERID = unsigned short;

class COptimizer
{
public:
	enum OptimizerId
	{
		GD,
		NAG,
		ADAGRAD,
		ADAM
	};

public:
	virtual void Update(std::shared_ptr<CNeuralNetwork>& neuralNet, const double& lr) = 0;
};

class CGD : public COptimizer
{
public:
	void Update(std::shared_ptr<CNeuralNetwork>& neuralNet, const double& lr) override
	{
		for (auto& layer : neuralNet->GetLayers())
		{
			std::shared_ptr<CMatrix>& weight = layer->GetWeight();
			std::shared_ptr<CMatrix>& weightGradient = layer->GetWeightGradient();
			weightGradient->ConstantMul(lr);
			weight->Subtract(weightGradient.get());
		}
	}
};

class CNAG : public COptimizer
{
public:
	void Update(std::shared_ptr<CNeuralNetwork>& neuralNet, const double& lr) override
	{

	}
};

class CAdagrad : public COptimizer
{
public:
	void Update(std::shared_ptr<CNeuralNetwork>& neuralNet, const double& lr) override
	{

	}
};

class CAdam : public COptimizer
{
public:
	void Update(std::shared_ptr<CNeuralNetwork>& neuralNet, const double& lr) override
	{

	}
};