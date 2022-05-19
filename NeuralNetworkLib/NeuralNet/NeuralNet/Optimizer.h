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
		MOMENTUM,
		ADAGRAD,
		ADAM
	};

protected:
	std::vector<std::shared_ptr<CMatrix>> lastGradient;

public:
	virtual void Update(std::shared_ptr<CNeuralNetwork>& neuralNet, const double& lr) = 0;
};

class CGD : public COptimizer
{
public:
	void Update(std::shared_ptr<CNeuralNetwork>& neuralNet, const double& lr) override;
};

class CMOMENTUM : public COptimizer
{
private:
	float momentum = 0.9;

public:
	void Update(std::shared_ptr<CNeuralNetwork>& neuralNet, const double& lr) override;
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