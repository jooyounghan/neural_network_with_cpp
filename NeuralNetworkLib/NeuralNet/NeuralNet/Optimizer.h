#pragma once
#include "Matrix.h"

class CNeuralNetwork;
class CLayer2D;

class COptimizer
{
public:
	virtual void Update(std::shared_ptr<CNeuralNetwork>, const double& lr) = 0;
};

class CGD : public COptimizer
{
public:
	void Update(std::shared_ptr<CNeuralNetwork>, const double& lr) override
	{
		
	}
};

class CNAG : public COptimizer
{
public:
	void Update(std::shared_ptr<CNeuralNetwork>, const double& lr) override
	{

	}
};

class CAdagrad : public COptimizer
{
public:
	void Update(std::shared_ptr<CNeuralNetwork>, const double& lr) override
	{

	}
};

class CAdam : public COptimizer
{
public:
	void Update(std::shared_ptr<CNeuralNetwork>, const double& lr) override
	{

	}
};