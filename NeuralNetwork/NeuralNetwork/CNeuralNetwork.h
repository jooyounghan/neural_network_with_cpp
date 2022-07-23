#pragma once
#include "CInitializer.h"

class CMatrix;
class CLayer;
class COptimizer;

class CNeuralNetwork
{
public:
	enum class ACTTYPE
	{
		RELU,
		SIGMOID,
		SOFTMAX
	};

	enum class LOSSTYPE
	{
		LEASTSQR,
		CROSSENT,
	};

protected:
	std::vector<CLayer*> layers;
	COptimizer* optimizer;
	
public:
	CNeuralNetwork();
	~CNeuralNetwork();

public:
	bool AddActivationLayer(ACTTYPE type);
	bool AddLossLayer(LOSSTYPE type);
	bool AddHiddenLayer(const UINT& dim);


public:
	bool SetInputProperties(const UINT& inputDim, const UINT& num);
	bool InitializeWeight(CInitializer::INITTYPE type);

protected:
	bool CompareInputProperties(std::shared_ptr<CMatrix> input);

public:
	CMatrix Predict(std::shared_ptr<CMatrix> input);
	double Train(std::shared_ptr<CMatrix> input, std::shared_ptr<CMatrix> label);
};

