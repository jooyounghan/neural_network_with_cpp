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
	// TODO : INPUT / LABEL 과 관련된 별도의 클래스를 생성하여
	// shared_ptr를 내부적으로 숨겨야 함.
	CMatrix Predict(std::shared_ptr<CMatrix> input);
	double Train(std::shared_ptr<CMatrix> input, std::shared_ptr<CMatrix> label);
};

