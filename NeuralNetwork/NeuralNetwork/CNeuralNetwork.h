#pragma once

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

protected:
	bool SetInputProperties(const UINT& inputDim, const UINT& num);
	bool CompareInputProperties(CMatrix* input);

public:
	CMatrix Predict(CMatrix* input);
	double Train(CMatrix* input);
};

