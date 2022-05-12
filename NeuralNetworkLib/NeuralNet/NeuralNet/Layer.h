#pragma once
#include "ActFunc.h"

class CLayer2D
{
private:
	std::shared_ptr<CMatrix> inputMatrix;
	std::shared_ptr<CMatrix> activatedMatrix;
	std::shared_ptr<CMatrix> weight;
	std::shared_ptr<CMatrix> gradient;

private:
	std::shared_ptr<CMatrix> derivativeActFunc;
	std::shared_ptr<CMatrix> transposedWeight;
	std::shared_ptr<CMatrix> activatedTransposedInput;
	std::shared_ptr<CMatrix> weightGradient;

private:
	std::shared_ptr<CActFunc> activationFunc;

public:
	std::weak_ptr <CLayer2D> former;
	std::weak_ptr <CLayer2D> latter;

public:
	CLayer2D();
	~CLayer2D();

public:
	std::shared_ptr<CMatrix>& GetInput();
	std::shared_ptr<CMatrix>& GetActivatedInput();
	std::shared_ptr<CMatrix>& GetWeight();
	std::shared_ptr<CMatrix>& GetGradient();
	std::shared_ptr<CActFunc>& GetActivationFunc();

public:
	std::shared_ptr<CMatrix>& GetDerivativeActFunc();
	std::shared_ptr<CMatrix>& GetTransposedWeight();
	std::shared_ptr<CMatrix>& GetActivatedTransposedInput();
	std::shared_ptr<CMatrix>& GetWeightGradient();

public:
	void SetInput(std::shared_ptr<CMatrix> newInput);
	void SetActivavtedInput(std::shared_ptr<CMatrix> newInput);
	void SetWeight(std::shared_ptr<CMatrix> newWeight);
	void SetGradient(std::shared_ptr<CMatrix> newGradient);
	void SetActivateFunc(ACTID actId);

	void SetDerivativeActivatedInput(std::shared_ptr<CMatrix> newInput);
	void SetTransposedWeight(std::shared_ptr<CMatrix> newWeight);
	void SetActivatedTransposedInput(std::shared_ptr<CMatrix> newInput);
	void SetWeigthGradient(std::shared_ptr<CMatrix> newWeight);
};
