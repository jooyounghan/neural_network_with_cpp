#include "pch.h"
#include "Layer.h"

#pragma region Constructor
CLayer2D::CLayer2D() :
	inputMatrix(nullptr), activatedMatrix(nullptr), weight(nullptr), gradient(nullptr),
	derivativeActFunc(nullptr), transposedWeight(nullptr), activatedTransposedInput(nullptr), weightGradient(nullptr),
	activationFunc(nullptr), former(nullptr), latter(nullptr)
{

}
#pragma endregion


#pragma region Destructor
CLayer2D::~CLayer2D()
{
	DELETEPTR(inputMatrix);
	DELETEPTR(activatedMatrix);
	DELETEPTR(weight);
	DELETEPTR(gradient);

	DELETEPTR(derivativeActFunc);
	DELETEPTR(transposedWeight);
	DELETEPTR(activatedTransposedInput);
	DELETEPTR(weightGradient);

	DELETEPTR(activationFunc);
}
#pragma endregion

#pragma region Getter
CMatrix*& CLayer2D::GetInput()
{
	return inputMatrix;
}
CMatrix*& CLayer2D::GetActivatedInput()
{
	return activatedMatrix;
}
CMatrix*& CLayer2D::GetWeight()
{
	return weight;
}
CMatrix*& CLayer2D::GetGradient()
{
	return gradient;
}

CActFunc*& CLayer2D::GetActivationFunc()
{
	return activationFunc;
}

CMatrix*& CLayer2D::GetDerivativeActFunc()
{
	return derivativeActFunc;
}
CMatrix*& CLayer2D::GetTransposedWeight()
{
	return transposedWeight;
}
CMatrix*& CLayer2D::GetActivatedTransposedInput()
{
	return activatedTransposedInput;
}
CMatrix*& CLayer2D::GetWeightGradient()
{
	return weightGradient;
}
#pragma endregion


#pragma region Setter
void CLayer2D::SetInput(CMatrix* newInput)
{
	DELETEPTR(inputMatrix);
	inputMatrix = newInput;
}

void CLayer2D::SetActivavtedInput(CMatrix* newInput)
{
	DELETEPTR(activatedMatrix);
	activatedMatrix = newInput;
}

void CLayer2D::SetWeight(CMatrix* newWeight)
{
	DELETEPTR(weight);
	weight = newWeight;

}
void CLayer2D::SetGradient(CMatrix* newGradient)
{
	DELETEPTR(gradient);
	gradient = newGradient;

}
void CLayer2D::SetActivateFunc(uint16 ACTID)
{
	DELETEPTR(activationFunc);
	switch (ACTID)
	{
	case SIGMOID:
		activationFunc = new Sigmoid();
		break;
	case RELU:
		activationFunc = new Relu();
		break;
	case IDENTITY:
		activationFunc = new Identity();
		break;
	default:
		break;
	}
	ASSERT_CRASH(activationFunc != nullptr);
}

void CLayer2D::SetDerivativeActivatedInput(CMatrix* newInput)
{
	DELETEPTR(derivativeActFunc);
	derivativeActFunc = newInput;
}
void CLayer2D::SetTransposedWeight(CMatrix* newWeight)
{
	DELETEPTR(transposedWeight);
	transposedWeight = newWeight;

}
void CLayer2D::SetActivatedTransposedInput(CMatrix* newInput)
{
	DELETEPTR(activatedTransposedInput);
	activatedTransposedInput = newInput;
}

void CLayer2D::SetWeigthGradient(CMatrix* newWeight)
{
	DELETEPTR(weightGradient);
	weightGradient = newWeight;
}
#pragma endregion
