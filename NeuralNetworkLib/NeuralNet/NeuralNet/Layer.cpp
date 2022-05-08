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
}
#pragma endregion

#pragma region Getter
std::shared_ptr<CMatrix>& CLayer2D::GetInput()
{
	return inputMatrix;
}
std::shared_ptr<CMatrix>& CLayer2D::GetActivatedInput()
{
	return activatedMatrix;
}
std::shared_ptr<CMatrix>& CLayer2D::GetWeight()
{
	return weight;
}
std::shared_ptr<CMatrix>& CLayer2D::GetGradient()
{
	return gradient;
}

std::shared_ptr<CActFunc>& CLayer2D::GetActivationFunc()
{
	return activationFunc;
}

std::shared_ptr<CMatrix>& CLayer2D::GetDerivativeActFunc()
{
	return derivativeActFunc;
}
std::shared_ptr<CMatrix>& CLayer2D::GetTransposedWeight()
{
	return transposedWeight;
}
std::shared_ptr<CMatrix>& CLayer2D::GetActivatedTransposedInput()
{
	return activatedTransposedInput;
}
std::shared_ptr<CMatrix>& CLayer2D::GetWeightGradient()
{
	return weightGradient;
}
#pragma endregion


#pragma region Setter
void CLayer2D::SetInput(std::shared_ptr<CMatrix> newInput)
{
	inputMatrix = newInput;
}

void CLayer2D::SetActivavtedInput(std::shared_ptr<CMatrix> newInput)
{
	activatedMatrix = newInput;
}

void CLayer2D::SetWeight(std::shared_ptr<CMatrix> newWeight)
{
	weight = newWeight;

}
void CLayer2D::SetGradient(std::shared_ptr<CMatrix> newGradient)
{
	gradient = newGradient;

}
void CLayer2D::SetActivateFunc(ACTID actId)
{
	switch (actId)
	{
	case AF_SIGMOID:
		activationFunc = std::make_shared<CSigmoid>();
		break;
	case AF_RELU:
		activationFunc = std::make_shared <CRelu>();
		break;
	case AF_IDENTITY:
		activationFunc = std::make_shared <CIdentity>();
		break;
	default:
		break;
	}
	ASSERT_CRASH(activationFunc != nullptr);
}

void CLayer2D::SetDerivativeActivatedInput(std::shared_ptr<CMatrix> newInput)
{
	derivativeActFunc = newInput;
}
void CLayer2D::SetTransposedWeight(std::shared_ptr<CMatrix> newWeight)
{
	transposedWeight = newWeight;

}
void CLayer2D::SetActivatedTransposedInput(std::shared_ptr<CMatrix> newInput)
{
	activatedTransposedInput = newInput;
}

void CLayer2D::SetWeigthGradient(std::shared_ptr<CMatrix> newWeight)
{
	weightGradient = newWeight;
}
#pragma endregion
