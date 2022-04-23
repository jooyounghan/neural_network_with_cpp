#include "pch.h"
#include "Layer.h"

#pragma region Constructor
CLayer2D::CLayer2D() :
	inputMatrix(nullptr), weight(nullptr), gradient(nullptr), activationFunc(nullptr),
	former(nullptr), latter(nullptr)
{

}
#pragma endregion


#pragma region Destructor
CLayer2D::~CLayer2D()
{
	DELETEPTR(inputMatrix);
	DELETEPTR(weight);
	DELETEPTR(gradient);
	DELETEPTR(activationFunc);
}
#pragma endregion

#pragma region Getter
CMatrix* CLayer2D::GetInput()
{
	return inputMatrix;
}
CMatrix* CLayer2D::GetWeight()
{
	return weight;
}
CMatrix* CLayer2D::GetGradient()
{
	return gradient;
}
#pragma endregion


#pragma region Setter
void CLayer2D::SetInput(CMatrix* newInput)
{
	DELETEPTR(inputMatrix);
	inputMatrix = newInput;
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
#pragma endregion
