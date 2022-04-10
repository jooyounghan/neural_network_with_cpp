#include "pch.h"
#include "Layer.h"

CLayer2D::CLayer2D() : inputMatrix(nullptr), weight(nullptr), gradient(nullptr), activationFunc(nullptr)
{

}

CLayer2D::~CLayer2D()
{
	DELETEPTR(inputMatrix);
	DELETEPTR(weight);
	DELETEPTR(gradient);
	DELETEPTR(activationFunc);
}
