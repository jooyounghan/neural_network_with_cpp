#pragma once
#include "NeuralNet.h"

class NNModel
{
private:
	CNeuralNetwork* NeuralNet;
	CMatrix* outputMatrix;
	CMatrix* loss;

public:
};

//#pragma region Propagation
//void CNeuralNetwork::PushInput(CMatrix* inputMatrix)
//{
//	ASSERT_CRASH(layers.size());
//	layers[0]->SetInput(inputMatrix);
//}
//
//void CNeuralNetwork::ForwardPropagation()
//{
//	ASSERT_CRASH(layers.size());
//	CLayer2D*& calcLayer = layers[0].latter;
//
//	while (calcLayer != nullptr)
//	{
//		calcLayer->GetInput()->GetMatMul(calcLayer->former->GetInput(), calcLayer->former->GetWeight());
//		calcLayer = calcLayer->latter;
//	}
//
//	CLayer2D*& lastLayer = layers[layers.size() - 1];
//	outputMatrix->GetMatMul(lastLayer->GetInput(), lastLayer->GetWeight());
//}
//
//void CNeuralNetwork::BackwardPropagation()
//{
//	ASSERT_CRASH(layers.size());
//
//}
//#pragma endregion


