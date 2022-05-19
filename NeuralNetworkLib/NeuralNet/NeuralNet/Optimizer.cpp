#include "pch.h"
#include "Optimizer.h"

#pragma region Gradient Descent
void CGD::Update(std::shared_ptr<CNeuralNetwork>& neuralNet, const double& lr)
{
	for (auto& layer : neuralNet->GetLayers())
	{
		std::shared_ptr<CMatrix>& weight = layer->GetWeight();
		std::shared_ptr<CMatrix>& weightGradient = layer->GetWeightGradient();
		weightGradient->ConstantMul(lr);
		weight->Subtract(weightGradient.get());
	}
}
#pragma endregion

#pragma region Gradient Descent Momentum
void CMOMENTUM::Update(std::shared_ptr<CNeuralNetwork>& neuralNet, const double& lr)
{
	uint32 idx = 0;
	for (auto& layer : neuralNet->GetLayers())
	{
		if (lastGradient.size() == neuralNet->GetLayers().size())
		{
			std::shared_ptr<CMatrix>& weight = layer->GetWeight();
			std::shared_ptr<CMatrix>& weightGradient = layer->GetWeightGradient();
			lastGradient[idx]->ConstantMul(momentum);
			weightGradient->Add(lastGradient[idx].get());

			lastGradient[idx]->CopyMatrix(layer->GetWeightGradient().get());

			weightGradient->ConstantMul(lr);
			weight->Subtract(weightGradient.get());
		}
		else
		{
			std::shared_ptr<CMatrix>& weight = layer->GetWeight();
			std::shared_ptr<CMatrix>& weightGradient = layer->GetWeightGradient();

			lastGradient.push_back(layer->GetWeightGradient()->GetSharedCopyMatrix());

			weightGradient->ConstantMul(lr);
			weight->Subtract(weightGradient.get());
		}
		idx++;
	}
}
#pragma endregion
