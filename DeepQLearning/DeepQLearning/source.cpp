#include <pch.h>
#include <CNeuralNetwork.h>
#include <CInitializer.h>
#include <iostream>
int main()
{
	while (true)
	{
		CNeuralNetwork network;
		network.AddHiddenLayer(16);
		network.AddActivationLayer(CNeuralNetwork::ACTTYPE::RELU);
		network.AddHiddenLayer(32);
		network.AddActivationLayer(CNeuralNetwork::ACTTYPE::RELU);
		network.AddHiddenLayer(16);
		network.AddActivationLayer(CNeuralNetwork::ACTTYPE::RELU);
		network.AddLossLayer(CNeuralNetwork::LOSSTYPE::LEASTSQR);
		network.SetInputProperties(2, 1);
		network.InitializeWeight(CInitializer::INITTYPE::HE);
		std::cout << "test" << std::endl;
	}
}