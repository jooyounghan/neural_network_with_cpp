#include <pch.h>
#include <CNeuralNetwork.h>

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
	}
}