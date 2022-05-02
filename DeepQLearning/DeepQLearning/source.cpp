#include <pch.h>

int main()
{
	CNNModel model = CNNModel(1, 2, 3, 64, 128, 64);
	model.SetActivationFunc(AF_IDENTITY, AF_SIGMOID, AF_SIGMOID, AF_IDENTITY);
	model.SetLossFunc(LF_SOFTMAX);
	model.InitializeWeights(XAVIER, XAVIER, XAVIER, XAVIER);

	bool check = true;
}