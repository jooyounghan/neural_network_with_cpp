#include <pch.h>

int main()
{
	CNNModel model = CNNModel(1, 1, 3, 64, 128, 64);
	model.SetActivationFunc(4, AF_IDENTITY, AF_SIGMOID, AF_SIGMOID, AF_IDENTITY);
	model.SetLossFunc(LF_SOFTMAX);
	model.InitializeWeights(4, XAVIER, XAVIER, XAVIER, XAVIER);

	CMatrix a1 = CMatrix(1, 1, new double{ 1 });
	CMatrix b1 = CMatrix(1, 1, new double{ 2 });
	model.PushInput(a1, b1, 0.01);

	CMatrix a2 = CMatrix(1, 1, new double{ 2 });
	CMatrix b2 = CMatrix(1, 1, new double{ 4 });
	model.PushInput(a2, b2, 0.01);

	CMatrix a3 = CMatrix(1, 1, new double{ 3 });
	CMatrix b3 = CMatrix(1, 1, new double{ 6 });
	model.PushInput(a3, b3, 0.01);

	CMatrix a4 = CMatrix(1, 1, new double{ 4 });
	CMatrix b4 = CMatrix(1, 1, new double{ 8 });
	model.PushInput(a4, b4, 0.01);


	CMatrix a5 = CMatrix(1, 1, new double{ 5 });
	CMatrix* result = model.GetResult(a5);

}