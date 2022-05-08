#include <pch.h>

int main()
{
	CNNModel model = CNNModel(1, 1, 5, 64, 128, 256, 128, 64);
	model.SetActivationFunc(6, AF_IDENTITY, AF_SIGMOID, AF_SIGMOID, AF_SIGMOID, AF_SIGMOID, AF_IDENTITY);
	model.InitializeWeights(6, AF_RELU, AF_RELU, AF_RELU, AF_RELU, AF_RELU, AF_RELU);
	model.SetLossFunc(LF_SOFTMAX);

	for (uint32 iter = 0; iter < 1000; ++iter)
	{
		std::shared_ptr<CMatrix> a1 = std::make_shared<CMatrix>(1, 1, new double{ 1 });
		std::shared_ptr<CMatrix> b1 = std::make_shared<CMatrix>(1, 1, new double{ 2 });
		model.PushInput(a1, b1, 0.1);

		std::shared_ptr<CMatrix> a2 = std::make_shared<CMatrix>(1, 1, new double{ 2 });
		std::shared_ptr<CMatrix> b2 = std::make_shared<CMatrix>(1, 1, new double{ 4 });
		model.PushInput(a2, b2, 0.1);

		std::shared_ptr<CMatrix> a3 = std::make_shared<CMatrix>(1, 1, new double{ 3 });
		std::shared_ptr<CMatrix> b3 = std::make_shared<CMatrix>(1, 1, new double{ 6 });
		model.PushInput(a3, b3, 0.1);

		std::shared_ptr<CMatrix> a4 = std::make_shared<CMatrix>(1, 1, new double{ 4 });
		std::shared_ptr<CMatrix> b4 = std::make_shared<CMatrix>(1, 1, new double{ 8 });
		model.PushInput(a4, b4, 0.1);
	}

	std::shared_ptr<CMatrix> a5 = std::make_shared<CMatrix>(1, 1, new double{ 5 });
	std::shared_ptr<CMatrix> result = model.GetResult(a5);
	result.get()->PrintData();
}