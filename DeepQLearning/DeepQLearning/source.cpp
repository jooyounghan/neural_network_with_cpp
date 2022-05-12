//#include <pch.h>
//
//int main()
//{
//	while (true)
//	{
//		CNNModel model = CNNModel(1, 1, 1, 5, 32, 64, 128, 64, 32);
//		model.SetActivationFunc(6, AF_IDENTITY, AF_RELU, AF_RELU, AF_RELU, AF_RELU, AF_RELU);
//		model.InitializeWeights(6, HE, HE, HE, HE, HE, HE);
//		model.SetLossFunc(LF_SUMATION);
//
//		for (uint32 iter = 0; iter < 1000; ++iter)
//		{
//			std::shared_ptr<CMatrix> a1 = std::make_shared<CMatrix>(1, 1, new double{ 1 });
//			std::shared_ptr<CMatrix> b1 = std::make_shared<CMatrix>(1, 1, new double{ 2 });
//			model.PushInput(a1, b1, 0.01);
//
//			std::shared_ptr<CMatrix> a2 = std::make_shared<CMatrix>(1, 1, new double{ 2.1 });
//			std::shared_ptr<CMatrix> b2 = std::make_shared<CMatrix>(1, 1, new double{ 4.2 });
//			model.PushInput(a2, b2, 0.01);
//
//			std::shared_ptr<CMatrix> a3 = std::make_shared<CMatrix>(1, 1, new double{ 3 });
//			std::shared_ptr<CMatrix> b3 = std::make_shared<CMatrix>(1, 1, new double{ 6 });
//			model.PushInput(a3, b3, 0.01);
//
//			std::shared_ptr<CMatrix> a4 = std::make_shared<CMatrix>(1, 1, new double{ 3.4 });
//			std::shared_ptr<CMatrix> b4 = std::make_shared<CMatrix>(1, 1, new double{ 6.8 });
//			model.PushInput(a4, b4, 0.01);
//
//			std::shared_ptr<CMatrix> a5 = std::make_shared<CMatrix>(1, 1, new double{ 1.7 });
//			std::shared_ptr<CMatrix> b5 = std::make_shared<CMatrix>(1, 1, new double{ 3.4 });
//			model.PushInput(a5, b5, 0.01);
//
//			std::shared_ptr<CMatrix> a6 = std::make_shared<CMatrix>(1, 1, new double{ 3.7 });
//			std::shared_ptr<CMatrix> b6 = std::make_shared<CMatrix>(1, 1, new double{ 7.4 });
//			model.PushInput(a6, b6, 0.01);
//
//			std::shared_ptr<CMatrix> a7 = std::make_shared<CMatrix>(1, 1, new double{ 2.9 });
//			std::shared_ptr<CMatrix> b7 = std::make_shared<CMatrix>(1, 1, new double{ 5.8 });
//			model.PushInput(a7, b7, 0.01);
//
//
//			if (iter % 200 == 0)
//				std::cout << "iterations : " << iter << " Loss : " << model.GetLoss() << std::endl;
//		}
//
//		std::shared_ptr<CMatrix> a8 = std::make_shared<CMatrix>(1, 1, new double{ 3.5 });
//		std::shared_ptr<CMatrix> result = model.GetResult(a8);
//		result.get()->PrintData();
//	}
//}

#include <pch.h>

int main()
{
	while (true)
	{
		CNNModel model = CNNModel(1, 2, 2, 3, 128, 256, 128);
		model.SetActivationFunc(4, AF_RELU, AF_RELU, AF_RELU, AF_RELU);
		model.InitializeWeights(4, NORMAL, NORMAL, NORMAL, NORMAL);
		model.SetLossFunc(LF_SOFTMAX);

		for (uint32 iter = 0; iter < 100; ++iter)
		{
			std::shared_ptr<CMatrix> a1 = std::make_shared<CMatrix>(1, 2, new double[]{ 1, 2 });
			std::shared_ptr<CMatrix> b1 = std::make_shared<CMatrix>(1, 2, new double[] { 0, 1 });
			model.PushInput(a1, b1, 0.01);

			std::shared_ptr<CMatrix> a2 = std::make_shared<CMatrix>(1, 2, new double[]{ 1, 0 });
			std::shared_ptr<CMatrix> b2 = std::make_shared<CMatrix>(1, 2, new double[] { 1, 0 });
			model.PushInput(a2, b2, 0.01);

			std::shared_ptr<CMatrix> a3 = std::make_shared<CMatrix>(1, 2, new double[]{ 2, 3 });
			std::shared_ptr<CMatrix> b3 = std::make_shared<CMatrix>(1, 2, new double[] { 0, 1 });
			model.PushInput(a3, b3, 0.01);

			std::shared_ptr<CMatrix> a4 = std::make_shared<CMatrix>(1, 2, new double[]{ 2, 1 });
			std::shared_ptr<CMatrix> b4 = std::make_shared<CMatrix>(1, 2, new double[] { 1, 0 });
			model.PushInput(a4, b4, 0.01);

			std::shared_ptr<CMatrix> a5 = std::make_shared<CMatrix>(1, 2, new double[]{ 3, 4 });
			std::shared_ptr<CMatrix> b5 = std::make_shared<CMatrix>(1, 2, new double[] { 0, 1 });
			model.PushInput(a5, b5, 0.01);

			std::shared_ptr<CMatrix> a6 = std::make_shared<CMatrix>(1, 2, new double[]{ 3, 2 });
			std::shared_ptr<CMatrix> b6 = std::make_shared<CMatrix>(1, 2, new double[] { 1, 0 });
			model.PushInput(a6, b6, 0.01); 

			std::shared_ptr<CMatrix> a7 = std::make_shared<CMatrix>(1, 2, new double[]{ 4, 5 });
			std::shared_ptr<CMatrix> b7 = std::make_shared<CMatrix>(1, 2, new double[] { 0, 1 });
			model.PushInput(a7, b7, 0.01);

			std::shared_ptr<CMatrix> a8 = std::make_shared<CMatrix>(1, 2, new double[] { 4, 3 });
			std::shared_ptr<CMatrix> b8 = std::make_shared<CMatrix>(1, 2, new double[] { 1, 0 });
			model.PushInput(a8, b8, 0.01);

			if (iter % 20 == 0)
				std::cout << "iterations : " << iter << " Loss : " << model.GetLoss() << std::endl;
		}
		std::shared_ptr<CMatrix> a9 = std::make_shared<CMatrix>(1, 2, new double[]{ 8, 14 });
		std::shared_ptr<CMatrix> result9 = model.GetResult(a9);
		std::cout << "result of (5, 6) : ";
		result9.get()->PrintData();
		std::shared_ptr<CMatrix> a10 = std::make_shared<CMatrix>(1, 2, new double[]{ 8, 5 });
		std::shared_ptr<CMatrix> result10 = model.GetResult(a10);
		std::cout << "result of (5, 4) : ";
		result10.get()->PrintData();
	}
}