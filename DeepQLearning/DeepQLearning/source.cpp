#include <pch.h>

int main()
{
	while (true)
	{
		CTimer timer;
		timer.StartCheck();
		CNNModel model{ 10, 1, 1, 2, 32, 64};
		model.SetActivationFunc(3, CActFunc::ActID::RELU, CActFunc::ActID::RELU, CActFunc::ActID::RELU);
		model.InitializeWeights(3, CMatrix::Initializer::XAVIER, CMatrix::Initializer::XAVIER, CMatrix::Initializer::XAVIER);
		model.SetLossFunc(CLossFunc::LossID::SUMATION);
		model.SetOptimizer(COptimizer::OptimizerId::GD);
		for (uint32 iter = 0; iter < 100; ++iter)
		{
			//std::shared_ptr<CMatrix> a1 = std::make_shared<CMatrix>(1, 1, new double{ 1 });
			//std::shared_ptr<CMatrix> b1 = std::make_shared<CMatrix>(1, 1, new double{ 2 });
			//model.PushInput(a1, b1, 0.01);

			//std::shared_ptr<CMatrix> a2 = std::make_shared<CMatrix>(1, 1, new double{ 2.1 });
			//std::shared_ptr<CMatrix> b2 = std::make_shared<CMatrix>(1, 1, new double{ 4.2 });
			//model.PushInput(a2, b2, 0.01);

			//std::shared_ptr<CMatrix> a3 = std::make_shared<CMatrix>(1, 1, new double{ 3 });
			//std::shared_ptr<CMatrix> b3 = std::make_shared<CMatrix>(1, 1, new double{ 6 });
			//model.PushInput(a3, b3, 0.01);

			//std::shared_ptr<CMatrix> a4 = std::make_shared<CMatrix>(1, 1, new double{ 3.4 });
			//std::shared_ptr<CMatrix> b4 = std::make_shared<CMatrix>(1, 1, new double{ 6.8 });
			//model.PushInput(a4, b4, 0.01);

			//std::shared_ptr<CMatrix> a5 = std::make_shared<CMatrix>(1, 1, new double{ 1.7 });
			//std::shared_ptr<CMatrix> b5 = std::make_shared<CMatrix>(1, 1, new double{ 3.4 });
			//model.PushInput(a5, b5, 0.01);

			//std::shared_ptr<CMatrix> a6 = std::make_shared<CMatrix>(1, 1, new double{ 3.7 });
			//std::shared_ptr<CMatrix> b6 = std::make_shared<CMatrix>(1, 1, new double{ 7.4 });
			//model.PushInput(a6, b6, 0.01);

			//std::shared_ptr<CMatrix> a7 = std::make_shared<CMatrix>(1, 1, new double{ 2.9 });
			//std::shared_ptr<CMatrix> b7 = std::make_shared<CMatrix>(1, 1, new double{ 5.8 });
			//model.PushInput(a7, b7, 0.01);

			std::shared_ptr<CMatrix> a1 = std::make_shared<CMatrix>(10, 1, new double[]{
				1,
				2,
				3,
				4,
				5,
				6,
				7,
				8,
				9,
				10
				});
			std::shared_ptr<CMatrix> b1 = std::make_shared<CMatrix>(10, 1, new double[]{
				0.841470985,
				0.909297427,
				0.141120008,
				- 0.756802495,
				- 0.958924275,
				- 0.279415498,
				0.656986599,
				0.989358247,
				0.412118485,
				- 0.544021111
				});
			model.PushInput(a1, b1, 0.1);
			if (iter % 20 == 0)
				std::cout << "iterations : " << iter << " Loss : " << model.GetLoss() << std::endl;
		}
		timer.EndCheck();
		timer.PrintElapsedTime("Time Spent : ");
		std::shared_ptr<CMatrix> a8 = std::make_shared<CMatrix>(1, 1, new double{ 21 });
		std::shared_ptr<CMatrix> result = model.GetResult(a8);
		std::cout << "result of 3.5 : ";
		result.get()->PrintData();
	}
}


//int main()
//{
//	while (true)
//	{
//		CTimer timer;
//		timer.StartCheck();
//		CNNModel model{ 8, 2, 2, 4, 1024, 512, 512, 128 };
//		model.SetActivationFunc(5, 2, 2, 2, 2, 2);
//		model.InitializeWeights(5, 0, 0, 0, 0, 0, 0);
//		model.SetLossFunc(1);
//		model.SetOptimizer(0);
//
//		for (uint32 iter = 0; iter < 100; ++iter)
//		{
//			// Using Serial Data
//			//std::shared_ptr<CMatrix> a1 = std::make_shared<CMatrix>(1, 2, new double[]{ 1, 2 });
//			//std::shared_ptr<CMatrix> b1 = std::make_shared<CMatrix>(1, 2, new double[] { 0, 1 });
//			//model.PushInput(a1, b1, 0.01);
//
//			//std::shared_ptr<CMatrix> a2 = std::make_shared<CMatrix>(1, 2, new double[]{ 1, 0 });
//			//std::shared_ptr<CMatrix> b2 = std::make_shared<CMatrix>(1, 2, new double[] { 1, 0 });
//			//model.PushInput(a2, b2, 0.01);
//
//			//std::shared_ptr<CMatrix> a3 = std::make_shared<CMatrix>(1, 2, new double[]{ 2, 3 });
//			//std::shared_ptr<CMatrix> b3 = std::make_shared<CMatrix>(1, 2, new double[] { 0, 1 });
//			//model.PushInput(a3, b3, 0.01);
//
//			//std::shared_ptr<CMatrix> a4 = std::make_shared<CMatrix>(1, 2, new double[]{ 2, 1 });
//			//std::shared_ptr<CMatrix> b4 = std::make_shared<CMatrix>(1, 2, new double[] { 1, 0 });
//			//model.PushInput(a4, b4, 0.01);
//
//			//std::shared_ptr<CMatrix> a5 = std::make_shared<CMatrix>(1, 2, new double[]{ 3, 4 });
//			//std::shared_ptr<CMatrix> b5 = std::make_shared<CMatrix>(1, 2, new double[] { 0, 1 });
//			//model.PushInput(a5, b5, 0.01);
//
//			//std::shared_ptr<CMatrix> a6 = std::make_shared<CMatrix>(1, 2, new double[]{ 3, 2 });
//			//std::shared_ptr<CMatrix> b6 = std::make_shared<CMatrix>(1, 2, new double[] { 1, 0 });
//			//model.PushInput(a6, b6, 0.01); 
//
//			//std::shared_ptr<CMatrix> a7 = std::make_shared<CMatrix>(1, 2, new double[]{ 4, 5 });
//			//std::shared_ptr<CMatrix> b7 = std::make_shared<CMatrix>(1, 2, new double[] { 0, 1 });
//			//model.PushInput(a7, b7, 0.01);
//
//			//std::shared_ptr<CMatrix> a8 = std::make_shared<CMatrix>(1, 2, new double[] { 4, 3 });
//			//std::shared_ptr<CMatrix> b8 = std::make_shared<CMatrix>(1, 2, new double[] { 1, 0 });
//			//model.PushInput(a8, b8, 0.01);
//
//
//			// Using Batch Data
//			std::shared_ptr<CMatrix> a1 = std::make_shared<CMatrix>(8, 2, new double[]{ 1, 2, 1, 0, 2, 3, 2, 1, 3, 4, 3, 2, 4, 5, 4, 3});
//			std::shared_ptr<CMatrix> b1 = std::make_shared<CMatrix>(8, 2, new double[] { 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0 });
//			model.PushInput(a1, b1, 0.0001);
//
//			if (iter % 20 == 0)
//				std::cout << "iterations : " << iter << " Loss : " << model.GetLoss() << std::endl;
//		}
//		timer.EndCheck();
//		timer.PrintElapsedTime("Time Spent : ");
//
//		std::shared_ptr<CMatrix> a9 = std::make_shared<CMatrix>(1, 2, new double[]{ 8, 14 });
//		std::shared_ptr<CMatrix> result9 = model.GetResult(a9);
//		std::cout << "result of (5, 6) : ";
//		result9.get()->PrintData();
//		std::shared_ptr<CMatrix> a10 = std::make_shared<CMatrix>(1, 2, new double[]{ 8, 5 });
//		std::shared_ptr<CMatrix> result10 = model.GetResult(a10);
//		std::cout << "result of (5, 4) : ";
//		result10.get()->PrintData();
//	}
//}