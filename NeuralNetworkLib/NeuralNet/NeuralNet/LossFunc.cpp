#include "pch.h"
#include "LossFunc.h"

#pragma region LossFunction
double CLossFunc::ResultParallel(CMatrix* input, CMatrix* label, std::function<double(double&, double&)> func)
{
	const uint32& inputMatrixCol = input->GetCol();
	ASSERT_CRASH(inputMatrixCol == 1);
	const uint32& inputDataNum = input->GetDataNum();
	double*& inputMatrixData = input->GetMatrixData();
	
	const uint32& labelMatrixCol = label->GetCol();
	ASSERT_CRASH(labelMatrixCol == 1);
	const uint32& labelDataNum = label->GetDataNum();
	double*& labelMatrixData = label->GetMatrixData();

	ASSERT_CRASH(inputDataNum == labelDataNum);

	std::vector<std::future<double>> workThreadVector;

	double* newMatrixData = new double[inputDataNum];
	uint32 workSize = std::ceil(static_cast<double>(inputDataNum) / THREADNUM);
	
	double lossResult = 0.0;

	for (uint32 threadNum = 0; threadNum < THREADNUM; ++threadNum)
	{
		int32 startIdx = threadNum * workSize;
		workThreadVector.push_back(std::async([&, startIdx]()
			{
				double asyncResult = 0.0;
				for (uint32 idx = startIdx; idx < startIdx + workSize; ++idx)
				{
					if (idx >= inputDataNum)	break;
					else
					{
						asyncResult += func(inputMatrixData[idx], labelMatrixData[idx]);
					}
				}
				return asyncResult;
			}));
	}

	for (auto& result : workThreadVector)
	{
		lossResult += result.get();
	}

	WAITTHREADVECTOR(workThreadVector);
	return lossResult;
}

double CLossFunc::ResultSerial(CMatrix* input, CMatrix* label, std::function<double(double&, double&)> func)
{
	const uint32& inputMatrixCol = input->GetCol();
	ASSERT_CRASH(inputMatrixCol == 1);
	const uint32& inputDataNum = input->GetDataNum();
	double*& inputMatrixData = input->GetMatrixData();

	const uint32& labelMatrixCol = label->GetCol();
	ASSERT_CRASH(labelMatrixCol == 1);
	const uint32& labelDataNum = label->GetDataNum();
	double*& labelMatrixData = label->GetMatrixData();

	ASSERT_CRASH(inputDataNum == labelDataNum);

	double lossResult = 0;

	for (uint32 idx = 0; idx < inputDataNum; ++idx)
	{
		lossResult += func(inputMatrixData[idx], labelMatrixData[idx]);
	}
	return lossResult;
}

void CLossFunc::ResultParallel(double& refDouble, CMatrix* input, CMatrix* label, std::function<double(double&, double&)> func)
{
	const uint32& inputMatrixCol = input->GetCol();
	ASSERT_CRASH(inputMatrixCol == 1);
	const uint32& inputDataNum = input->GetDataNum();
	double*& inputMatrixData = input->GetMatrixData();

	const uint32& labelMatrixCol = label->GetCol();
	ASSERT_CRASH(labelMatrixCol == 1);
	const uint32& labelDataNum = label->GetDataNum();
	double*& labelMatrixData = label->GetMatrixData();

	ASSERT_CRASH(inputDataNum == labelDataNum);

	std::vector<std::future<double>> workThreadVector;

	uint32 workSize = std::ceil(static_cast<double>(inputDataNum) / THREADNUM);

	for (uint32 threadNum = 0; threadNum < THREADNUM; ++threadNum)
	{
		int32 startIdx = threadNum * workSize;
		workThreadVector.push_back(std::async([&, startIdx]()
			{
				double asyncResult = 0.0;
				for (uint32 idx = startIdx; idx < startIdx + workSize; ++idx)
				{
					if (idx >= inputDataNum)	break;
					else
					{
						asyncResult += func(inputMatrixData[idx], labelMatrixData[idx]);
					}
				}
				return asyncResult;
			}));
	}
	
	refDouble = 0;

	for (auto& result : workThreadVector)
	{
		refDouble += result.get();
	}
}

void CLossFunc::ResultSerial(double& refDouble, CMatrix* input, CMatrix* label, std::function<double(double&, double&)> func)
{
	const uint32& inputMatrixCol = input->GetCol();
	ASSERT_CRASH(inputMatrixCol == 1);
	const uint32& inputDataNum = input->GetDataNum();
	double*& inputMatrixData = input->GetMatrixData();

	const uint32& labelMatrixCol = label->GetCol();
	ASSERT_CRASH(labelMatrixCol == 1);
	const uint32& labelDataNum = label->GetDataNum();
	double*& labelMatrixData = label->GetMatrixData();

	ASSERT_CRASH(inputDataNum == labelDataNum);

	refDouble = 0;

	for (uint32 idx = 0; idx < inputDataNum; ++idx)
	{
		refDouble += func(inputMatrixData[idx], labelMatrixData[idx]);
	}
}
#pragma endregion


//#pragma region Sumation
//CMatrix* CSumation::GetResult(CMatrix* input, CMatrix* label)
//{
//#ifdef PARALLEL
//	return ResultParallel(input, label, [](double& variable) { return 1.0 / (1.0 + exp(-variable)); });
//#else
//	return ResultSerial(input, [](double& variable) { return 1.0 / (1.0 + exp(-variable)); });
//#endif
//}
//
//void CSumation::CalcResult(CMatrix* refMat, CMatrix* input, CMatrix* label)
//{
//#ifdef PARALLEL
//	return ResultParallel(refMat, input, [](double& variable) { return 1.0 / (1.0 + exp(-variable)); });
//#else
//	return ResultSerial(refMat, input, [](double& variable) { return 1.0 / (1.0 + exp(-variable)); });
//#endif
//}
//
//CMatrix* CSumation::GetDeriviate(CMatrix* input, CMatrix* label)
//{
//#ifdef PARALLEL
//	return ResultParallel(input, [](double& variable) { return (1.0 / (1.0 + exp(-variable))) * (1.0 - 1.0 / (1.0 + exp(-variable))); });
//#else
//	return ResultSerial(input, [](double& variable) { return (1.0 / (1.0 + exp(-variable))) * (1.0 - 1.0 / (1.0 + exp(-variable))); });
//#endif
//}
//#pragma endregion
//
//
//#pragma region CRelu
//CMatrix* CSoftmax::GetResult(CMatrix* input)
//{
//#ifdef PARALLEL
//	return ResultParallel(input, [](double& variable)
//		{
//			if (variable > 0)
//			{
//				return variable;
//			}
//			else
//			{
//				return 0.0;
//			}
//		});
//
//#else
//	return ResultSerial(input, [](double& variable)
//		{
//			if (variable > 0)
//			{
//				return 1.0;
//			}
//			else
//			{
//				return 0.0;
//			}
//		});
//
//#endif
//}
//
//void CSoftmax::CalcResult(CMatrix* refMat, CMatrix* input)
//{
//#ifdef PARALLEL
//	return ResultParallel(refMat, input, [](double& variable)
//		{
//			if (variable > 0)
//			{
//				return variable;
//			}
//			else
//			{
//				return 0.0;
//			}
//		});
//
//#else
//	return ResultSerial(refMat, input, [](double& variable)
//		{
//			if (variable > 0)
//			{
//				return 1.0;
//			}
//			else
//			{
//				return 0.0;
//			}
//		});
//
//#endif
//}
//
//
//CMatrix* CSoftmax::GetDeriviate(CMatrix* input)
//{
//#ifdef PARALLEL
//	return ResultParallel(input, [](double& variable)
//		{
//			if (variable > 0)
//			{
//				return 1.0;
//			}
//			else
//			{
//				return 0.0;
//			}
//		});
//
//#else
//	return ResultSerial(input, [](double& variable)
//		{
//			if (variable > 0)
//			{
//				return 1.0;
//			}
//			else
//			{
//				return 0.0;
//			}
//		});
//
//#endif
//}
//#pragma endregion