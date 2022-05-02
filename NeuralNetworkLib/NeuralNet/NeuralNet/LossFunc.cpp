#include "pch.h"
#include "LossFunc.h"


#pragma region LossFunction
CLossFunc::CLossFunc() {}

void CLossFunc::GetLoss(double& refDouble, CMatrix* inputMat)
{
	const uint32& inputDataNum = inputMat->GetDataNum();
	double*& inputMatrixData = inputMat->GetMatrixData();

	refDouble = 0;
	for (uint32 idx = 0; idx < inputDataNum; ++idx)
	{
		refDouble += std::pow(inputMatrixData[idx], 2) / 2.0;
	}
}
#pragma endregion

#pragma region Sumation
void CSumation::GetResult(CMatrix* refMat, CMatrix* inputMat)
{
	return;
}

void CSumation::GetLossGradient(CMatrix* refMat, CMatrix* inputMat, CMatrix* labelMat)
{
#ifdef PARALLEL
	return CSumation::LossGradientParallel(refMat, inputMat, labelMat);
#else
	return CSumation::LossGradientSerial(refMat, inputMat, labelMat);
#endif
}

void CSumation::LossGradientParallel(CMatrix* refMat, CMatrix* inputMat, CMatrix* labelMat)
{
	const uint32& refMatDataNum = refMat->GetDataNum();
	const uint32& inputMatDataNum = inputMat->GetDataNum();
	const uint32& labelMatDataNum = labelMat->GetDataNum();

	ASSERT_CRASH(refMatDataNum == inputMatDataNum);
	ASSERT_CRASH(inputMatDataNum == labelMatDataNum);

	double*& refMatData = refMat->GetMatrixData();
	double*& inputMatData = inputMat->GetMatrixData();
	double*& labelMatData = labelMat->GetMatrixData();

	std::vector<std::future<void>> workThreadVector;
	uint32 workSize = std::ceil(static_cast<double>(inputMatDataNum) / THREADNUM);

	for (uint32 threadNum = 0; threadNum < THREADNUM; ++threadNum)
	{
		int32 startIdx = threadNum * workSize;
		workThreadVector.push_back(std::async([&, startIdx]()
			{
				for (uint32 idx = startIdx; idx < startIdx + workSize; ++idx)
				{
					if (idx >= inputMatDataNum)	break;
					else
					{
						refMatData[idx] = 0;
						refMatData[idx] = labelMatData[idx] - inputMatData[idx];
					}
				}
			}));
	}
	WAITTHREADVECTOR(workThreadVector);
}

void CSumation::LossGradientSerial(CMatrix* refMat, CMatrix* inputMat, CMatrix* labelMat)
{
	const uint32& refMatDataNum = refMat->GetDataNum();
	const uint32& inputMatDataNum = inputMat->GetDataNum();
	const uint32& labelMatDataNum = labelMat->GetDataNum();

	ASSERT_CRASH(refMatDataNum == inputMatDataNum);
	ASSERT_CRASH(inputMatDataNum == labelMatDataNum);

	double*& refMatData = refMat->GetMatrixData();
	double*& inputMatData = inputMat->GetMatrixData();
	double*& labelMatData = labelMat->GetMatrixData();

	for (uint32 idx = 0; idx < inputMatDataNum; ++idx)
	{
		refMatData[idx] = 0;
		refMatData[idx] = labelMatData[idx] - inputMatData[idx];
	}
}
#pragma endregion

#pragma region Softmax
void CSoftmax::GetResult(CMatrix* refMat, CMatrix* inputMat)
{
#ifdef PARALLEL
	return CSoftmax::ResultParallel(refMat, inputMat);
#else
	return CSoftmax::ResultSerial(refMat, inputMat);
#endif
}

void CSoftmax::GetLossGradient(CMatrix* refMat, CMatrix* inputMat, CMatrix* labelMat)
{
#ifdef PARALLEL
	return CSoftmax::LossGradientParallel(refMat, inputMat, labelMat);
#else
	return CSoftmax::LossGradientSerial(refMat, inputMat, labelMat);
#endif
}

void CSoftmax::ResultParallel(CMatrix* refMat, CMatrix* inputMat)
{
	const uint32& refMatDataNum = refMat->GetDataNum();
	const uint32& inputMatDataNum = inputMat->GetDataNum();

	ASSERT_CRASH(refMatDataNum == inputMatDataNum);

	double*& refMatData = refMat->GetMatrixData();
	double*& inputMatData = inputMat->GetMatrixData();

	std::vector<std::future<void>> workThreadVector;
	std::vector<std::future<double>> totalProThreadVector;

	double totalProbability = 0.0;

	uint32 workSize = std::ceil(static_cast<double>(inputMatDataNum) / THREADNUM);

	for (uint32 threadNum = 0; threadNum < THREADNUM; ++threadNum)
	{
		int32 startIdx = threadNum * workSize;
		totalProThreadVector.push_back(std::async([&, startIdx]()
			{
				double asyncResult = 0.0;
				for (uint32 idx = startIdx; idx < startIdx + workSize; ++idx)
				{
					if (idx >= inputMatDataNum)	break;
					else
					{
						asyncResult += std::exp(inputMatData[idx]);
					}
				}
				return asyncResult;
			}));
	}
	
	for (uint32 threadNum = 0; threadNum < THREADNUM; ++threadNum)
	{
		totalProbability += totalProThreadVector[threadNum].get();
	}

	for (uint32 threadNum = 0; threadNum < THREADNUM; ++threadNum)
	{
		int32 startIdx = threadNum * workSize;
		workThreadVector.push_back(std::async([&, startIdx]()
			{
				for (uint32 idx = startIdx; idx < startIdx + workSize; ++idx)
				{
					if (idx >= inputMatDataNum)	break;
					else
					{
						refMatData[idx] = 0;
						refMatData[idx] = std::exp(inputMatData[idx]) / totalProbability;
					}
				}
			}));
	}
	WAITTHREADVECTOR(workThreadVector);
}

void CSoftmax::ResultSerial(CMatrix* refMat, CMatrix* inputMat)
{
	const uint32& refMatDataNum = refMat->GetDataNum();
	const uint32& inputMatDataNum = inputMat->GetDataNum();

	ASSERT_CRASH(refMatDataNum == inputMatDataNum);

	double*& refMatData = refMat->GetMatrixData();
	double*& inputMatData = inputMat->GetMatrixData();

	double totalProbability = 0.0;
	
	for (uint32 idx = 0; idx < inputMatDataNum; ++idx)
	{
		totalProbability += std::exp(inputMatData[idx]);
	}

	for (uint32 idx = 0; idx < inputMatDataNum; ++idx)
	{
		refMatData[idx] = std::exp(inputMatData[idx]) / totalProbability;
	}
}

void CSoftmax::LossGradientParallel(CMatrix* refMat, CMatrix* inputMat, CMatrix* labelMat)
{
	const uint32& refMatDataNum = refMat->GetDataNum();
	const uint32& inputMatDataNum = inputMat->GetDataNum();
	const uint32& labelMatDataNum = labelMat->GetDataNum();

	ASSERT_CRASH(refMatDataNum == inputMatDataNum);
	ASSERT_CRASH(inputMatDataNum == labelMatDataNum);

	double*& refMatData = refMat->GetMatrixData();
	double*& inputMatData = inputMat->GetMatrixData();
	double*& labelMatData = labelMat->GetMatrixData();

	std::vector<std::future<void>> workThreadVector;

	uint32 workSize = std::ceil(static_cast<double>(inputMatDataNum) / THREADNUM);

	for (uint32 threadNum = 0; threadNum < THREADNUM; ++threadNum)
	{
		int32 startIdx = threadNum * workSize;
		workThreadVector.push_back(std::async([&, startIdx]()
			{
				for (uint32 idx = startIdx; idx < startIdx + workSize; ++idx)
				{
					if (idx >= inputMatDataNum)	break;
					else
					{
						refMatData[idx] = 0;
						refMatData[idx] = labelMatData[idx] - inputMatData[idx];
					}
				}
			}));
	}
	WAITTHREADVECTOR(workThreadVector);
}

void CSoftmax::LossGradientSerial(CMatrix* refMat, CMatrix* inputMat, CMatrix* labelMat)
{
	const uint32& refMatDataNum = refMat->GetDataNum();
	const uint32& inputMatDataNum = inputMat->GetDataNum();
	const uint32& labelMatDataNum = labelMat->GetDataNum();

	ASSERT_CRASH(refMatDataNum == inputMatDataNum);
	ASSERT_CRASH(inputMatDataNum == labelMatDataNum);

	double*& refMatData = refMat->GetMatrixData();
	double*& inputMatData = inputMat->GetMatrixData();
	double*& labelMatData = labelMat->GetMatrixData();

	for (uint32 idx = 0; idx < inputMatDataNum; ++idx)
	{
		refMatData[idx] = 0;
		refMatData[idx] = labelMatData[idx] - inputMatData[idx];
	}
}
#pragma endregion