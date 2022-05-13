#include "pch.h"
#include "LossFunc.h"


#pragma region LossFunction
CLossFunc::CLossFunc() {}

void CLossFunc::GetLoss(double& refDouble, CMatrix* resultMat, CMatrix* labelMat)
{
	const uint32& resultDataNum = resultMat->GetDataNum();
	const uint32& labelDataNum = labelMat->GetDataNum();
	ASSERT_CRASH(resultDataNum == labelDataNum);

	double*& resultMatrixData = resultMat->GetMatrixData();
	double*& labelMatrixData = labelMat->GetMatrixData();
	refDouble = 0;
	for (uint32 idx = 0; idx < resultDataNum; ++idx)
	{
		refDouble += std::pow(resultMatrixData[idx] - labelMatrixData[idx], 2) / 2.0;
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
	PARALLELBRANCH(
		refMat->GetDataNum(),
		CSumation::LossGradientParallel(refMat, inputMat, labelMat),
		CSumation::LossGradientSerial(refMat, inputMat, labelMat)
	);
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
						refMatData[idx] = inputMatData[idx] - labelMatData[idx];
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
		refMatData[idx] = inputMatData[idx] - labelMatData[idx];
	}
}
#pragma endregion

#pragma region Softmax
void CSoftmax::GetResult(CMatrix* refMat, CMatrix* inputMat)
{
	PARALLELBRANCH(
		refMat->GetDataNum(),
		CSoftmax::ResultParallel(refMat, inputMat),
		CSoftmax::ResultSerial(refMat, inputMat)
	);
}

void CSoftmax::GetLossGradient(CMatrix* refMat, CMatrix* inputMat, CMatrix* labelMat)
{
	PARALLELBRANCH(
		refMat->GetDataNum(),
		CSoftmax::LossGradientParallel(refMat, inputMat, labelMat),
		CSoftmax::LossGradientSerial(refMat, inputMat, labelMat)
	);
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
						refMatData[idx] = inputMatData[idx] - labelMatData[idx];
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
		refMatData[idx] = inputMatData[idx] - labelMatData[idx];
	}
}
#pragma endregion