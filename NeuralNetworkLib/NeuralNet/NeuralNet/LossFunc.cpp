#include "pch.h"
#include "LossFunc.h"


#pragma region LossFunction
CLossFunc::CLossFunc(CMatrix* outputMatrix) : predictedMatrix(nullptr)
{
	ASSERT_CRASH(outputMatrix != nullptr);

	const uint32& outputMatRow = outputMatrix->GetRow();
	const uint32& outputMatCol = outputMatrix->GetCol();

	ASSERT_CRASH(outputMatCol == 1);

	DELETEPTR(predictedMatrix);
	predictedMatrix = new CMatrix(outputMatRow, outputMatCol);
}

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
void CSumation::GetResult(CMatrix* inputMat, CMatrix* labelMat)
{
#ifdef PARALLEL
	return ResultParallel(predictedMatrix, inputMat, labelMat);
#else
	return ResultSerial(predictedMatrix, inputMat, labelMat);
#endif
}

void CSumation::ResultParallel(CMatrix* refMat, CMatrix* inputMat, CMatrix* labelMat)
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
						refMatData[idx] = labelMatData[idx] - inputMatData[idx];
					}
				}
			}));
	}
	WAITTHREADVECTOR(workThreadVector);
}

void CSumation::ResultSerial(CMatrix* refMat, CMatrix* inputMat, CMatrix* labelMat)
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
		refMatData[idx] = labelMatData[idx] - inputMatData[idx];
	}
}
#pragma endregion

#pragma region Softmax
void CSoftmax::GetResult(CMatrix* inputMat, CMatrix* labelMat)
{
#ifdef PARALLEL
	return ResultParallel(predictedMatrix, inputMat, labelMat);
#else
	return ResultSerial(predictedMatrix, inputMat, labelMat);
#endif
}

void CSoftmax::ResultParallel(CMatrix* refMat, CMatrix* inputMat, CMatrix* labelMat)
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
						refMatData[idx] = labelMatData[idx] - std::exp(inputMatData[idx]) / totalProbability;
					}
				}
			}));
	}
	WAITTHREADVECTOR(workThreadVector);
}

void CSoftmax::ResultSerial(CMatrix* refMat, CMatrix* inputMat, CMatrix* labelMat)
{
	const uint32& refMatDataNum = refMat->GetDataNum();
	const uint32& inputMatDataNum = inputMat->GetDataNum();
	const uint32& labelMatDataNum = labelMat->GetDataNum();

	ASSERT_CRASH(refMatDataNum == inputMatDataNum);
	ASSERT_CRASH(inputMatDataNum == labelMatDataNum);

	double*& refMatData = refMat->GetMatrixData();
	double*& inputMatData = inputMat->GetMatrixData();
	double*& labelMatData = labelMat->GetMatrixData();

	double totalProbability = 0.0;
	
	for (uint32 idx = 0; idx < inputMatDataNum; ++idx)
	{
		totalProbability += std::exp(inputMatData[idx]);
	}

	for (uint32 idx = 0; idx < inputMatDataNum; ++idx)
	{
		refMatData[idx] = labelMatData[idx] - std::exp(inputMatData[idx]) / totalProbability;
	}
}
#pragma endregion