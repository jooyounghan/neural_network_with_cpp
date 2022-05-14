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
	const uint32& refMatDataNum = refMat->GetDataNum();
	const uint32& inputMatDataNum = inputMat->GetDataNum();

	ASSERT_CRASH(refMatDataNum == inputMatDataNum);

	const uint32& matRow = refMat->GetRow();
	const uint32& matCol = refMat->GetCol();

	double*& refMatData = refMat->GetMatrixData();
	double*& inputMatData = inputMat->GetMatrixData();

	std::vector<double> totalProbabilities;
	totalProbabilities.resize(matRow);

	for (uint32 row = 0; row < matRow; ++row)
	{
		totalProbabilities[row] = 0.0;
		for (uint32 col = 0; col < matCol; ++col)
		{
			totalProbabilities[row] += std::exp(inputMatData[row * matCol + col]);
		}
	}

	for (uint32 row = 0; row < matRow; ++row)
	{
		for (uint32 col = 0; col < matCol; ++col)
		{
			const uint32 idx = row * matCol + col;
			refMatData[idx] = std::exp(inputMatData[idx]) / totalProbabilities[row];
		}
	}
}

void CSoftmax::GetLossGradient(CMatrix* refMat, CMatrix* inputMat, CMatrix* labelMat)
{
	PARALLELBRANCH(
		refMat->GetDataNum(),
		CSoftmax::LossGradientParallel(refMat, inputMat, labelMat),
		CSoftmax::LossGradientSerial(refMat, inputMat, labelMat)
	);
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