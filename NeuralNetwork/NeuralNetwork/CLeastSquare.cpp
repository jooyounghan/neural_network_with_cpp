#include "pch.h"
#include "CLeastSquare.h"

#include "CMatrix.h"

CLeastSquare::CLeastSquare()
{
}

CLeastSquare::~CLeastSquare()
{
}

void CLeastSquare::SetOutput()
{
	ASSERT_CRASH(input != nullptr)
	output = std::make_shared<CMatrix>(1, input->GetCol());
}

void CLeastSquare::ForwardProp()
{
	const UINT& rowNum = input->GetRow();
	const UINT& colNum = input->GetCol();
	CMatrix& inputData = *input;
	CMatrix& outputData = *output;
	CMatrix& labelData = *labelMatrix;

	ASSERT_CRASH(inputData.GetRow() == labelData.GetRow() && inputData.GetCol() == labelData.GetCol());
	ASSERT_CRASH(outputData.GetCol() == colNum);


	for (UINT cIdx = 0; cIdx < colNum; ++cIdx)
	{
		outputData[cIdx] = 0;
	}

	for (UINT rIdx = 0; rIdx < rowNum; ++rIdx)
	{
		for (UINT cIdx = 0; cIdx < colNum; ++cIdx)
		{
			outputData[cIdx] += 1.0 * std::pow((inputData[rIdx * colNum + cIdx] - labelData[rIdx * colNum + cIdx]), 2) / 2.0;
		}
	}
}

CMatrix CLeastSquare::BackwardProp(CLayer* layer)
{
	CMatrix& inputData = *input;
	CMatrix& labelData = *labelMatrix;

	ASSERT_CRASH(inputData.GetRow() == labelData.GetRow() && inputData.GetCol() == labelData.GetCol());

	return inputData - labelData;
}
