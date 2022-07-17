#include "pch.h"
#include "CSigmoid.h"

#include "CMatrix.h"

CSigmoid::CSigmoid()
{
}

CSigmoid::~CSigmoid()
{
}

void CSigmoid::ForwardProp()
{
	const UINT& rowNum = input->GetRow();
	const UINT& colNum = input->GetCol();
	CMatrix& inputData = *input;
	CMatrix& outputData = *output;
	for (UINT rIdx = 0; rIdx < rowNum; ++rIdx)
	{
		for (UINT cIdx = 0; cIdx < colNum; ++cIdx)
		{
			outputData[rIdx * colNum + cIdx] = 1 / (1 + std::exp(-inputData[rIdx * colNum + cIdx]));
		}
	}
}

CMatrix CSigmoid::BackwardProp()
{

}

CMatrix CSigmoid::GetDeriviated()
{
	const UINT& rowNum = input->GetRow();
	const UINT& colNum = input->GetCol();
	CMatrix& inputData = *input;
	CMatrix result = CMatrix{ rowNum, colNum };
	for (UINT rIdx = 0; rIdx < rowNum; ++rIdx)
	{
		for (UINT cIdx = 0; cIdx < colNum; ++cIdx)
		{
			const float& sigmoid = 1 / (1 + std::exp(-inputData[rIdx * colNum + cIdx]));
			result[rIdx * colNum + cIdx] = sigmoid * (1 - sigmoid);
		}
	}
	return result;
}
