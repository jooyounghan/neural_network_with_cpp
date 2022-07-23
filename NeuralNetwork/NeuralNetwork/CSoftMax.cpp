#include "pch.h"
#include "CSoftMax.h"

#include "CMatrix.h"

CSoftMax::CSoftMax()
{
}

CSoftMax::~CSoftMax()
{

}

void CSoftMax::ForwardProp()
{
	const UINT& rowNum = input->GetRow();
	const UINT& colNum = input->GetCol();
	CMatrix& inputData = *input;
	CMatrix& outputData = *output;

	std::vector<double> denominators;
	denominators.resize(colNum, 0.0);

	for (UINT rIdx = 0; rIdx < rowNum; ++rIdx)
	{
		for (UINT cIdx = 0; cIdx < colNum; ++cIdx)
		{
			denominators[cIdx] += std::exp(inputData[rIdx * colNum + cIdx]);
		}
	}

	for (UINT rIdx = 0; rIdx < rowNum; ++rIdx)
	{
		for (UINT cIdx = 0; cIdx < colNum; ++cIdx)
		{
			outputData[rIdx * colNum + cIdx] = inputData[rIdx * colNum + cIdx] / denominators[cIdx];
		}
	}
}

CMatrix CSoftMax::BackwardProp(CLayer* layer)
{
	// need to Proof
	return CMatrix{ 0, 0 };
}

CMatrix CSoftMax::GetDeriviated()
{
	// need to Proof
	return CMatrix{ 0, 0 };
}
