#include "pch.h"
#include "CRelu.h"

#include "CMatrix.h"

CRelu::CRelu()
{
}

CRelu::~CRelu()
{
}

void CRelu::ForwardProp()
{
	const UINT& rowNum = input->GetRow();
	const UINT& colNum = input->GetCol();
	CMatrix& inputData = *input;
	CMatrix& outputData = *output;
	for (UINT rIdx = 0; rIdx < rowNum; ++rIdx)
	{
		for (UINT cIdx = 0; cIdx < colNum; ++cIdx)
		{
			outputData[rIdx * colNum + cIdx] = inputData[rIdx * colNum + cIdx] > 0 ? inputData[rIdx * colNum + cIdx] : 0;
		}
	}
}

CMatrix CRelu::BackwardProp(CLayer* layer)
{
	CMatrix& postLayerGradient = *(layer->GetGradient());
	return postLayerGradient * GetDeriviated();
}

CMatrix CRelu::GetDeriviated()
{
	const UINT& rowNum = input->GetRow();
	const UINT& colNum = input->GetCol();
	CMatrix& inputData = *input;
	CMatrix result = CMatrix{ rowNum, colNum };
	for (UINT rIdx = 0; rIdx < rowNum; ++rIdx)
	{
		for (UINT cIdx = 0; cIdx < colNum; ++cIdx)
		{
			result[rIdx * colNum + cIdx] = inputData[rIdx * colNum + cIdx] > 0 ? 1 : 0;
		}
	}
	return result;
}
