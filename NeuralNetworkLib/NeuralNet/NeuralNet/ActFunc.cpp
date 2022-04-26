#include "pch.h"
#include "ActFunc.h"

#pragma region ActivationFunc
CMatrix* CActFunc::ResultParallel(CMatrix* input, std::function<double(double&)> func)
{
	const uint32& inputDataNum = input->GetDataNum();
	double*& matrixData = input->GetMatrixData();
	std::vector<std::future<void>> workThreadVector;

	double* newMatrixData = new double[inputDataNum];
	uint32 workSize = std::ceil(static_cast<double>(inputDataNum) / THREADNUM);

	for (uint32 threadNum = 0; threadNum < THREADNUM; ++threadNum)
	{
		int32 startIdx = threadNum * workSize;
		workThreadVector.push_back(std::async([&, startIdx]()
			{
				for (uint32 idx = startIdx; idx < startIdx + workSize; ++idx)
				{
					if (idx >= inputDataNum)	break;
					else
					{
						newMatrixData[idx] = func(matrixData[idx]);
					}
				}
			}));
	}

	WAITTHREADVECTOR(workThreadVector);

	CMatrix* newMatrix = new CMatrix(input->GetRow(), input->GetCol(), newMatrixData);
	return newMatrix;
}

CMatrix* CActFunc::ResultSerial(CMatrix* input, std::function<double(double&)> func)
{
	const uint32& inputDataNum = input->GetDataNum();
	double*& matrixData = input->GetMatrixData();

	double* newMatrixData = new double[inputDataNum];
	for (uint32 idx = 0; idx < inputDataNum; ++idx)
	{
		newMatrixData[idx] = func(matrixData[idx]);
	}
	CMatrix* newMatrix = new CMatrix(input->GetRow(), input->GetCol(), newMatrixData);
	return newMatrix;
}

void CActFunc::ResultParallel(CMatrix* refMat, CMatrix* input, std::function<double(double&)> func)
{
	const uint32& inputDataNum = input->GetDataNum();
	double*& matrixData = input->GetMatrixData();
	std::vector<std::future<void>> workThreadVector;

	double*& newMatrixData = refMat->GetMatrixData();
	uint32 workSize = std::ceil(static_cast<double>(inputDataNum) / THREADNUM);

	for (uint32 threadNum = 0; threadNum < THREADNUM; ++threadNum)
	{
		int32 startIdx = threadNum * workSize;
		workThreadVector.push_back(std::async([&, startIdx]()
			{
				for (uint32 idx = startIdx; idx < startIdx + workSize; ++idx)
				{
					if (idx >= inputDataNum)	break;
					else
					{
						newMatrixData[idx] = func(matrixData[idx]);
					}
				}
			}));
	}

	WAITTHREADVECTOR(workThreadVector);
}

void CActFunc::ResultSerial(CMatrix* refMat, CMatrix* input, std::function<double(double&)> func)
{
	const uint32& inputDataNum = input->GetDataNum();
	double*& matrixData = input->GetMatrixData();

	double*& newMatrixData = refMat->GetMatrixData();
	for (uint32 idx = 0; idx < inputDataNum; ++idx)
	{
		newMatrixData[idx] = func(matrixData[idx]);
	}
}
#pragma endregion


#pragma region Sigmoid
CMatrix* Sigmoid::GetResult(CMatrix* input)
{
#ifdef PARALLEL
	return ResultParallel(input, [](double& variable) { return 1.0 / (1.0 + exp(-variable)); });
#else
	return ResultSerial(input, [](double& variable) { return 1.0 / (1.0 + exp(-variable)); });
#endif
}

void Sigmoid::CalcResult(CMatrix* refMat, CMatrix* input)
{
#ifdef PARALLEL
	return ResultParallel(refMat, input, [](double& variable) { return 1.0 / (1.0 + exp(-variable)); });
#else
	return ResultSerial(refMat, input, [](double& variable) { return 1.0 / (1.0 + exp(-variable)); });
#endif
}

CMatrix* Sigmoid::GetDeriviate(CMatrix* input)
{
#ifdef PARALLEL
	return ResultParallel(input, [](double& variable) { return (1.0 / (1.0 + exp(-variable))) * (1.0 - 1.0 / (1.0 + exp(-variable))); });
#else
	return ResultSerial(input, [](double& variable) { return (1.0 / (1.0 + exp(-variable))) * (1.0 - 1.0 / (1.0 + exp(-variable))); });
#endif
}
#pragma endregion


#pragma region Relu
CMatrix* Relu::GetResult(CMatrix* input)
{
#ifdef PARALLEL
	return ResultParallel(input, [](double& variable)
		{
			if (variable > 0)
			{
				return variable;
			}
			else
			{
				return 0.0;
			}
		});

#else
	return ResultSerial(input, [](double& variable)
		{
			if (variable > 0)
			{
				return 1.0;
			}
			else
			{
				return 0.0;
			}
		});

#endif
}

void Relu::CalcResult(CMatrix* refMat, CMatrix* input)
{
#ifdef PARALLEL
	return ResultParallel(refMat, input, [](double& variable)
		{
			if (variable > 0)
			{
				return variable;
			}
			else
			{
				return 0.0;
			}
		});

#else
	return ResultSerial(refMat, input, [](double& variable)
		{
			if (variable > 0)
			{
				return 1.0;
			}
			else
			{
				return 0.0;
			}
		});

#endif
}


CMatrix* Relu::GetDeriviate(CMatrix* input)
{
#ifdef PARALLEL
	return ResultParallel(input, [](double& variable)
		{
			if (variable > 0)
			{
				return 1.0;
			}
			else
			{
				return 0.0;
			}
		});

#else
	return ResultSerial(input, [](double& variable)
		{
			if (variable > 0)
			{
				return 1.0;
			}
			else
			{
				return 0.0;
			}
		});

#endif
}
#pragma endregion


#pragma region Identity
CMatrix* Identity::GetResult(CMatrix* input)
{
	return input->GetCopyMatrix();
}

void Identity::CalcResult(CMatrix* refMat, CMatrix* input)
{
	return refMat->CopyMatrix(input);
}

CMatrix* Identity::GetDeriviate(CMatrix* input)
{
#ifdef PARALLEL
	return ResultParallel(input, [](double& variable) { return (1.0); });
#else
	return ResultSerial(input, [](double& variable) { return (1.0); });
#endif
}
#pragma endregion


#pragma region Softmax
//CMatrix* Softmax::GetResult(CMatrix* input)
//{
//	return ResultSerial(input, [](double& variable) { return 1.0 / (1.0 + exp(-variable)); });
//}

//CMatrix* Softmax::GetDeriviate(CMatrix* input)
//{
//
//}
#pragma endregion

