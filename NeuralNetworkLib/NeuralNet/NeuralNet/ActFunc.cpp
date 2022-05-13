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


#pragma region CSigmoid
CMatrix* CSigmoid::GetResult(CMatrix* input)
{
	PARALLELBRANCH(
		input->GetDataNum(),
		ResultParallel(input, [](double& variable) { return 1.0 / (1.0 + exp(-variable)); }),
		ResultSerial(input, [](double& variable) { return 1.0 / (1.0 + exp(-variable)); })
	);
}

void CSigmoid::CalcResult(CMatrix* refMat, CMatrix* input)
{
	PARALLELBRANCH(
		refMat->GetDataNum(),
		CActFunc::ResultParallel(refMat, input, [](double& variable) { return 1.0 / (1.0 + exp(-variable)); }),
		CActFunc::ResultSerial(refMat, input, [](double& variable) { return 1.0 / (1.0 + exp(-variable)); })
	);
}

CMatrix* CSigmoid::GetDeriviate(CMatrix* input)
{
	PARALLELBRANCH(
		input->GetDataNum(),
		ResultParallel(input, [](double& variable) { return (1.0 / (1.0 + exp(-variable))) * (1.0 - 1.0 / (1.0 + exp(-variable))); }),
		ResultSerial(input, [](double& variable) { return (1.0 / (1.0 + exp(-variable))) * (1.0 - 1.0 / (1.0 + exp(-variable))); })
	);
}

void CSigmoid::CalcDeriviate(CMatrix* refMat, CMatrix* input)
{
	PARALLELBRANCH(
		refMat->GetDataNum(),
		CActFunc::ResultParallel(refMat, input, [](double& variable) { return (1.0 / (1.0 + exp(-variable))) * (1.0 - 1.0 / (1.0 + exp(-variable))); }),
		CActFunc::ResultSerial(refMat, input, [](double& variable) { return (1.0 / (1.0 + exp(-variable))) * (1.0 - 1.0 / (1.0 + exp(-variable))); })
	);
}
#pragma endregion


#pragma region CRelu
CMatrix* CRelu::GetResult(CMatrix* input)
{
	PARALLELBRANCH(
		input->GetDataNum(),
		ResultParallel(input, [](double& variable)
			{
				if (variable > 0)
				{
					return variable;
				}
				else
				{
					return 0.0;
				}
			}),
		ResultSerial(input, [](double& variable)
			{
				if (variable > 0)
				{
					return variable;
				}
				else
				{
					return 0.0;
				}
			})
		);
}

void CRelu::CalcResult(CMatrix* refMat, CMatrix* input)
{
	PARALLELBRANCH(
		refMat->GetDataNum(),
		CActFunc::ResultParallel(refMat, input, [](double& variable)
			{
				if (variable > 0)
				{
					return variable;
				}
				else
				{
					return 0.0;
				}
			}),
		CActFunc::ResultSerial(refMat, input, [](double& variable)
			{
				if (variable > 0)
				{
					return variable;
				}
				else
				{
					return 0.0;
				}
			})	
	);
}


CMatrix* CRelu::GetDeriviate(CMatrix* input)
{
	PARALLELBRANCH(
		input->GetDataNum(),
		ResultParallel(input, [](double& variable)
			{
				if (variable > 0)
				{
					return 1.0;
				}
				else
				{
					return 0.0;
				}
			}),
		ResultSerial(input, [](double& variable)
			{
				if (variable > 0)
				{
					return 1.0;
				}
				else
				{
					return 0.0;
				}
			})
	);
}

void CRelu::CalcDeriviate(CMatrix* refMat, CMatrix* input)
{
	PARALLELBRANCH(
		refMat->GetDataNum(),
		CActFunc::ResultParallel(refMat, input, [](double& variable)
			{
				if (variable > 0)
				{
					return 1.0;
				}
				else
				{
					return 0.0;
				}
			}),
		CActFunc::ResultSerial(refMat, input, [](double& variable)
			{
				if (variable > 0)
				{
					return 1.0;
				}
				else
				{
					return 0.0;
				}
			})
		);
}
#pragma endregion


#pragma region CIdentity
CMatrix* CIdentity::GetResult(CMatrix* input)
{
	PARALLELBRANCH(
		input->GetDataNum(),
		ResultParallel(input, [](double& variable) { return (variable); }),
		ResultSerial(input, [](double& variable) { return (variable); })
	);
}

void CIdentity::CalcResult(CMatrix* refMat, CMatrix* input)
{
	PARALLELBRANCH(
		refMat->GetDataNum(),
		CActFunc::ResultParallel(refMat, input, [](double& variable) { return (variable); }),
		CActFunc::ResultSerial(refMat, input, [](double& variable) { return (variable); })
	);
}

CMatrix* CIdentity::GetDeriviate(CMatrix* input)
{
	PARALLELBRANCH(
		input->GetDataNum(),
		ResultParallel(input, [](double& variable) { return (1.0); }),
		ResultSerial(input, [](double& variable) { return (1.0); })
	);
}


void CIdentity::CalcDeriviate(CMatrix* refMat, CMatrix* input)
{
	PARALLELBRANCH(
		refMat->GetDataNum(),
		CActFunc::ResultParallel(refMat, input, [](double& variable) { return (1.0); }),
		CActFunc::ResultSerial(refMat, input, [](double& variable) { return (1.0); })
	);
}
#pragma endregion