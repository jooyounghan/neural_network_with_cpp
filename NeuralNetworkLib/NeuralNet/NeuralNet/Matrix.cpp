#include "pch.h"
#include "Matrix.h"

/* --------------------------------------------------- */
#pragma region Constructor
CMatrix::CMatrix(const int32& row_in, const int32& col_in) : row(row_in), col(col_in), dataNum(row * col), matrixData(nullptr)
{
	matrixData = new double[static_cast<size_t>(dataNum)];
}

CMatrix::CMatrix(const int32& row_in, const int32& col_in, double* matrix_data_in) : row(row_in), col(col_in), dataNum(row * col), matrixData(matrix_data_in)
{

}
#pragma endregion
/* --------------------------------------------------- */


/* --------------------------------------------------- */
#pragma region Destructor
CMatrix::~CMatrix()
{
	DELETEARRPTR(matrixData);
}
#pragma endregion
/* --------------------------------------------------- */


/* --------------------------------------------------- */
#pragma region Getter
uint32 CMatrix::GetRow()
{
	return row;
}

uint32 CMatrix::GetCol()
{
	return col;
}

uint32 CMatrix::GetDataNum()
{
	return dataNum;
}

double* CMatrix::GetMatrixData()
{
	return matrixData;
}
#pragma endregion
/* --------------------------------------------------- */


/* --------------------------------------------------- */
#pragma region Data Setter
void CMatrix::Transpose()
{
#ifdef ASYNC
	double* newMatrixData = TransposeParallel();
#else
	double* newMatrixData = TransposeSerial();
#endif
	ChangeMatrixData(newMatrixData);
	std::swap(row, col);
}

double* CMatrix::GetTransposeMatrixData()
{
#ifdef ASYNC
	double* newMatrixData = TransposeParallel();
#else
	double* newMatrixData = TransposeSerial();
#endif
	return newMatrixData;
}

void CMatrix::ChangeMatrixData(double* matrix_data)
{
	DELETEARRPTR(matrixData);
	matrixData = matrix_data;
	return;
}

double* CMatrix::TransposeParallel()
{
	double* newMatrixData = new double[dataNum];

	std::vector<std::future<void>> workThreadVector;
	
	uint32 workSize = dataNum / 2;
	if (workSize == 0)	workSize = 1;

	for (uint32 threadNum = 0; threadNum < THREADNUM; ++threadNum)
	{
		int32 startIdx = threadNum * workSize;
		workThreadVector.push_back(std::async([&, startIdx]()
			{
				for (uint32 idx = startIdx; idx < startIdx + workSize; ++idx)
				{
					if (idx >= dataNum)	break;
					else
					{
						const int32 rowNow = idx / col;
						const int32 colNow = idx % col;
						newMatrixData[colNow * row + rowNow] = matrixData[rowNow * col + colNow];
					}
				}
			}));
	}
	for (uint32 threadNum = 0; threadNum < THREADNUM; ++threadNum)
	{
		workThreadVector[threadNum].wait();
	}
	return newMatrixData;
}

double* CMatrix::TransposeSerial()
{
	double* newMatrixData = new double[dataNum];
	for (uint32 r = 0; r < row; ++r)
	{
		for (uint32 c = 0; c < col; ++c)
		{
			newMatrixData[c * row + r] = matrixData[r * col + c];
		}
	}
	return newMatrixData;
}
#pragma endregion
/* --------------------------------------------------- */


/* --------------------------------------------------- */
#pragma region Utility
void CMatrix::PrintData()
{
	for (uint32 r = 0; r < row; ++r)
	{
		for (uint32 c = 0; c < col; ++c)
		{
			std::cout << matrixData[r * col + c] << " ";
		}
		std::cout << "\n";
	}
}
#pragma endregion
/* --------------------------------------------------- */

/* --------------------------------------------------- */
#pragma region Initializer
void CMatrix::NormalInitialize(const double& mean, const double& sigma)
{
	CRandomGenerator randomGenerator;
	double* result = randomGenerator.getNormalDistVector(dataNum, mean, sigma);
	return ChangeMatrixData(result);
}

void CMatrix::XavierNormalInitialize(const int& node_in, const int& node_out)
{
	CRandomGenerator randomGenerator;
	double* result = randomGenerator.getNormalDistVector(dataNum, 0, std::sqrt(2.0 / ((double)node_in + (double)node_out)));
	return ChangeMatrixData(result);
}

void CMatrix::HeNormalInitialize(const int& node_in)
{
	CRandomGenerator randomGenerator;
	double* result = randomGenerator.getNormalDistVector(dataNum, 0, std::sqrt(2.0 / (node_in)));
	return ChangeMatrixData(result);
}
#pragma endregion
/* --------------------------------------------------- */
