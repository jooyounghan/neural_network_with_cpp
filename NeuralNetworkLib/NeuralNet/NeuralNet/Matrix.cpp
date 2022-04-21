#include "pch.h"
#include "Matrix.h"

/* --------------------------------------------------- */
#pragma region Constructor
CMatrix::CMatrix(const uint32& row_in, const uint32& col_in) : row(row_in), col(col_in), dataNum(row * col), matrixData(nullptr)
{
	matrixData = new double[static_cast<size_t>(dataNum)];
}

CMatrix::CMatrix(const uint32& row_in, const uint32& col_in, double* matrix_data_in) : row(row_in), col(col_in), dataNum(row * col), matrixData(matrix_data_in)
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

void CMatrix::SetMatrixData(double* matrix_data)
{
	DELETEARRPTR(matrixData);
	matrixData = matrix_data;
	return;
}

CMatrix* CMatrix::CopyMatrix()
{
#ifdef PARALLEL
	double* newMatrixData = CopyParallel();
#else
	double* newMatrixData = CopySerial();
#endif
	CMatrix* newMatrix = new CMatrix(row, col, newMatrixData);
	return newMatrix;
}

CMatrix* CMatrix::GetTranspose()
{
#ifdef PARALLEL
	double* newMatrixData = TransposeParallel();
#else
	double* newMatrixData = TransposeSerial();
#endif
	return new CMatrix(col, row, newMatrixData);
}

double* CMatrix::GetTransposeMatrixData()
{
#ifdef PARALLEL
	double* newMatrixData = TransposeParallel();
#else
	double* newMatrixData = TransposeSerial();
#endif
	return newMatrixData;
}

CMatrix* CMatrix::GetMatMul(CMatrix* matrix)
{
	ASSERT_CRASH(this->col == matrix->row);
#ifdef PARALLEL
	double* newMatrixData = MatmulParallel(this, matrix);
#else
	double* newMatrixData = MatmulSerial(this, matrix);
#endif
	CMatrix* newMatrix = new CMatrix(this->row, matrix->col, newMatrixData);
	return newMatrix;
}

void CMatrix::GetMatMul(CMatrix* matrixA, CMatrix* matrixB)
{
	ASSERT_CRASH(this->row == matrixA->row && this->col == matrixB->col);
	ASSERT_CRASH(matrixA->col == matrixB->row);
#ifdef PARALLEL
	CMatrix::MatmulParallel(matrixData, matrixA, matrixB);
#else
	CMatrix::MatmulSerial(matrixData, matrixA, matrixB);
#endif
}

double* CMatrix::GetMatMulData(CMatrix* matrix)
{
	ASSERT_CRASH(this->col == matrix->row);
#ifdef PARALLEL
	double* newMatrixData = MatmulParallel(this, matrix);
#else
	double* newMatrixData = MatmulSerial(this, matrix);
#endif
	return newMatrixData;
}

double* CMatrix::TransposeParallel()
{
	double* newMatrixData = new double[dataNum];

	std::vector<std::future<void>> workThreadVector;
	
	uint32 workSize = static_cast<uint32>(std::ceil(static_cast<double>(dataNum) / THREADNUM));
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

// assume that we can multiply matrix_1 and matrix_2
double* CMatrix::MatmulParallel(CMatrix* matrix_1, CMatrix* matrix_2)
{
	const uint32& newRow = matrix_1->row;
	const uint32& newCol = matrix_2->col;
	const uint32& kValue = matrix_1->col;

	double* matrix1Data = matrix_1->GetMatrixData();
	double* matrix2Data = matrix_2->GetMatrixData();

	double* newMatrixData = new double[newRow * newCol]{ 0 };

	std::vector<std::future<void>> workThreadVector;

	bool RowFlag = false;
	if (newRow > newCol)	RowFlag = true;

	uint32 workSize;
	if (RowFlag)			workSize = static_cast<uint32>(std::ceil(static_cast<double>(newRow) / THREADNUM));
	else					workSize = static_cast<uint32>(std::ceil(static_cast<double>(newCol) / THREADNUM));

	for (uint32 threadNum = 0; threadNum < THREADNUM; ++threadNum)
	{
		uint32 startIdx = threadNum * workSize;
		workThreadVector.push_back(std::async([&, startIdx]()
			{
				if (RowFlag)
				{
					for (uint32 rowIdx = startIdx; rowIdx < startIdx + workSize; ++rowIdx)
					{
						if (rowIdx >= newRow)	break;
						for (uint32 colIdx = 0; colIdx < newCol; ++colIdx)
						{
							const int arrIdx = rowIdx * newCol + colIdx;
							for (uint32 kIdx = 0; kIdx < kValue; ++kIdx)
							{
								newMatrixData[arrIdx] += matrix1Data[rowIdx * kValue + kIdx] * matrix2Data[kIdx * newCol + colIdx];
							}
						}
					}
				}
				else
				{
					for (uint32 rowIdx = 0; rowIdx < newRow; ++rowIdx)
					{
						for (uint32 colIdx = startIdx; colIdx < startIdx + workSize; ++colIdx)
						{
							if (colIdx >= newCol)	break;
							const int arrIdx = rowIdx * newCol + colIdx;
							for (uint32 kIdx = 0; kIdx < kValue; ++kIdx)
							{
								newMatrixData[arrIdx] += matrix1Data[rowIdx * kValue + kIdx] * matrix2Data[kIdx * newCol + colIdx];
							}
						}
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

double* CMatrix::MatmulSerial(CMatrix* matrix_1, CMatrix* matrix_2)
{
	const int& newRow = matrix_1->row;
	const int& newCol = matrix_2->col;
	const int& kValue = matrix_1->col;

	double* matrix1Data = matrix_1->GetMatrixData();
	double* matrix2Data = matrix_2->GetMatrixData();

	double* newMatrixData = new double[newRow * newCol]{ 0 };

	for (int rowIdx = 0; rowIdx < newRow; ++rowIdx)
	{
		for (int colIdx = 0; colIdx < newCol; ++colIdx)
		{
			const int arrIdx = rowIdx * newCol + colIdx;
			for (int kIdx = 0; kIdx < kValue; ++kIdx)
			{
				newMatrixData[arrIdx] += matrix1Data[rowIdx * kValue + kIdx] * matrix2Data[kIdx * newCol + colIdx];
			}
		}
	}
	return newMatrixData;
}

void CMatrix::MatmulParallel(double* newMatrixData, CMatrix* matrix_1, CMatrix* matrix_2)
{
	const uint32& newRow = matrix_1->row;
	const uint32& newCol = matrix_2->col;
	const uint32& kValue = matrix_1->col;

	double* matrix1Data = matrix_1->GetMatrixData();
	double* matrix2Data = matrix_2->GetMatrixData();

	std::vector<std::future<void>> workThreadVector;

	bool RowFlag = false;
	if (newRow > newCol)	RowFlag = true;

	uint32 workSize;
	if (RowFlag)			workSize = static_cast<uint32>(std::ceil(static_cast<double>(newRow) / THREADNUM));
	else					workSize = static_cast<uint32>(std::ceil(static_cast<double>(newCol) / THREADNUM));

	for (uint32 threadNum = 0; threadNum < THREADNUM; ++threadNum)
	{
		uint32 startIdx = threadNum * workSize;
		workThreadVector.push_back(std::async([&, startIdx]()
			{
				if (RowFlag)
				{
					for (uint32 rowIdx = startIdx; rowIdx < startIdx + workSize; ++rowIdx)
					{
						if (rowIdx >= newRow)	break;
						for (uint32 colIdx = 0; colIdx < newCol; ++colIdx)
						{
							const int arrIdx = rowIdx * newCol + colIdx;
							newMatrixData[arrIdx] = 0;
							for (uint32 kIdx = 0; kIdx < kValue; ++kIdx)
							{
								newMatrixData[arrIdx] += matrix1Data[rowIdx * kValue + kIdx] * matrix2Data[kIdx * newCol + colIdx];
							}
						}
					}
				}
				else
				{
					for (uint32 rowIdx = 0; rowIdx < newRow; ++rowIdx)
					{
						for (uint32 colIdx = startIdx; colIdx < startIdx + workSize; ++colIdx)
						{
							if (colIdx >= newCol)	break;
							const int arrIdx = rowIdx * newCol + colIdx;
							newMatrixData[arrIdx] = 0;
							for (uint32 kIdx = 0; kIdx < kValue; ++kIdx)
							{
								newMatrixData[arrIdx] += matrix1Data[rowIdx * kValue + kIdx] * matrix2Data[kIdx * newCol + colIdx];
							}
						}
					}
				}
			}));
	}
	for (uint32 threadNum = 0; threadNum < THREADNUM; ++threadNum)
	{
		workThreadVector[threadNum].wait();
	}
	return;
}

void CMatrix::MatmulSerial(double* newMatrixData, CMatrix* matrix_1, CMatrix* matrix_2)
{
	const int& newRow = matrix_1->row;
	const int& newCol = matrix_2->col;
	const int& kValue = matrix_1->col;

	double* matrix1Data = matrix_1->GetMatrixData();
	double* matrix2Data = matrix_2->GetMatrixData();

	for (int rowIdx = 0; rowIdx < newRow; ++rowIdx)
	{
		for (int colIdx = 0; colIdx < newCol; ++colIdx)
		{
			const int arrIdx = rowIdx * newCol + colIdx;
			newMatrixData[arrIdx] = 0;
			for (int kIdx = 0; kIdx < kValue; ++kIdx)
			{
				newMatrixData[arrIdx] += matrix1Data[rowIdx * kValue + kIdx] * matrix2Data[kIdx * newCol + colIdx];
			}
		}
	}
	return;
}


double* CMatrix::CopyParallel()
{
	std::vector<std::future<void>> workThreadVector;

	double* newMatrixData = new double[dataNum];
	uint32 workSize = static_cast<uint32>(std::ceil(static_cast<double>(dataNum) / THREADNUM));

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
						newMatrixData[idx] = matrixData[idx];
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
double* CMatrix::CopySerial()
{
	double* newMatrixData = new double[dataNum];
	for (uint32 idx = 0; idx < dataNum; ++idx)
	{
		newMatrixData[idx] = matrixData[idx];
	}
	return newMatrixData ;
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
	return SetMatrixData(result);
}

void CMatrix::XavierNormalInitialize(const uint32& node_in, const uint32& node_out)
{
	CRandomGenerator randomGenerator;
	double* result = randomGenerator.getNormalDistVector(dataNum, 0, std::sqrt(2.0 / ((double)node_in + (double)node_out)));
	return SetMatrixData(result);
}

void CMatrix::HeNormalInitialize(const uint32& node_in)
{
	CRandomGenerator randomGenerator;
	double* result = randomGenerator.getNormalDistVector(dataNum, 0, std::sqrt(2.0 / (node_in)));
	return SetMatrixData(result);
}
#pragma endregion
/* --------------------------------------------------- */


#pragma region Interface
double& CMatrix::operator[](const uint32& num)
{
	return matrixData[num];
}
#pragma endregion
