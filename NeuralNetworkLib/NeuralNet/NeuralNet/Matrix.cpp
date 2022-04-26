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
const uint32& CMatrix::GetRow()
{
	return row;
}

const uint32& CMatrix::GetCol()
{
	return col;
}

const uint32& CMatrix::GetDataNum()
{
	return dataNum;
}

double*& CMatrix::GetMatrixData()
{
	return matrixData;
}
#pragma endregion
/* --------------------------------------------------- */

/* --------------------------------------------------- */
#pragma region Setter
void CMatrix::SetMatrixData(double* matrix_data)
{
	DELETEARRPTR(matrixData);
	matrixData = matrix_data;
	return;
}
#pragma endregion
/* --------------------------------------------------- */

/* --------------------------------------------------- */
#pragma region Transpose
CMatrix* CMatrix::GetTranspose()
{
#ifdef PARALLEL
	double* newMatrixData = TransposeParallel();
#else
	double* newMatrixData = TransposeSerial();
#endif
	return new CMatrix(col, row, newMatrixData);
}

double* CMatrix::TransposeMatrixData()
{
#ifdef PARALLEL
	double* newMatrixData = TransposeParallel();
#else
	double* newMatrixData = TransposeSerial();
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
	WAITTHREADVECTOR(workThreadVector);
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
#pragma region MatMul
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

void CMatrix::MatMul(CMatrix* matrixA, CMatrix* matrixB)
{
	ASSERT_CRASH(this->row == matrixA->row && this->col == matrixB->col);
	ASSERT_CRASH(matrixA->col == matrixB->row);
#ifdef PARALLEL
	CMatrix::MatmulParallel(this, matrixA, matrixB);
#else
	CMatrix::MatmulSerial(this, matrixA, matrixB);
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

// assume that we can multiply matrix_1 and matrix_2
double* CMatrix::MatmulParallel(CMatrix* matrix_1, CMatrix* matrix_2)
{
	const uint32& newRow = matrix_1->row;
	const uint32& newCol = matrix_2->col;
	const uint32& kValue = matrix_1->col;

	double*& matrix1Data = matrix_1->GetMatrixData();
	double*& matrix2Data = matrix_2->GetMatrixData();

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
	WAITTHREADVECTOR(workThreadVector);
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

void CMatrix::MatmulParallel(CMatrix* newMatrix, CMatrix* matrix_1, CMatrix* matrix_2)
{
	const uint32& newRow = matrix_1->row;
	const uint32& newCol = matrix_2->col;
	const uint32& kValue = matrix_1->col;

	double*& newMatrixData = newMatrix->GetMatrixData();
	double*& matrix1Data = matrix_1->GetMatrixData();
	double*& matrix2Data = matrix_2->GetMatrixData();

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
	WAITTHREADVECTOR(workThreadVector);
	return;
}

void CMatrix::MatmulSerial(CMatrix* newMatrix, CMatrix* matrix_1, CMatrix* matrix_2)
{
	const int& newRow = matrix_1->row;
	const int& newCol = matrix_2->col;
	const int& kValue = matrix_1->col;

	double*& newMatrixData = newMatrix->GetMatrixData();
	double*& matrix1Data = matrix_1->GetMatrixData();
	double*& matrix2Data = matrix_2->GetMatrixData();

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
#pragma endregion
/* --------------------------------------------------- */

/* --------------------------------------------------- */
#pragma region Copy
CMatrix* CMatrix::GetCopyMatrix()
{
#ifdef PARALLEL
	double* newMatrixData = CopyParallel();
#else
	double* newMatrixData = CopySerial();
#endif
	CMatrix* newMatrix = new CMatrix(row, col, newMatrixData);
	return newMatrix;
}

void CMatrix::CopyMatrix(CMatrix* input)
{
#ifdef PARALLEL
	return CMatrix::CopyParallel(this, input);
#else
	return CMatrix::CopySerial(this, input);
#endif
}

double* CMatrix::CopyParallel()
{
	std::vector<std::future<void>> workThreadVector;

	double* newMatrixData = new double[dataNum];
	uint32 workSize = static_cast<uint32>(std::ceil(static_cast<double>(dataNum) / THREADNUM));

	for (uint32 threadNum = 0; threadNum < THREADNUM; ++threadNum)
	{
		uint32 startIdx = threadNum * workSize;
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

	WAITTHREADVECTOR(workThreadVector);

	return newMatrixData;
}
double* CMatrix::CopySerial()
{
	double* newMatrixData = new double[dataNum];
	for (uint32 idx = 0; idx < dataNum; ++idx)
	{
		newMatrixData[idx] = matrixData[idx];
	}
	return newMatrixData;
}
void CMatrix::CopyParallel(CMatrix* refMat, CMatrix* inputMat)
{
	const uint32& refDataNum = refMat->GetDataNum();
	const uint32& inputDataNum = inputMat->GetDataNum();
	ASSERT_CRASH(refDataNum == inputDataNum);
	double*& refMatrixData = refMat->GetMatrixData();
	double*& inputMatrixData = inputMat->GetMatrixData();

	std::vector<std::future<void>> workThreadVector;

	uint32 workSize = static_cast<uint32>(std::ceil(static_cast<double>(inputDataNum) / THREADNUM));

	for (uint32 threadNum = 0; threadNum < THREADNUM; ++threadNum)
	{
		int32 startIdx = threadNum * workSize;
		workThreadVector.push_back(std::async([&, startIdx]()
			{
				for (uint32 idx = startIdx; idx < startIdx + workSize; ++idx)
				{
					if (idx >= inputDataNum) break;
					else
					{
						refMatrixData[idx] = inputMatrixData[idx];
					}
				}
			}));
	}

	WAITTHREADVECTOR(workThreadVector);
}
void CMatrix::CopySerial(CMatrix* refMat, CMatrix* inputMat)
{
	const uint32& refDataNum = refMat->GetDataNum();
	const uint32& inputDataNum = inputMat->GetDataNum();
	ASSERT_CRASH(refDataNum == inputDataNum);
	double*& refMatrixData = refMat->GetMatrixData();
	double*& inputMatrixData = inputMat->GetMatrixData();

	for (uint32 idx = 0; idx < inputDataNum; ++idx)
	{
		refMatrixData[idx] = inputMatrixData[idx];
	}
}
#pragma endregion
/* --------------------------------------------------- */

/* --------------------------------------------------- */
#pragma region ElementWise
CMatrix* CMatrix::GetElementWiseMul(CMatrix* input)
{
#ifdef PARALLEL
	double* newMatrixData = ElementWiseMulParallel(input);
#else
	double* newMatrixData = ElementWiseMulSerial(input);
#endif
	CMatrix* newMatrix = new CMatrix(row, col, newMatrixData);
	return newMatrix;
}

double* CMatrix::ElementWiseMulParallel(CMatrix* inputMatrix)
{
	const uint32& refDataNum = this->GetDataNum();
	const uint32& inputDataNum = inputMatrix->GetDataNum();
	ASSERT_CRASH(refDataNum == inputDataNum);

	double*& refMatrixData = this->GetMatrixData();
	double*& inputMatrixData = inputMatrix->GetMatrixData();

	double* newMatrixData = new double[inputDataNum]{ 0 };

	std::vector<std::future<void>> workThreadVector;

	bool RowFlag = false;

	uint32 workSize = static_cast<uint32>(std::ceil(static_cast<double>(inputDataNum) / THREADNUM));
	
	for (uint32 threadNum = 0; threadNum < THREADNUM; ++threadNum)
	{
		uint32 startIdx = threadNum * workSize;
		workThreadVector.push_back(std::async([&, startIdx]()
			{
				for (uint32 idx = startIdx; idx < startIdx + workSize; ++idx)
				{
					if (idx >= dataNum)	break;
					else
					{
						newMatrixData[idx] = refMatrixData[idx] * inputMatrixData[idx];
					}
				}
			}));
	}
	WAITTHREADVECTOR(workThreadVector);
	return newMatrixData;
}

double* CMatrix::ElementWiseMulSerial(CMatrix* inputMatrix)
{
	const uint32& refDataNum = this->GetDataNum();
	const uint32& inputDataNum = inputMatrix->GetDataNum();
	ASSERT_CRASH(refDataNum == inputDataNum);

	double*& refMatrixData = this->GetMatrixData();
	double*& inputMatrixData = inputMatrix->GetMatrixData();

	double* newMatrixData = new double[inputDataNum] { 0 };

	for (uint32 idx = 0; idx < inputDataNum; ++idx)
	{
		newMatrixData[idx] = refMatrixData[idx] * inputMatrixData[idx];
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

/* --------------------------------------------------- */
#pragma region Interface
double& CMatrix::operator[](const uint32& num)
{
	return matrixData[num];
}
#pragma endregion
/* --------------------------------------------------- */