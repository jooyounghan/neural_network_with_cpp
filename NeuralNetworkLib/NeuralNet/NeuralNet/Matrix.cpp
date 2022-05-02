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

void CMatrix::Transpose(CMatrix* inputMat)
{
#ifdef PARALLEL
	return TransposeParallel(this, inputMat);
#else
	return TransposeSerial(this, inputMat);
#endif
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

void CMatrix::TransposeParallel(CMatrix* refMatrix, CMatrix* inputMat)
{
	const uint32& row = inputMat->GetRow();
	const uint32& col = inputMat->GetCol();
	const uint32& dataNum = inputMat->GetDataNum();

	double* refMatrixData = refMatrix->GetMatrixData();
	double* inputMatData = inputMat->GetMatrixData();

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
						refMatrixData[colNow * row + rowNow] = inputMatData[rowNow * col + colNow];
					}
				}
			}));
	}
	WAITTHREADVECTOR(workThreadVector);
}

void CMatrix::TransposeSerial(CMatrix* refMatrix, CMatrix* inputMat)
{
	const uint32& row = inputMat->GetRow();
	const uint32& col = inputMat->GetCol();

	double* refMatrixData = refMatrix->GetMatrixData();
	double* inputMatData = inputMat->GetMatrixData();
	for (uint32 r = 0; r < row; ++r)
	{
		for (uint32 c = 0; c < col; ++c)
		{
			refMatrixData[c * row + r] = inputMatData[r * col + c];
		}
	}
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
	return CMatrix::MatmulParallel(this, matrixA, matrixB);
#else
	return CMatrix::MatmulSerial(this, matrixA, matrixB);
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
			newMatrixData[arrIdx] = 0;
			for (int kIdx = 0; kIdx < kValue; ++kIdx)
			{
				newMatrixData[arrIdx] += matrix1Data[rowIdx * kValue + kIdx] * matrix2Data[kIdx * newCol + colIdx];
			}
		}
	}
	return newMatrixData;
}

void CMatrix::MatmulParallel(CMatrix* refMatrix, CMatrix* matrix_1, CMatrix* matrix_2)
{
	const uint32& newRow = matrix_1->row;
	const uint32& newCol = matrix_2->col;
	const uint32& kValue = matrix_1->col;

	double*& refMatrixData = refMatrix->GetMatrixData();
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
							refMatrixData[arrIdx] = 0;
							for (uint32 kIdx = 0; kIdx < kValue; ++kIdx)
							{
								refMatrixData[arrIdx] += matrix1Data[rowIdx * kValue + kIdx] * matrix2Data[kIdx * newCol + colIdx];
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
							refMatrixData[arrIdx] = 0;
							for (uint32 kIdx = 0; kIdx < kValue; ++kIdx)
							{
								refMatrixData[arrIdx] += matrix1Data[rowIdx * kValue + kIdx] * matrix2Data[kIdx * newCol + colIdx];
							}
						}
					}
				}
			}));
	}
	WAITTHREADVECTOR(workThreadVector);
	return;
}

void CMatrix::MatmulSerial(CMatrix* refMatrix, CMatrix* matrix_1, CMatrix* matrix_2)
{
	const int& newRow = matrix_1->row;
	const int& newCol = matrix_2->col;
	const int& kValue = matrix_1->col;

	double*& refMatrixData = refMatrix->GetMatrixData();
	double*& matrix1Data = matrix_1->GetMatrixData();
	double*& matrix2Data = matrix_2->GetMatrixData();

	for (int rowIdx = 0; rowIdx < newRow; ++rowIdx)
	{
		for (int colIdx = 0; colIdx < newCol; ++colIdx)
		{
			const int arrIdx = rowIdx * newCol + colIdx;
			refMatrixData[arrIdx] = 0;
			for (int kIdx = 0; kIdx < kValue; ++kIdx)
			{
				refMatrixData[arrIdx] += matrix1Data[rowIdx * kValue + kIdx] * matrix2Data[kIdx * newCol + colIdx];
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
#pragma region ElementWiseMul
CMatrix* CMatrix::GetElementWiseMul(CMatrix* inputMat)
{
#ifdef PARALLEL
	double* newMatrixData = ElementWiseMulParallel(inputMat);
#else
	double* newMatrixData = ElementWiseMulSerial(input);
#endif
	CMatrix* newMatrix = new CMatrix(row, col, newMatrixData);
	return newMatrix;
}

void CMatrix::ElementWiseMul(CMatrix* matrixA, CMatrix* matrixB)
{
#ifdef PARALLEL
	return CMatrix::ElementWiseMulParallel(this, matrixA, matrixB);
#else
	return CMatrix::ElementWiseMulSerial(this, matrixA, matrixB);
#endif
}

double* CMatrix::ElementWiseMulParallel(CMatrix* inputMat)
{
	const uint32& refDataNum = this->GetDataNum();
	const uint32& inputDataNum = inputMat->GetDataNum();
	ASSERT_CRASH(refDataNum == inputDataNum);

	double*& refMatrixData = this->GetMatrixData();
	double*& inputMatrixData = inputMat->GetMatrixData();

	double* newMatrixData = new double[inputDataNum]{ 0 };

	std::vector<std::future<void>> workThreadVector;

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

double* CMatrix::ElementWiseMulSerial(CMatrix* inputMat)
{
	const uint32& refDataNum = this->GetDataNum();
	const uint32& inputDataNum = inputMat->GetDataNum();
	ASSERT_CRASH(refDataNum == inputDataNum);

	double*& refMatrixData = this->GetMatrixData();
	double*& inputMatrixData = inputMat->GetMatrixData();

	double* newMatrixData = new double[inputDataNum] { 0 };

	for (uint32 idx = 0; idx < inputDataNum; ++idx)
	{
		newMatrixData[idx] = refMatrixData[idx] * inputMatrixData[idx];
	}
	return newMatrixData;
}

void CMatrix::ElementWiseMulParallel(CMatrix* refMatrix, CMatrix* matrix_1, CMatrix* matrix_2)
{
	const uint32& refMatrixDataNum = refMatrix->GetDataNum();
	const uint32& matrix1DataNum = matrix_1->GetDataNum();
	const uint32& matrix2DataNum = matrix_2->GetDataNum();
	ASSERT_CRASH(refMatrixDataNum == matrix1DataNum && refMatrixDataNum == matrix2DataNum);

	double*& refMatrixData = refMatrix->GetMatrixData();
	double*& matrix1Data = matrix_1->GetMatrixData();
	double*& matrix2Data = matrix_2->GetMatrixData();

	std::vector<std::future<void>> workThreadVector;

	uint32 workSize = static_cast<uint32>(std::ceil(static_cast<double>(refMatrixDataNum) / THREADNUM));

	for (uint32 threadNum = 0; threadNum < THREADNUM; ++threadNum)
	{
		uint32 startIdx = threadNum * workSize;
		workThreadVector.push_back(std::async([&, startIdx]()
			{
				for (uint32 idx = startIdx; idx < startIdx + workSize; ++idx)
				{
					if (idx >= refMatrixDataNum)	break;
					else
					{
						refMatrixData[idx] = matrix1Data[idx] * matrix2Data[idx];
					}
				}
			}));
	}
	WAITTHREADVECTOR(workThreadVector);
}

void CMatrix::ElementWiseMulSerial(CMatrix* refMatrix, CMatrix* matrix_1, CMatrix* matrix_2)
{
	const uint32& refMatrixDataNum = refMatrix->GetDataNum();
	const uint32& matrix1DataNum = matrix_1->GetDataNum();
	const uint32& matrix2DataNum = matrix_2->GetDataNum();
	ASSERT_CRASH(refMatrixDataNum == matrix1DataNum && refMatrixDataNum == matrix2DataNum);

	double*& refMatrixData = refMatrix->GetMatrixData();
	double*& matrix1Data = matrix_1->GetMatrixData();
	double*& matrix2Data = matrix_2->GetMatrixData();

	for (uint32 idx = 0; idx < refMatrixDataNum; ++idx)
	{
		refMatrixData[idx] = matrix1Data[idx] * matrix2Data[idx];
	}
}
#pragma endregion
/* --------------------------------------------------- */

/* --------------------------------------------------- */
#pragma region Subtract
void CMatrix::Subtract(CMatrix* input)
{
#ifdef PARALLEL
	return CMatrix::SubtractParallel(this, input);
#else
	return CMatrix::SubtractSerial(this, input);
#endif
}

void CMatrix::SubtractParallel(CMatrix* refMat, CMatrix* inputMat)
{
	const uint32& refMatDataNum = refMat->GetDataNum();
	const uint32& inputMatDataNum = inputMat->GetDataNum();
	ASSERT_CRASH(refMatDataNum == inputMatDataNum);

	double*& refMatData = refMat->GetMatrixData();
	double*& inputMatData = inputMat->GetMatrixData();

	std::vector<std::future<void>> workThreadVector;

	uint32 workSize = static_cast<uint32>(std::ceil(static_cast<double>(refMatDataNum) / THREADNUM));

	for (uint32 threadNum = 0; threadNum < THREADNUM; ++threadNum)
	{
		uint32 startIdx = threadNum * workSize;
		workThreadVector.push_back(std::async([&, startIdx]()
			{
				for (uint32 idx = startIdx; idx < startIdx + workSize; ++idx)
				{
					if (idx >= refMatDataNum)	break;
					else
					{
						refMatData[idx] = refMatData[idx] - inputMatData[idx];
					}
				}
			}));
	}
	WAITTHREADVECTOR(workThreadVector);
}
void CMatrix::SubtractSerial(CMatrix* refMat, CMatrix* inputMat)
{
	const uint32& refMatDataNum = refMat->GetDataNum();
	const uint32& inputMatDataNum = inputMat->GetDataNum();
	ASSERT_CRASH(refMatDataNum == inputMatDataNum);

	double*& refMatData = refMat->GetMatrixData();
	double*& inputMatData = inputMat->GetMatrixData();

	for (uint32 idx = 0; idx < refMatDataNum; ++idx)
	{
		refMatData[idx] = refMatData[idx] - inputMatData[idx];
	}
}
#pragma endregion

#pragma region ConstantMul
void CMatrix::ConstantMul(const double& constant)
{
#ifdef PARALLEL
	return CMatrix::ConstantMulParallel(this, constant);
#else
	return CMatrix::ConstantMulSerial(this, constant);
#endif
}
void CMatrix::ConstantMulParallel(CMatrix* refMat, const double& constant)
{
	const uint32& refMatDataNum = refMat->GetDataNum();
	double*& refMatData = refMat->GetMatrixData();

	std::vector<std::future<void>> workThreadVector;

	uint32 workSize = static_cast<uint32>(std::ceil(static_cast<double>(refMatDataNum) / THREADNUM));

	for (uint32 threadNum = 0; threadNum < THREADNUM; ++threadNum)
	{
		uint32 startIdx = threadNum * workSize;
		workThreadVector.push_back(std::async([&, startIdx]()
			{
				for (uint32 idx = startIdx; idx < startIdx + workSize; ++idx)
				{
					if (idx >= refMatDataNum)	break;
					else
					{
						refMatData[idx] = refMatData[idx] * constant;
					}
				}
			}));
	}
	WAITTHREADVECTOR(workThreadVector);
}
void CMatrix::ConstantMulSerial(CMatrix* refMat, const double& constant)
{
	const uint32& refMatDataNum = refMat->GetDataNum();
	double*& refMatData = refMat->GetMatrixData();

	for (uint32 idx = 0; idx < refMatDataNum; ++idx)
	{
		refMatData[idx] = refMatData[idx] * constant;
	}
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