#include "pch.h"
#include "Matrix.h"

/* --------------------------------------------------- */
#pragma region Constructor
CMatrix::CMatrix(const int32& row_in, const int32& col_in) : row(row_in), col(col_in), dataNum(row * col) matrixData(nullptr)
{
	matrixData = new double[static_cast<size_t>(dataNum)];
}

CMatrix::CMatrix(const int32& row_in, const int32& col_in, double* matrix_data_in) : row(row_in), col(col_in), dataNum(row* col) matrixData(matrix_data_in)
{

}
#pragma endregion
/* --------------------------------------------------- */


/* --------------------------------------------------- */
#pragma region Destructor
CMatrix::~CMatrix()
{
	delete[] matrixData;
}
#pragma endregion
/* --------------------------------------------------- */


/* --------------------------------------------------- */
#pragma region Getter
int32 CMatrix::GetRow()
{
	return row;
}

int32 CMatrix::GetCol()
{
	return col;
}

int32 CMatrix::GetDataNum()
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
void CMatrix::ChangeMatrixData(double* matrix_data)
{
	if (matrixData != nullptr)
	{
		delete[] matrixData;
		matrixData = nullptr;
	}

	matrixData = matrix_data;
	return;
}
#pragma endregion

inline void CMatrix::Transpose() {}

inline double* CMatrix::GetTransposeMatrixData() {}
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
	double* result = randomGenerator.getNormalDistVector(dataNum, 0, std::sqrt(2.0 / (node_in + node_out)));
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
