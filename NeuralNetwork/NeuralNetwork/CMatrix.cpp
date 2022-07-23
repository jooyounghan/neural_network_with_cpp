#include "pch.h"
#include "CMatrix.h"


CMatrix::CMatrix()
{
}

CMatrix::CMatrix(const CMatrix& mat)
{
	*this = mat;
}

CMatrix::CMatrix(CMatrix&& mat) noexcept
{
	*this = std::move(mat);
}

CMatrix::CMatrix(const UINT& row, const UINT& col)
	: row(row), col(col), size(row* col), data(nullptr)
{
	data = new double[size];
}

CMatrix::~CMatrix() { if (data != nullptr) delete[] data; }

void CMatrix::DeleteData()
{
	DELETEPTR(data);
}

CMatrix& CMatrix::operator= (const CMatrix& mat)
{
	row = mat.row;
	col = mat.col;
	size = mat.size;

	DeleteData();

	data = new double[size];
	for (UINT idx = 0; idx < size; ++idx)
	{
		data[idx] = mat.data[idx];
	}
	return *this;
}

CMatrix& CMatrix::operator= (CMatrix&& mat) noexcept
{
	row = mat.row;
	col = mat.col;
	size = mat.size;
	data = mat.data;
	mat.data = nullptr;
	return *this;
}

CMatrix CMatrix::operator+(const CMatrix& mat)
{
	ASSERT_CRASH(row == mat.row && col == mat.col);

	CMatrix result{ row, col };
	for (UINT idx = 0; idx < size; ++idx)
	{
		result.data[idx] = data[idx] + mat.data[idx];
	}
	return result;
}

CMatrix CMatrix::operator-(const CMatrix& mat)
{
	ASSERT_CRASH(row == mat.row && col == mat.col);

	CMatrix result{ row, col };
	for (UINT idx = 0; idx < size; ++idx)
	{
		result.data[idx] = data[idx] - mat.data[idx];
	}
	return result;
}

CMatrix CMatrix::operator*(const CMatrix& mat)
{
	ASSERT_CRASH(row == mat.row && col == mat.col);

	CMatrix result{ row, col };
	for (UINT idx = 0; idx < size; ++idx)
	{
		result.data[idx] = data[idx] * mat.data[idx];
	}
	return result;
}

double& CMatrix::operator[](const UINT& idx)
{
	return data[idx];
}

CMatrix CMatrix::MatMul(const CMatrix& mat)
{
	ASSERT_CRASH(col == mat.row);

	CMatrix result{ row, mat.col };
	for (UINT rIdx = 0; rIdx < row; ++rIdx)
	{
		for (UINT cIdx = 0; cIdx < mat.col; ++cIdx)
		{
			const UINT& sizeIdx = rIdx * mat.col + cIdx;
			result.data[sizeIdx] = 0;
			for (UINT kIdx = 0; kIdx < col; ++kIdx)
			{
				result.data[sizeIdx] += data[rIdx * col + kIdx] * mat.data[kIdx * mat.col + cIdx];
			}
		}
	}
	return result;
}

CMatrix CMatrix::Transpose()
{
	CMatrix result{ col, row };
	for (UINT rIdx = 0; rIdx < row; ++rIdx)
	{
		for (UINT cIdx = 0; cIdx < col; ++cIdx)
		{
			result.data[cIdx * row + rIdx] = data[rIdx * col + cIdx];
		}
	}
	return result;
}

void CMatrix::RefAdd(const CMatrix& mat, CMatrix& ref)
{
	ASSERT_CRASH(row == mat.row && col == mat.col);
	ASSERT_CRASH(row == ref.row && col == ref.col);

	for (UINT idx = 0; idx < size; ++idx)
	{
		ref.data[idx] = data[idx] + mat.data[idx];
	}
}

void CMatrix::RefSub(const CMatrix& mat, CMatrix& ref)
{
	ASSERT_CRASH(row == mat.row && col == mat.col);
	ASSERT_CRASH(row == ref.row && col == ref.col);

	for (UINT idx = 0; idx < size; ++idx)
	{
		ref.data[idx] = data[idx] - mat.data[idx];
	}
}

void CMatrix::RefElementwiseMul(const CMatrix& mat, CMatrix& ref)
{
	ASSERT_CRASH(row == mat.row && col == mat.col);
	ASSERT_CRASH(row == ref.row && col == ref.col);

	for (UINT idx = 0; idx < size; ++idx)
	{
		ref.data[idx] = data[idx] * mat.data[idx];
	}
}

void CMatrix::RefMatMul(const CMatrix& mat, CMatrix& ref)
{
	ASSERT_CRASH(col == mat.row);
	ASSERT_CRASH(row == ref.row && mat.col == ref.col);

	for (UINT rIdx = 0; rIdx < row; ++rIdx)
	{
		for (UINT cIdx = 0; cIdx < mat.col; ++cIdx)
		{
			const UINT& sizeIdx = rIdx * mat.col + cIdx;
			ref.data[sizeIdx] = 0;
			for (UINT kIdx = 0; kIdx < col; ++kIdx)
			{
				ref.data[sizeIdx] += data[rIdx * col + kIdx] * mat.data[kIdx * mat.col + cIdx];
			}
		}
	}
}

void CMatrix::RefTranspose(CMatrix& ref)
{
	ASSERT_CRASH(row == ref.col && col == ref.row);
	for (UINT rIdx = 0; rIdx < row; ++rIdx)
	{
		for (UINT cIdx = 0; cIdx < col; ++cIdx)
		{
			ref.data[cIdx * row + rIdx] = data[rIdx * col + cIdx];
		}
	}
}

void CMatrix::PrintData()
{
	for (UINT rIdx = 0; rIdx < row; ++rIdx)
	{
		for (UINT cIdx = 0; cIdx < col; ++cIdx)
		{
			std::cout << data[rIdx * col + cIdx] << " ";
		}
		std::cout << '\n';
	}
}

const UINT& CMatrix::GetRow() { return row; }

const UINT& CMatrix::GetCol() { return col; }

double* CMatrix::GetData() { return data; }

CMatrix CMatrix::Copy()
{
	CMatrix result = CMatrix{ row, col };
	for (UINT rIdx = 0; rIdx < row; ++rIdx)
	{
		for (UINT cIdx = 0; cIdx < col; ++cIdx)
		{
			result.data[cIdx * row + rIdx] = data[rIdx * col + cIdx];
		}
	}
	return result;
}

double CMatrix::GetSum()
{
	double result = 0;
	for (UINT idx = 0; idx < size; ++idx)
	{
		result += data[idx];
	}
	return result;
}
