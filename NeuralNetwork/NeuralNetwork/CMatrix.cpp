#include "pch.h"
#include "CMatrix.h"
#include "CInitializer.h"


CMatrix::CMatrix(const CMatrix& mat)
{
	*this = mat;
}

CMatrix::CMatrix(CMatrix&& mat) noexcept
{
	*this = std::move(mat);
}

CMatrix::CMatrix(const unsigned int& row, const unsigned int& col)
	: row(row), col(col), size(row* col), data(nullptr)
{
	data = new float[size];
}

CMatrix::~CMatrix() { if (data != nullptr) delete[] data; }

void CMatrix::DeleteData()
{
	if (data != nullptr)
	{
		delete[] data;
		data = nullptr;
	}
}

CMatrix& CMatrix::operator= (const CMatrix& mat)
{
	row = mat.row;
	col = mat.col;
	size = mat.size;

	DeleteData();

	data = new float[size];
	for (unsigned int idx = 0; idx < size; ++idx)
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
	for (unsigned int idx = 0; idx < size; ++idx)
	{
		result.data[idx] = data[idx] + mat.data[idx];
	}
	return result;
}

CMatrix CMatrix::operator-(const CMatrix& mat)
{
	ASSERT_CRASH(row == mat.row && col == mat.col);

	CMatrix result{ row, col };
	for (unsigned int idx = 0; idx < size; ++idx)
	{
		result.data[idx] = data[idx] - mat.data[idx];
	}
	return result;
}

CMatrix CMatrix::operator*(const CMatrix& mat)
{
	ASSERT_CRASH(row == mat.row && col == mat.col);

	CMatrix result{ row, col };
	for (unsigned int idx = 0; idx < size; ++idx)
	{
		result.data[idx] = data[idx] * mat.data[idx];
	}
	return result;
}

//CMatrix CMatrix::MatMul(const CMatrix& mat)
//{
//	ASSERT_CRASH(col == mat.row);
//}
//
//CMatrix CMatrix::Transpose()
//{
//
//}
