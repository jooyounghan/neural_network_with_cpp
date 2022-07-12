#pragma once

class CMatrix
{
private:
	unsigned int row;
	unsigned int col;
	unsigned int size;
	float* data;

public:
	CMatrix(const CMatrix& mat);
	CMatrix(CMatrix&& mat) noexcept;
	CMatrix(const unsigned int& row, const unsigned int& col);
	~CMatrix();

public:
	void DeleteData();

public:
	CMatrix& operator= (const CMatrix& mat);
	CMatrix& operator= (CMatrix&& mat) noexcept;


	CMatrix operator+(const CMatrix& mat);
	CMatrix operator-(const CMatrix& mat);
	CMatrix operator*(const CMatrix& mat);

	CMatrix MatMul(const CMatrix& mat);
	CMatrix Transpose();

	void RefAdd(const CMatrix& mat, CMatrix& ref)
	{

	}

	void RefSub(const CMatrix& mat, CMatrix& ref)
	{

	}

	void RefElementwiseMul(const CMatrix& mat, CMatrix& ref)
	{

	}

	void RefMatMul(const CMatrix& mat, CMatrix& ref)
	{

	}
	void RefTranspose(const CMatrix& mat, CMatrix& ref)
	{

	}
};

