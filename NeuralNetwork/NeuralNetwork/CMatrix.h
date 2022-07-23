#pragma once

class CMatrix
{	
friend class CInitializer;

protected:
	UINT row;
	UINT col;
	UINT size;
	double* data;

public:
	CMatrix();
	CMatrix(const CMatrix& mat);
	CMatrix(CMatrix&& mat) noexcept;
	CMatrix(const UINT& row, const UINT& col);
	~CMatrix();

public:
	void DeleteData();

public:
	CMatrix& operator= (const CMatrix& mat);
	CMatrix& operator= (CMatrix&& mat) noexcept;


	CMatrix operator+(const CMatrix& mat);
	CMatrix operator-(const CMatrix& mat);
	CMatrix operator*(const CMatrix& mat);
	double& operator[](const UINT& idx);

	CMatrix MatMul(const CMatrix& mat);
	CMatrix Transpose();

	void RefAdd(const CMatrix& mat, CMatrix& ref);
	void RefSub(const CMatrix& mat, CMatrix& ref);
	void RefElementwiseMul(const CMatrix& mat, CMatrix& ref);

	void RefMatMul(const CMatrix& mat, CMatrix& ref);
	void RefTranspose(CMatrix& ref);

	void PrintData();

	const UINT& GetRow();
	const UINT& GetCol();
	double* GetData();

	CMatrix Copy();
	double GetSum();
};


