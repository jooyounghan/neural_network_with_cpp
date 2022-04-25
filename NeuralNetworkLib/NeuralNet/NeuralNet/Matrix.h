#pragma once
#include "RandomGenerator.h"
#include "IncludePart.h"

class CMatrix
{
private:
	uint32 row;
	uint32 col;
	uint32 dataNum;
	double* matrixData;

public:
	CMatrix(const uint32& row_in, const uint32& col_in);
	CMatrix(const uint32& row_in, const uint32& col_in, double* matrix_data_in);
	~CMatrix();

public:
	const uint32& GetRow();
	const uint32& GetCol();
	const uint32& GetDataNum();
	double*& GetMatrixData();
	void SetMatrixData(double* matrix_data);

public:
	CMatrix* GetTranspose();
	double* GetTransposeMatrixData();

public:
	CMatrix* GetMatMul(CMatrix* matrix);
	void GetMatMul(CMatrix* matrixA, CMatrix* matrixB);
	double* GetMatMulData(CMatrix* matrix);

private:
	double* TransposeParallel();
	double* TransposeSerial();
	double* MatmulParallel(CMatrix* matrix_1, CMatrix* matrix_2);
	double* MatmulSerial(CMatrix* matrix_1, CMatrix* matrix_2);

	static void MatmulParallel(CMatrix* newMatrix, CMatrix* matrix_1, CMatrix* matrix_2);
	static void MatmulSerial(CMatrix* newMatrix, CMatrix* matrix_1, CMatrix* matrix_2);


public:
	CMatrix* CopyMatrix();
	static void CopyMatrix(CMatrix* refMat, CMatrix* input);

private:
	double* CopyParallel();
	double* CopySerial();
	
	static void CopyParallel(CMatrix* refMat, CMatrix* inputMat);
	static void CopySerial(CMatrix* refMat, CMatrix* inputMat);

public:
	void PrintData();

public:
	void NormalInitialize(const double& mean, const double& sigma);
	void XavierNormalInitialize(const uint32& node_in, const uint32& node_out);
	void HeNormalInitialize(const uint32& node_in);

public:
	double& operator[](const uint32& num);
};

