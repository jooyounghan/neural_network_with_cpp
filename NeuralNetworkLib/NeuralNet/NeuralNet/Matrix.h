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
	uint32 GetRow();
	uint32 GetCol();
	uint32 GetDataNum();
	double* GetMatrixData();
	void SetMatrixData(double* matrix_data);
	CMatrix* CopyMatrix();
public:
	void Transpose();
	double* GetTransposeMatrixData();
	CMatrix* GetMatMul(CMatrix& matrix);
	double* GetMatMulData(CMatrix& matrix);

private:
	double* TransposeParallel();
	double* TransposeSerial();
	double* MatmulParallel(CMatrix* matrix_1, CMatrix* matrix_2);
	double* MatmulSerial(CMatrix* matrix_1, CMatrix* matrix_2);
	double* CopyParallel();
	double* CopySerial();

public:
	void PrintData();

public:
	void NormalInitialize(const double& mean, const double& sigma);
	void XavierNormalInitialize(const uint32& node_in, const uint32& node_out);
	void HeNormalInitialize(const uint32& node_in);

public:
	double& operator[](const uint32& num);
};

