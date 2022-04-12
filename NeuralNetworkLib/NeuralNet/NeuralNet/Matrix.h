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
	CMatrix(const int32& row_in, const int32& col_in);
	CMatrix(const int32& row_in, const int32& col_in, double* matrix_data_in);
	~CMatrix();

// Getter
public:
	uint32 GetRow();
	uint32 GetCol();
	uint32 GetDataNum();
	double* GetMatrixData();

public:
	void Transpose();
	double* GetTransposeMatrixData();
	CMatrix* GetMatMul(CMatrix& matrix);
	double* GetMatMulData(CMatrix& matrix);

private:
	void ChangeMatrixData(double* matrix_data);
	double* TransposeParallel();
	double* TransposeSerial();

private:
	double* MatmulParallel(CMatrix* matrix_1, CMatrix* matrix_2);
	double* MatmulSerial(CMatrix* matrix_1, CMatrix* matrix_2);

public:
	void PrintData();

public:
	void NormalInitialize(const double& mean, const double& sigma);
	void XavierNormalInitialize(const int& node_in, const int& node_out);
	void HeNormalInitialize(const int& node_in);
};

