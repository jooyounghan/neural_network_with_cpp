#pragma once
#include "RandomGenerator.h"

class CMatrix
{
private:
	int32 row;
	int32 col;
	int32 dataNum;
	double* matrixData;

public:
	CMatrix(const int32& row_in, const int32& col_in);
	CMatrix(const int32& row_in, const int32& col_in, double* matrix_data_in);
	~CMatrix();

// Getter
public:
	int32 GetRow();
	int32 GetCol();
	int32 GetDataNum();
	double* GetMatrixData();

public:
	void ChangeMatrixData(double* matrix_data);
	void Transpose();
	double* GetTransposeMatrixData();

public:
	void NormalInitialize(const double& mean, const double& sigma);
	void XavierNormalInitialize(const int& node_in, const int& node_out);
	void HeNormalInitialize(const int& node_in);
};

