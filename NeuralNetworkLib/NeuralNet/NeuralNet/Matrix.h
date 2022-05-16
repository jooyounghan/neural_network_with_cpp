#pragma once
#include "RandomGenerator.h"
#include "IncludePart.h"


class CMatrix
{
public:
	enum Initializer
	{
		NORMAL,
		XAVIER,
		HE
	};

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

public:
	void SetMatrixData(double* matrix_data);

public:
	CMatrix* GetTranspose();
	double* TransposeMatrixData();
	void Transpose(CMatrix* inputMat);

private:
	double* TransposeParallel();
	double* TransposeSerial();
	static void TransposeParallel(CMatrix* refMatrix, CMatrix* inputMat);
	static void TransposeSerial(CMatrix* refMatrix, CMatrix* inputMat);

public:
	CMatrix* GetMatMul(CMatrix* matrix);
	void MatMul(CMatrix* matrixA, CMatrix* matrixB);
	double* GetMatMulData(CMatrix* matrix);
private:
	double* MatmulParallel(CMatrix* matrix_1, CMatrix* matrix_2);
	double* MatmulSerial(CMatrix* matrix_1, CMatrix* matrix_2);
	static void MatmulParallel(CMatrix* refMatrix, CMatrix* matrix_1, CMatrix* matrix_2);
	static void MatmulSerial(CMatrix* refMatrix, CMatrix* matrix_1, CMatrix* matrix_2);


public:
	CMatrix* GetCopyMatrix();
	std::shared_ptr<CMatrix> GetSharedCopyMatrix();
	void CopyMatrix(CMatrix* input);
private:
	double* CopyParallel();
	double* CopySerial();
	static void CopyParallel(CMatrix* refMat, CMatrix* inputMat);
	static void CopySerial(CMatrix* refMat, CMatrix* inputMat);

public:
	CMatrix* GetElementWiseMul(CMatrix* inputMat);
	void ElementWiseMul(CMatrix* matrixA, CMatrix* matrixB);
private:
	double* ElementWiseMulParallel(CMatrix* inputMat);
	double* ElementWiseMulSerial(CMatrix* inputMat);
	static void ElementWiseMulParallel(CMatrix* refMatrix, CMatrix* matrix_1, CMatrix* matrix_2);
	static void ElementWiseMulSerial(CMatrix* refMatrix, CMatrix* matrix_1, CMatrix* matrix_2);

public:
	void Subtract(CMatrix* input);
private:
	static void SubtractParallel(CMatrix* refMat, CMatrix* inputMat);
	static void SubtractSerial(CMatrix* refMat, CMatrix* inputMat);

public:
	void ConstantMul(const double& constant);
private:
	static void ConstantMulParallel(CMatrix* refMat, const double& constant);
	static void ConstantMulSerial(CMatrix* refMat, const double& constant);

public:
	void PrintData();

public:
	void NormalInitialize(const double& mean, const double& sigma);
	void XavierNormalInitialize(const uint32& node_in, const uint32& node_out);
	void HeNormalInitialize(const uint32& node_in);

public:
	double& operator[](const uint32& num);
};

