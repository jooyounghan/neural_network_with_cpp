#include "pch.h"
#include <iostream>
#include <pch.h>


int main()
{	
	CMatrix a = CMatrix(3, 4);
	a.XavierNormalInitialize(3, 4);

	a.PrintData();
	std::cout << "\n";

	a.Transpose();
	a.PrintData();
	std::cout << "\n";

	//double* b = a.GetTransposeMatrixData();
	//for (int c = 0; c < a.GetCol(); ++c)
	//{
	//	for (int r = 0; r < a.GetRow(); ++r)
	//	{
	//		std::cout << b[c * a.GetRow() + r] << " ";
	//	}
	//	std::cout << "\n";
	//}
}