#include <pch.h>
#include "pch.h"
int main()
{
	CMatrix a = CMatrix(4, 7);
	a.XavierNormalInitialize(1, 1);
	a.PrintData();
	std::cout << "\n";
	CTimer timer;
	timer.StartCheck();
	a.Transpose();
	timer.EndCheck();
	timer.PrintElapsedTime("async");
	a.PrintData();
	std::cout << "\n";
#ifdef ASYNC
	std::cout << " async " << std::endl;
#endif

}