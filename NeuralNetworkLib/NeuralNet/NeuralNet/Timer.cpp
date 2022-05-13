#include "pch.h"
#include "Timer.h"

CTimer::CTimer()
{

}

void CTimer::StartCheck()
{
	startPoint = std::chrono::system_clock::now();
}

void CTimer::EndCheck()
{
	endPoint = std::chrono::system_clock::now();
}

double CTimer::GetElapsedTime()
{
	std::chrono::duration<double> sec = endPoint - startPoint;
	return sec.count();
}

void CTimer::PrintElapsedTime(std::string text)
{
	std::cout << text << " : " << GetElapsedTime() << "\n";
}