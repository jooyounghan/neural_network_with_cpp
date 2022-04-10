#include "pch.h"
#include "Timer.h"

template<typename T>
void CTimer<T>::StartCheck()
{
	startPoint = std::chrono::system_clock::now();
}

template<typename T>
void CTimer<T>::EndCheck()
{
	endPoint = std::chrono::system_clock::now();
}

template<typename T>
T CTimer<T>::GetElapsedTime()
{
	std::chrono::duration<T> sec = endPoint - startPoint;
	return sec.count();
}
