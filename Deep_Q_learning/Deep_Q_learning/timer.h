#pragma once
#include <chrono>
#include <iostream>

class timer
{
private:
	std::chrono::system_clock::time_point start;
	std::chrono::system_clock::time_point end;
public:
	void checkStart()
	{
		start = std::chrono::system_clock::now();
	}
	void checkEnd()
	{
		end = std::chrono::system_clock::now();
	}
	void elasped(const char* text)
	{
		std::cout << text << " : " << std::chrono::duration<double>(end - start).count() << "\n";
	}
	void checkTime()
	{
		std::cout << std::chrono::duration<double>(std::chrono::system_clock::now() - start).count() << "\n";
	}

	std::chrono::system_clock::rep getElasped()
	{
		return (end - start).count();
	}

};