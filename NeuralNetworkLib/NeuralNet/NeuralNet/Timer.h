#pragma once

class CTimer
{
public:
	CTimer();

private:
	timepoint startPoint;
	timepoint endPoint;

private:
	std::thread timerThread;
	std::atomic<bool> timerFlag;
	std::vector<timepoint> timeVector;

public:
	void StartCheck();
	void EndCheck();
	double GetElapsedTime();
	void PrintElapsedTime(std::string text);
};

