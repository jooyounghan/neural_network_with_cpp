#pragma once

template<typename T>
class CTimer
{
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

	T GetElapsedTime();

public:


};

