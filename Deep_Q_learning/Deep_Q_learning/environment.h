#pragma once
#include <vector>

class Environment {
public:
	// 현재 상태를 저장하고 있는 멤버 변수
	std::vector<int> current_state;


public:
	void processInput(const int& action) {
		// do something to respond user input
	}

	void update(const float& dt, float& reward, int& flag) {
		// update current_state with dt change
		// reward = high number when player chose good action
		// reward = low number when player chose bad action
		// flag = terminal flag : check wheter game has ended ro not
	}

	std::vector<int>& getCrurrentState() {
		return current_state;
	}

	void rendering() {

	}
};