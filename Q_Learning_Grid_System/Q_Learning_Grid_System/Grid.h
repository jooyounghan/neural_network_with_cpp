#pragma once
#include <random>
#include "GridCell.h"
#include "Agent.h"

class Grid {
public:
	int row, col;
	int size;
	Agent* agent = nullptr;
	GridCell* gridmap = nullptr;

public:
	Grid(const int& row_in, const int& col_in)
		: row(row_in), col(col_in), size(row_in * col_in) {
		gridmap = new GridCell[size];
	}


	//Grid 시스템에 Agent를 추가하는 함수
	void set_agent(Agent* agent_in) {
		agent = agent_in;
	}


	//row, col에 해당하는 Cell를 참조변수로 반환해주는 함수
	GridCell& get_cell(const int& row_in, const int& col_in) {
		return gridmap[row_in * col + col_in];
	}

	//Agent를 랜덤으로 움직이는 함수이다.
	void random_move(std::vector<std::pair<int, int>>& directions) {
		int random_access = rand() % 4;
		std::pair<int, int>& dir = directions[random_access];
		while (agent->row + dir.first < 0 || agent->row + dir.first >= row || agent->col + dir.second < 0 || agent->col + dir.second >= col) {
			random_access = (random_access + 1) % 4;
			dir = directions[random_access];
		}
		
		GridCell& cell_now = get_cell(agent->row, agent->col);
		agent->row += dir.first;
		agent->col += dir.second;
		GridCell& cell_after = get_cell(agent->row, agent->col);
		cell_now.update(random_access, &cell_after);

		if (cell_after.goal == true) {
			agent->row = 0;
			agent->col = 0;
		}
	}

	// row, col에 해당하는 Cell이 목적지인지를 설정하는 함수이다.
	void set_goal(const int& row_in, const int& col_in) {
		GridCell& cell_now = get_cell(row_in, col_in);
		cell_now.goal = true;
	}

	// row, col에 해당하는 Cell의 이득값을 설정하는 함수이다.
	void set_profit(const int& row_in, const int& col_in, const float& profit_in) {
		GridCell& cell_now = get_cell(row_in, col_in);
		cell_now.profit = profit_in;
	}

	~Grid(){
		if (gridmap != nullptr) {
			delete[] gridmap;
		}
	}
};