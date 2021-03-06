#pragma once
#include <random>
#include <iostream>
#include "GridCell.h"
#include "Agent.h"

class Grid {
public:
	int row, col;
	int size;
	std::vector<std::pair<int, int>> directions{ {1, 0}, {0, -1}, {-1, 0}, {0, 1} };
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
	void random_move() {
		int random_access = rand() % 4;
		std::pair<int, int>* dir = &(directions[random_access]);
		while (agent->row + dir->first < 0 || agent->row + dir->first >= row || agent->col + dir->second < 0 || agent->col + dir->second >= col) {
			random_access = (random_access + 1) % 4;
			dir = &directions[random_access];
		}
		GridCell& cell_now = get_cell(agent->row, agent->col);
		agent->row += dir->first;
		agent->col += dir->second;
		GridCell& cell_after = get_cell(agent->row, agent->col);
		cell_now.update(random_access, &cell_after);

		if (cell_after.goal || cell_after.obstacle) {
			agent->row = 0;
			agent->col = 0;
		}
	}

	void move(const int& access) {
		std::pair<int, int>& dir = directions[access];
		if (agent->row + dir.first < 0 || agent->row + dir.first >= row || agent->col + dir.second < 0 || agent->col + dir.second >= col) {
			std::cout << "can't move(out of boundary)" << std::endl;
			return;
		}

		GridCell& cell_now = get_cell(agent->row, agent->col);
		agent->row += dir.first;
		agent->col += dir.second;
		GridCell& cell_after = get_cell(agent->row, agent->col);
		cell_now.update(access, &cell_after);

		if (cell_after.goal || cell_after.obstacle) {
			agent->row = 0;
			agent->col = 0;
		}
	}

	void train(const int& iterations) {
		std::cout << "train started" << std::endl;
		for (int i = 0; i < iterations; ++i) {
			if (!((i + 1) % (iterations / 10))) {
				std::cout << i + 1 << " iterations trained..." << "\n";
			}
			random_move();
		}
		std::cout << "train finished" << std::endl;
	}

	// row, col에 해당하는 Cell이 목적지인지를 설정하는 함수이다.
	void set_goal(const int& row_in, const int& col_in) {
		GridCell& cell_now = get_cell(row_in, col_in);
		cell_now.goal = true;
		for (int i = 0; i < 4; ++i) {
			cell_now.direction[i] = 1.0;
		}
	}

	// row, col에 해당하는 Cell이 장애물인지를 설정하는 함수이다.
	void set_obstacle(const int& row_in, const int& col_in) {
		GridCell& cell_now = get_cell(row_in, col_in);
		cell_now.obstacle = true;
		if (cell_now.goal) cell_now.goal = false;
		for (int i = 0; i < 4; ++i) {
			cell_now.direction[i] = -1.0;
		}
	}

	// row, col에 해당하는 Cell의 이득값을 설정하는 함수이다.
	void set_profit(const int& row_in, const int& col_in, const float& profit_in) {
		GridCell& cell_now = get_cell(row_in, col_in);
		cell_now.profit = profit_in;
	}

	void initialize_profit(const float& profit_in) {
		for (int i = 0; i < size; ++i) {
			gridmap[i].profit = profit_in;
		}
	}

	void go() {
		agent->row = 0;
		agent->col = 0;

		int cnt = 1;

		while (true) {
			GridCell& cell_now = get_cell(agent->row, agent->col);			
			std::pair<int, int>& dir = directions[cell_now.max_profit_dir()];
			if (agent->row + dir.first < 0 || agent->row + dir.first >= row || agent->col + dir.second < 0 || agent->col + dir.second >= col) {
				std::cout << "Training Result has problem(it trained to move out of the boundary" << std::endl;
				return;
			}
			agent->row = agent->row + dir.first;
			agent->col = agent->col + dir.second;

			std::cout << "moved to the (" << agent->row << ", " << agent->col << ")" << "\n";
			GridCell& cell_after = get_cell(agent->row, agent->col);
			if (cell_after.goal) {
				std::cout << "arrived!" << "\n";
				std::cout << "move times : " << cnt << "\n";
				break;
			}
			cnt++;
		}
	}

	~Grid(){
		if (gridmap != nullptr) {
			delete[] gridmap;
		}
	}
};