#pragma once
#include <assert.h>
class GridCell {
public:
	// 해당 방향으로 이동할 때의 이득 값을 나타내는 변수
	float direction[4]{ 0 };
	// 해당 셀에 도착하였을 때 얻게 되는 가중치를 나타내는 변수
	float profit;
	// 해당 셀이 목적지인지를 나타내는 변수
	bool goal;
	// 해당 셀이 장애물인지를 나타내는 변수
	bool obstacle;


public:
	GridCell() : profit(-0.1), goal(false), obstacle(false) {};

	/*
	dir = 0(우), 1(하), 2(좌), 3(상)
	어떤 방향으로 이동하였을 때, (그 셀을 통하여 얻을 수 있는 가중치 + 해당 셀에서의 최대 이득)을
	계산하여 이를 이동 전 셀의 이동방향에 업데이트 해준다.
	*/
	void update(const int& dir, GridCell* moved_cell) {
		direction[dir] = moved_cell->max_profit() + moved_cell->profit;
	}

	// 네 방향 중 최대 이득을 반환한다.
	float max_profit() {
		float temp_max = direction[0];
		for (int i = 0; i < 4; ++i) {
			if (temp_max < direction[i]) {
				temp_max = direction[i];
			}
		}
		return temp_max;
	}

	int max_profit_dir() {
		float temp_max = direction[0];
		int temp_dir = 0;
		for (int i = 1; i < 4; ++i) {
			if (temp_max < direction[i]) {
				temp_max = direction[i];
				temp_dir = i;
			}
		}
		return temp_dir;
	}
};