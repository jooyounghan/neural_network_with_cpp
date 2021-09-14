#pragma once
#include <assert.h>
class GridCell {
private:
	int direction[4]{ 0 };
	// 해당 방향으로 이동할 때의 이득 값
	int profit;
	// 해당 셀에 도착하였을 때 얻게 되는 가중치


public:
	GridCell() {};
	

	/*
	TODO
	
	update : 어떤 방향으로 이동하였을 때, (그 셀을 통하여 얻을 수 있는 가중치 + 해당 셀에서의 최대 이득)을
	계산하여 이를 이동 전 셀의 이동방향에 업데이트 해준다.

	maxprofit : 네 방향 중 최대 이득을 return

	*/
	void update() {}


};