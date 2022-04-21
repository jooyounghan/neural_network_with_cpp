#pragma once
#include <stdarg.h>
#include <iostream>
#include <algorithm>
#include <math.h>
#include <memory>
#include <thread>
#include <future>
#include <chrono>

#include <string>
#include <stack>
#include <queue>
#include <map>
#include <unordered_map>
#include <set>

#include "MacroPart.h"

using BYTE = unsigned char;
using int8 = __int8;
using int16 = __int16;
using int32 = __int32;
using int64 = __int64;
using uint8 = unsigned __int8;
using uint16 = unsigned __int16;
using uint32 = unsigned __int32;
using uint64 = unsigned __int64;

using timepoint = std::chrono::system_clock::time_point;

struct LayerInfo
{
	LayerInfo(const uint32& rowIn, const uint32& colIn, const uint32& actIDIn) : row(rowIn), col(colIn), actID(actIDIn) {};

	uint32 row;
	uint32 col;
	uint16 actID;
};
