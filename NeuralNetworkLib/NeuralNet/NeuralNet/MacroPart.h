#pragma once

/* ------------------------------------------------------ */
#pragma region Parallel
#define THREADNUM		(std::thread::hardware_concurrency() / 2)
#define LOCKGUARD(mtx)	std::lock_guard<std::mutex>{ mtx };
#define PARALLELLIMIT	1E4

#define PARALLELBRANCH(conditionvar, parallel, serial)				\
if (conditionvar > PARALLELLIMIT) { return parallel; }				\
else { return serial; }												\


#define WAITTHREADVECTOR(workThreadVector)							\
for (uint32 threadNum = 0; threadNum < THREADNUM; ++threadNum)		\
{																	\
	workThreadVector[threadNum].wait();								\
}																	\

/* ------------------------------------------------------ */

/* ------------------------------------------------------ */
#pragma region Crash
#define CRASH(cause)						\
{											\
	uint32*	crash = nullptr;				\
	__analysis_assume(crash != nullptr);	\
	*crash = 0xDEADBEEF;					\
}											\

#define ASSERT_CRASH(expr)					\
{											\
	if(!(expr)){							\
		CRASH("ASSERT_CRASH");				\
		__analysis_assume(expr);			\
	}										\
}
#pragma endregion
/* ------------------------------------------------------ */


/* ------------------------------------------------------ */
#pragma region Delete
#define DELETEPTR(ptr)						\
{											\
	if (ptr != nullptr)						\
	{										\
		delete ptr;							\
		ptr = nullptr;						\
	}										\
}											\

#define DELETEARRPTR(ptr)					\
{											\
	if (ptr != nullptr)						\
	{										\
		delete[] ptr;						\
		ptr = nullptr;						\
	}										\
}											\

/* ------------------------------------------------------ */