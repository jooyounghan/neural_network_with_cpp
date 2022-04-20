#pragma once

/* ------------------------------------------------------ */
#pragma region Parallel
#define PARALLEL
#define THREADNUM		(std::thread::hardware_concurrency() / 2)
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