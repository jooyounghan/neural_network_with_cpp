#pragma once

/* ------------------------------------------------------ */
#pragma region Parallel
#define PARALLEL
#define THREADNUM		(std::thread::hardware_concurrency() / 2)

#define WAITTHREADVECTOR(workThreadVector)						\
																\
for (uint32 threadNum = 0; threadNum < THREADNUM; ++threadNum)	\
{																\
	workThreadVector[threadNum].wait();							\
}																\
/* ------------------------------------------------------ */

/* ------------------------------------------------------ */
#pragma region ActivationFunctionID
#define SIGMOID		1
#define RELU		2
#define IDENTITY	3
#define	SUMATION	4
#define SOFTMAX		5  
#pragma endregion
/* ------------------------------------------------------ */

/* ------------------------------------------------------ */
#pragma region InitializeID
#define NORMAL		1
#define XAVIER		2
#define HE			3  
#pragma endregion
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