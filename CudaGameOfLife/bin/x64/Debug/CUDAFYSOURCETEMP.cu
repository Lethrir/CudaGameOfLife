
// CudaGameOfLife.Program
extern "C" __global__ void RunGeneration( bool* cells1, int cells1Len0, int cells1Len1,  bool* cells2, int cells2Len0, int cells2Len1);
// CudaGameOfLife.Program
__device__ bool GetTL(int x, int y,  bool* cells, int cellsLen0, int cellsLen1);
// CudaGameOfLife.Program
__device__ bool GetTC(int x, int y,  bool* cells, int cellsLen0, int cellsLen1);
// CudaGameOfLife.Program
__device__ bool GetTR(int x, int y,  bool* cells, int cellsLen0, int cellsLen1);
// CudaGameOfLife.Program
__device__ bool GetL(int x, int y,  bool* cells, int cellsLen0, int cellsLen1);
// CudaGameOfLife.Program
__device__ bool GetR(int x, int y,  bool* cells, int cellsLen0, int cellsLen1);
// CudaGameOfLife.Program
__device__ bool GetBL(int x, int y,  bool* cells, int cellsLen0, int cellsLen1);
// CudaGameOfLife.Program
__device__ bool GetBC(int x, int y,  bool* cells, int cellsLen0, int cellsLen1);
// CudaGameOfLife.Program
__device__ bool GetBR(int x, int y,  bool* cells, int cellsLen0, int cellsLen1);

// CudaGameOfLife.Program
extern "C" __global__ void RunGeneration( bool* cells1, int cells1Len0, int cells1Len1,  bool* cells2, int cells2Len0, int cells2Len1)
{
	int x = blockIdx.x;
	int y = blockIdx.y;
	int num = 0;
	if (GetTL(x, y, cells1, cells1Len0, cells1Len1))
	{
		num++;
	}
	if (GetTC(x, y, cells1, cells1Len0, cells1Len1))
	{
		num++;
	}
	if (GetTR(x, y, cells1, cells1Len0, cells1Len1))
	{
		num++;
	}
	if (GetL(x, y, cells1, cells1Len0, cells1Len1))
	{
		num++;
	}
	if (GetR(x, y, cells1, cells1Len0, cells1Len1))
	{
		num++;
	}
	if (GetBL(x, y, cells1, cells1Len0, cells1Len1))
	{
		num++;
	}
	if (GetBC(x, y, cells1, cells1Len0, cells1Len1))
	{
		num++;
	}
	if (GetBR(x, y, cells1, cells1Len0, cells1Len1))
	{
		num++;
	}
	cells2[(x) * cells2Len1 + ( y)] = (num == 3 || (cells1[(x) * cells1Len1 + ( y)] && num == 2));
}
// CudaGameOfLife.Program
__device__ bool GetTL(int x, int y,  bool* cells, int cellsLen0, int cellsLen1)
{
	if (x == 0)
	{
		x = 19;
	}
	else
	{
		x--;
	}
	if (y == 0)
	{
		y = 19;
	}
	else
	{
		y--;
	}
	return cells[(x) * cellsLen1 + ( y)];
}
// CudaGameOfLife.Program
__device__ bool GetTC(int x, int y,  bool* cells, int cellsLen0, int cellsLen1)
{
	if (y == 0)
	{
		y = 19;
	}
	else
	{
		y--;
	}
	return cells[(x) * cellsLen1 + ( y)];
}
// CudaGameOfLife.Program
__device__ bool GetTR(int x, int y,  bool* cells, int cellsLen0, int cellsLen1)
{
	if (x == 19)
	{
		x = 0;
	}
	else
	{
		x++;
	}
	if (y == 0)
	{
		y = 19;
	}
	else
	{
		y--;
	}
	return cells[(x) * cellsLen1 + ( y)];
}
// CudaGameOfLife.Program
__device__ bool GetL(int x, int y,  bool* cells, int cellsLen0, int cellsLen1)
{
	if (x == 0)
	{
		x = 19;
	}
	else
	{
		x--;
	}
	return cells[(x) * cellsLen1 + ( y)];
}
// CudaGameOfLife.Program
__device__ bool GetR(int x, int y,  bool* cells, int cellsLen0, int cellsLen1)
{
	if (x == 19)
	{
		x = 0;
	}
	else
	{
		x++;
	}
	return cells[(x) * cellsLen1 + ( y)];
}
// CudaGameOfLife.Program
__device__ bool GetBL(int x, int y,  bool* cells, int cellsLen0, int cellsLen1)
{
	if (x == 0)
	{
		x = 19;
	}
	else
	{
		x--;
	}
	if (y == 19)
	{
		y = 0;
	}
	else
	{
		y++;
	}
	return cells[(x) * cellsLen1 + ( y)];
}
// CudaGameOfLife.Program
__device__ bool GetBC(int x, int y,  bool* cells, int cellsLen0, int cellsLen1)
{
	if (y == 19)
	{
		y = 0;
	}
	else
	{
		y++;
	}
	return cells[(x) * cellsLen1 + ( y)];
}
// CudaGameOfLife.Program
__device__ bool GetBR(int x, int y,  bool* cells, int cellsLen0, int cellsLen1)
{
	if (x == 19)
	{
		x = 0;
	}
	else
	{
		x++;
	}
	if (y == 19)
	{
		y = 0;
	}
	else
	{
		y++;
	}
	return cells[(x) * cellsLen1 + ( y)];
}
