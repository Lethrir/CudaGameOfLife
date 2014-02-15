using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Cudafy;
using Cudafy.Host;
using Cudafy.Translator;

namespace CudaGameOfLife
{
    class Program
    {
        private const int gridSize = 1000;
        private const int Generations = 500;
        private static bool[,] _cells1 = new bool[gridSize, gridSize];
        private static bool[,] _cells2 = new bool[gridSize, gridSize];
        private static bool[,] _cells1Gpu;
        private static bool[,] _cells2Gpu;

        static void Main(string[] args)
        {

            CudafyTranslator.GenerateDebug = true;
            CudafyModule cm = CudafyTranslator.Cudafy();

            var gpu = CudafyHost.GetDevice(eGPUType.Cuda, CudafyModes.DeviceId);
            try
            {
                gpu.LoadModule(cm);
                RunGame(gpu);
            }
            finally
            {
                gpu.FreeAll();
            }
        }

        // Place a glider using x and y as bottom left corner
        private static void setupGlider(bool[,] cells, int x, int y)
        {
            cells[x, y] = true;
            cells[x + 1, y] = true;
            cells[x + 2, y] = true;
            cells[x + 2, y + 1] = true;
            cells[x + 1, y + 2] = true;
        }

        private static void TimeEvent(Action action, string name)
        {
            var start = DateTime.Now;
            action();
            var tt = DateTime.Now - start;
            Console.WriteLine("Time taken for {0}: {1}", name, tt.TotalMilliseconds);
        }

        private static void RunGame(GPGPU gpu)
        {
            setupGlider(_cells1, 10,10);

            TimeEvent(() => GpuGoL(gpu), "GPU GoL");
            TimeEvent(CpuGoL, "CPU GoL");

            Console.ReadKey();
        }

        private static void CpuGoL()
        {
            for (var i = 0; i < Generations; i++)
            {
                if (i%2 == 0)
                {
                    RunCpuGeneration(_cells1, _cells2);
                }
                else
                {
                    RunCpuGeneration(_cells2, _cells1);
                }
            }
        }

        private static void GpuGoL(GPGPU gpu)
        {
            _cells1Gpu = gpu.CopyToDevice(_cells1);
            _cells2Gpu = gpu.CopyToDevice(_cells2);

            for (var i = 0; i < Generations; i++)
            {
                if (i%2 == 0)
                {
                    gpu.Launch(new dim3(gridSize, gridSize), 1).RunGeneration(_cells1Gpu, _cells2Gpu);
                }
                else
                {
                    gpu.Launch(new dim3(gridSize, gridSize), 1).RunGeneration(_cells2Gpu, _cells1Gpu);
                }
            }

            gpu.CopyFromDevice(_cells1Gpu, _cells1);
            gpu.CopyFromDevice(_cells2Gpu, _cells2);
        }

        private static void RunCpuGeneration(bool[,] cells1, bool[,] cells2)
        {
            for (var x = 0; x < gridSize; x++)
            {
                for (var y = 0; y < gridSize; y++)
                {
                    cells2[x, y] = CheckCell(cells1, x, y);
                }
            }
        }

        [Cudafy]
        private static void RunGeneration(GThread thread, bool[,] cells1, bool[,] cells2)
        {
            // Get block location
            var x = thread.blockIdx.x;
            var y = thread.blockIdx.y;

            cells2[x, y] = CheckCell(cells1, x, y);
        }

        [Cudafy]
        private static bool CheckCell(bool[,] cells1, int x, int y)
        {
            var i = GetSurroundingCount(cells1, x, y);

            return IsAlive(cells1, i, x, y);
        }

        [Cudafy]
        private static bool IsAlive(bool[,] cells1, int i, int x, int y)
        {
            return (i == 3 || (cells1[x, y] && i == 2));
        }

        [Cudafy]
        private static int GetSurroundingCount(bool[,] cells1, int x, int y)
        {
            var i = 0;
            if (GetTL(x, y, cells1))
            {
                i++;
            }
            if (GetTC(x, y, cells1))
            {
                i++;
            }
            if (GetTR(x, y, cells1))
            {
                i++;
            }
            if (GetL(x, y, cells1))
            {
                i++;
            }
            if (GetR(x, y, cells1))
            {
                i++;
            }
            if (GetBL(x, y, cells1))
            {
                i++;
            }
            if (GetBC(x, y, cells1))
            {
                i++;
            }
            if (GetBR(x, y, cells1))
            {
                i++;
            }
            return i;
        }

        [Cudafy]
        private static bool GetTL(int x, int y, bool[,] cells)
        {
            if (x == 0)
            {
                x = gridSize - 1;
            }
            else
            {
                x--;
            }
            if (y == 0)
            {
                y = gridSize - 1;
            }
            else
            {
                y--;
            }

            return cells[x, y];
        }

        [Cudafy]
        private static bool GetTC(int x, int y, bool[,] cells)
        {
            if (y == 0)
            {
                y = gridSize - 1;
            }
            else
            {
                y--;
            }

            return cells[x, y];
        }

        [Cudafy]
        private static bool GetTR(int x, int y, bool[,] cells)
        {
            if (x == (gridSize - 1))
            {
                x = 0;
            }
            else
            {
                x++;
            }
            if (y == 0)
            {
                y = gridSize - 1;
            }
            else
            {
                y--;
            }

            return cells[x, y];
        }

        [Cudafy]
        private static bool GetL(int x, int y, bool[,] cells)
        {
            if (x == 0)
            {
                x = gridSize - 1;
            }
            else
            {
                x--;
            }

            return cells[x, y];
        }

        [Cudafy]
        private static bool GetR(int x, int y, bool[,] cells)
        {
            if (x == (gridSize - 1))
            {
                x = 0;
            }
            else
            {
                x++;
            }

            return cells[x, y];
        }

        [Cudafy]
        private static bool GetBL(int x, int y, bool[,] cells)
        {
            if (x == 0)
            {
                x = gridSize - 1;
            }
            else
            {
                x--;
            }
            if (y == (gridSize - 1))
            {
                y = 0;
            }
            else
            {
                y++;
            }

            return cells[x, y];
        }

        [Cudafy]
        private static bool GetBC(int x, int y, bool[,] cells)
        {
            if (y == (gridSize - 1))
            {
                y = 0;
            }
            else
            {
                y++;
            }

            return cells[x, y];
        }

        [Cudafy]
        private static bool GetBR(int x, int y, bool[,] cells)
        {
            if (x == (gridSize - 1))
            {
                x = 0;
            }
            else
            {
                x++;
            }
            if (y == (gridSize - 1))
            {
                y = 0;
            }
            else
            {
                y++;
            }

            return cells[x, y];
        }
    }
}
