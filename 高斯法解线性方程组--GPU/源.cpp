#include <iostream>

#include <fstream>
#include <sstream>
#include <CL/cl.h>
using namespace std;


const int ARRAY_SIZE = 10000;

//一、 选择OpenCL平台并创建一个上下文
cl_context CreateContext()

{

	cl_int errNum;

	cl_uint numPlatforms;

	cl_platform_id firstPlatformId;

	cl_context context = NULL;



	//选择可用的平台中的第一个

	errNum = clGetPlatformIDs(1, &firstPlatformId, &numPlatforms);

	if (errNum != CL_SUCCESS || numPlatforms <= 0)

	{

		std::cerr << "Failed to find any OpenCL platforms." << std::endl;

		return NULL;

	}



	//创建一个OpenCL上下文环境

	cl_context_properties contextProperties[] =

	{

		CL_CONTEXT_PLATFORM,

		(cl_context_properties)firstPlatformId,

		0

	};

	context = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_GPU,

		NULL, NULL, &errNum);



	return context;

}





//二、 创建设备并创建命令队列

cl_command_queue CreateCommandQueue(cl_context context, cl_device_id *device)

{

	cl_int errNum;

	cl_device_id *devices;

	cl_command_queue commandQueue = NULL;

	size_t deviceBufferSize = -1;



	// 获取设备缓冲区大小

	errNum = clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &deviceBufferSize);



	if (deviceBufferSize <= 0)

	{

		std::cerr << "No devices available.";

		return NULL;

	}



	// 为设备分配缓存空间

	devices = new cl_device_id[deviceBufferSize / sizeof(cl_device_id)];

	errNum = clGetContextInfo(context, CL_CONTEXT_DEVICES, deviceBufferSize, devices, NULL);



	//选取可用设备中的第一个

	commandQueue = clCreateCommandQueue(context, devices[0], 0, NULL);



	*device = devices[0];

	delete[] devices;

	return commandQueue;

}





// 三、创建和构建程序对象

cl_program CreateProgram(cl_context context, cl_device_id device, const char* fileName)

{

	cl_int errNum;

	cl_program program;



	std::ifstream kernelFile(fileName, std::ios::in);

	if (!kernelFile.is_open())

	{

		std::cerr << "Failed to open file for reading: " << fileName << std::endl;

		return NULL;

	}



	std::ostringstream oss;

	oss << kernelFile.rdbuf();



	std::string srcStdStr = oss.str();

	const char *srcStr = srcStdStr.c_str();

	program = clCreateProgramWithSource(context, 1,

		(const char**)&srcStr,

		NULL, NULL);



	errNum = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);



	return program;

}



//创建和构建程序对象

bool CreateMemObjects(cl_context context, cl_mem memObjects[3])

{

	memObjects[0] = clCreateBuffer(context, CL_MEM_READ_WRITE,

		sizeof(float)* 9, NULL, NULL);

	memObjects[1] = clCreateBuffer(context, CL_MEM_READ_WRITE,

		sizeof(float)* 3, NULL, NULL);

	memObjects[2] = clCreateBuffer(context, CL_MEM_READ_WRITE,

		sizeof(float)* 3, NULL, NULL);

	return true;

}





// 释放OpenCL资源

void Cleanup(cl_context context, cl_command_queue commandQueue,

	cl_program program, cl_kernel kernel, cl_mem memObjects[3])

{

	for (int i = 0; i < 3; i++)

	{

		if (memObjects[i] != 0)

			clReleaseMemObject(memObjects[i]);

	}

	if (commandQueue != 0)

		clReleaseCommandQueue(commandQueue);



	if (kernel != 0)

		clReleaseKernel(kernel);



	if (program != 0)

		clReleaseProgram(program);



	if (context != 0)

		clReleaseContext(context);
	return;
}
void gpu(cl_context context,

	cl_command_queue commandQueue,

	cl_program program,

	cl_device_id device,

	cl_kernel kernel1,

	cl_kernel kernel2,

	cl_mem memObjects[3],

	size_t globalWorkSize1[2],

	size_t globalWorkSize2[1],

	size_t localWorkSize[1],

	float *a, float *b, float *result)
{

	cl_int errNum;

	errNum = clEnqueueNDRangeKernel(commandQueue, kernel1, 2, NULL,

		globalWorkSize1, NULL,

		0, NULL, NULL);


	errNum = clEnqueueNDRangeKernel(commandQueue, kernel2, 1, NULL,

		globalWorkSize2, localWorkSize,

		0, NULL, NULL);

	// 六、 读取执行结果并释放OpenCL资源

	errNum = clEnqueueReadBuffer(commandQueue, memObjects[2], CL_TRUE,

		0, 3 * sizeof(float), result,

		0, NULL, NULL);

	for (int i = 0; i <3; i++)

	{

		std::cout << result[i] << " ";

	}

	std::cout << std::endl;

	std::cout << "Executed program succesfully." << std::endl;

}

int main(int argc, char** argv)

{
	cl_context context = 0;

	cl_command_queue commandQueue = 0;

	cl_program program = 0;

	cl_device_id device = 0;

	cl_kernel kernel1 = 0;

	cl_kernel kernel2 = 0;

	cl_mem memObjects[3] = { 0, 0, 0 };

	cl_int errNum;

	// 一、选择OpenCL平台并创建一个上下文

	context = CreateContext();

	// 二、 创建设备并创建命令队列

	commandQueue = CreateCommandQueue(context, &device);

	CreateMemObjects(context, memObjects);

	//创建和构建程序对象

	program = CreateProgram(context, device, "kernel.cl");

	// 四、 创建OpenCL内核并分配内存空间
	int n = 3;

	kernel1 = clCreateKernel(program, "touptriangle", NULL);

	errNum = clSetKernelArg(kernel1, 0, sizeof(cl_mem), (void*)(&n));

	errNum |= clSetKernelArg(kernel1, 1, sizeof(cl_mem), &memObjects[0]);

	errNum |= clSetKernelArg(kernel1, 2, sizeof(cl_mem), &memObjects[1]);

	kernel2 = clCreateKernel(program, "gauss_solve", NULL);

	errNum = clSetKernelArg(kernel2, 0, sizeof(cl_mem), (void*)(&n));

	errNum |= clSetKernelArg(kernel2, 1, sizeof(cl_mem), &memObjects[0]);

	errNum |= clSetKernelArg(kernel2, 2, sizeof(cl_mem), &memObjects[1]);

	errNum |= clSetKernelArg(kernel2, 3, sizeof(cl_mem), &memObjects[2]);

	//创建内存对象
	float A[] = { 2, 2, -1,
		1, -2, 4,
		5, 8, -1 };

	float x[] = { 0, 0, 0 };

	float b[] = { 6, 3, 27 };

	size_t globalWorkSize1[2] = { n, n };

	size_t globalWorkSize2[1] = { n };

	size_t localWorkSize[1] = { 1 };

	clEnqueueWriteBuffer(commandQueue, memObjects[0], CL_TRUE,

		0, 9 * sizeof(float), A,

		0, NULL, NULL);
	clEnqueueWriteBuffer(commandQueue, memObjects[1], CL_TRUE,

		0, 3 * sizeof(float), b,

		0, NULL, NULL);
	gpu(context, commandQueue, program, device, kernel1, kernel2, memObjects, globalWorkSize1, globalWorkSize2, localWorkSize, A, b, x);

	return 0;
}

