#pragma OPENCL EXTENSION cl_amd_printf : enable
__kernel void touptriangle(int n,__global float *A, __global float* b)
{
for(int k=0;k<n-1;k++)
{
int i=get_global_id(0);
int j=get_global_id(1);
float c=A[i*n + k] / A[k*n + k];

if(i>k)
	{
	A[i*n + j] = A[i*n + j] - c * A[k*n + j];
	
	b[i] = b[i] - c * b[k];
	
	}
}
}
__kernel void gauss_solve(int n, __global float*A, __global float* b, __global float* x)

{

	for (int i = n - 1; i >= 0; i--)
	{
		x[i] = b[i] / A[i*n + i];
		int j = get_global_id(0);
	
		if (j < i)
		{
			b[j] = b[j] - A[j*n + i] * x[i];
			A[j*n + i] = 0;
		}
	}
}


