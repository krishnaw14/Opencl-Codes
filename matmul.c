#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <OpenCL/opencl.h>
#include <unistd.h>

#define ORDER 300

char * kernelsource = "__kernel void matmul(                            \n" \
"   const int N,                                                        \n" \
"   __global float* A,                                                  \n" \
"   __global float* B,                                                  \n" \
"   __global float* C)                                                  \n" \
"{                                                                      \n" \
"   int i = get_global_id(0);                                           \n" \
"   int j = get_global_id(1);                                           \n" \
"   int k;                                                              \n" \
"   float tmp;                                                          \n" \
"   if( (i < N) && (j<N) )                                              \n" \
"   {                                                                   \n" \
"       tmp=0.0f;                                                       \n" \
"       for(k=0;k<N;k++)                                                \n" \
"           tmp+=A[i*N+k]*B[N*k+j];                                     \n" \
"       C[i*N+j]=tmp ;                                                  \n" \
"   }                                                                   \n" \
"}                                                                     \n" \
"\n";

int main(int argc, char** argv)
{
	float *h_A;
	float *h_B;
	float* h_C;
	float* h_D;
	int N;
	int size, i,j;

	cl_mem d_a, d_b, d_c;

	cl_int err;
	cl_device_id device;
	cl_context context;
	cl_command_queue commands;
	cl_program program;
	cl_kernel kernel;
	cl_uint numPlatforms;

	N = ORDER;
	size = N*N;

    h_A = (float *)malloc(size * sizeof(float));
    h_B = (float *)malloc(size * sizeof(float));
    h_C = (float *)malloc(size * sizeof(float));
    h_D = (float *)malloc(size * sizeof(float));

    err = clGetPlatformIDs(0, NULL, &numPlatforms);

	if(err!=CL_SUCCESS)
	{
		printf("Cannot find any opencl platforms\n");
		exit(0);
	}
	if(numPlatforms==0)
	{
		printf("No opencl platforms available \n");
		exit(0);
	}

	cl_platform_id Platform[numPlatforms];
	err = clGetPlatformIDs(numPlatforms, Platform, NULL);

	for(i=0;i<numPlatforms;i++)
	{
		err = clGetDeviceIDs(Platform[i], CL_DEVICE_TYPE_GPU, 1, &device, NULL);
		if(err == CL_SUCCESS)
			break;
	}
	if (device == NULL)
	{
		printf("Error in finding a GPU platform \n");
		exit(0);
	}

	context = clCreateContext(0,1,&device, NULL, NULL, &err);
	if(err != CL_SUCCESS)
	{
		printf("Unable to create context for the device \n");
		exit(0);
	}

	commands = clCreateCommandQueue(context, device, 0, &err);
	if(err != CL_SUCCESS)
	{
		printf("Unable to create a command queue \n");
		exit(0);
	}	

	//Initialize A, B and C
	for(i=0; i<N; i++)
		for(j=0;j<N;j++)
			h_A[i*N+j]=1;

	for(i=0; i<N; i++)
		for(j=0;j<N;j++)
			h_B[i*N+j]=3;

	for (i=0; i<N; i++)
		for (j = 0; j < N; j++)
			h_C[i*N+j] = 0.0f;

	d_a = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * size, h_A, &err);
	if(err != CL_SUCCESS)
	{
		printf("Unable to create buffer d_a \n");
		exit(0);
	}
	d_b = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * size, h_B, &err);
	if(err != CL_SUCCESS)
	{
		printf("Unable to create buffer d_b \n");
		exit(0);
	}
	d_c = clCreateBuffer(context, CL_MEM_WRITE_ONLY , sizeof(float) * size, NULL, &err);	
	if(err != CL_SUCCESS)
	{
		printf("Unable to create buffer d_c \n");
		exit(0);
	}

	program = clCreateProgramWithSource(context, 1, (const char **) & kernelsource, NULL, &err);
	if(err != CL_SUCCESS)
	{
		printf("Unable to create create program with source \n");
		exit(0);
	}	

	err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	if(err!= CL_SUCCESS)
	{
		printf("Not able to build program\n");
	}

	kernel = clCreateKernel(program, "matmul", &err);
	if(err != CL_SUCCESS)
	{
		printf("Unable to create kernel \n");
	}

	err = clSetKernelArg(kernel, 0, sizeof(int), &N);
	err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_a);
	err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_b);
	err = clSetKernelArg(kernel, 3, sizeof(cl_mem), &d_c);

	if (err != CL_SUCCESS)
	{
		printf("Unable to set the arguments of the kernel");
	}

	size_t global[2]={N,N};

	err = clEnqueueNDRangeKernel(commands, kernel, 2, NULL, global, NULL, 0, NULL, NULL);
	if(err != CL_SUCCESS)
	{
		printf("Unable to execute kernel \n");
	}

	err = clFinish(commands);

    err = clEnqueueReadBuffer(commands, d_c, CL_TRUE, 0, sizeof(float) * size, h_C, 0, NULL, NULL);	

    printf("\nAll commands executed. Now check the error with respect to serial code\n");

    float tmp,k;

    for(i=0; i<N; i++)
    {
    	for(j=0; j<N; j++)
    	{
    		printf("%f ", h_C[i*N+j]);
    	}
    	printf("\n");
    }


    free(h_A);
    free(h_B);
    free(h_C);
    clReleaseMemObject(d_a);
    clReleaseMemObject(d_b);
    clReleaseMemObject(d_c);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(commands);
    clReleaseContext(context);	


    return EXIT_SUCCESS;
}