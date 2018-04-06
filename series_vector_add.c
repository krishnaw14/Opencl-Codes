#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <OpenCl/opencl.h>


//Define Kernel
const char *KernelSource = "\n" \
"__kernel void vadd(                                                 \n" \
"   __global float* a,                                                  \n" \
"   __global float* b,                                                  \n" \
"   __global float* c,                                                  \n" \
"   const unsigned int count)                                           \n" \
"{                                                                      \n" \
"   int i = get_global_id(0);                                           \n" \
"   if(i < count)                                                       \n" \
"       c[i] = a[i] + b[i];                                             \n" \
"}                                                                      \n" \
"\n";

#define LENGTH 1024

int main(int argc, char** argv)
{
	cl_int err;

	float* h_a = (float*) calloc (LENGTH, sizeof(float));
	float* h_b = (float*) calloc (LENGTH, sizeof(float));
	float* h_c = (float*) calloc (LENGTH, sizeof(float));
	float* h_d = (float*) calloc (LENGTH, sizeof(float));
	float* h_e = (float*) calloc (LENGTH, sizeof(float));
	float* h_f = (float*) calloc (LENGTH, sizeof(float));
	float* h_g = (float*) calloc (LENGTH, sizeof(float));

	size_t global; //global domain size

	cl_device_id device_id;
	cl_context context;
	cl_command_queue commands;
	cl_program program;

	cl_kernel ko_vadd;

	cl_uint numPlatforms;

	cl_mem d_a; //device memory used for vector a
	cl_mem d_b;
	cl_mem d_c;
	cl_mem d_d;
	cl_mem d_e;
	cl_mem d_f;
	cl_mem d_g;


	int i=0;
	int count = LENGTH;

	for(i=0;i<count;i++)
	{
		h_a[i]=i;
		h_b[i]=2*i;
		h_e[i]=4*i;
		h_g[i]=8*i;
	}

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

	//Get a GPU
	for(i=0;i<numPlatforms;i++)
	{
		err = clGetDeviceIDs(Platform[i], CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
		if(err == CL_SUCCESS)
			break;
	}
	if (device_id == NULL)
	{
		printf("Error in finding a GPU platform \n");
		exit(0);
	}

	//Start with the main part

	context = clCreateContext(0,1,&device_id, NULL, NULL, &err);
	if(err != CL_SUCCESS)
	{
		printf("Unable to create context for the device \n");
		exit(0);
	}

	commands = clCreateCommandQueue(context, device_id, 0, &err);
	if(err != CL_SUCCESS)
	{
		printf("Unable to create a command queue \n");
		exit(0);
	}

	program = clCreateProgramWithSource(context, 1, (const char **) & KernelSource, NULL, &err);
	if(err != CL_SUCCESS)
	{
		printf("Unable to create a program \n");
		exit(0);		
	}

	err= clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	if (err != CL_SUCCESS)
    {
        size_t len;
        char buffer[2048];

        printf("Error: Failed to build program executable!\n");
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        printf("%s\n", buffer);
        return EXIT_FAILURE;
    }

    ko_vadd = clCreateKernel(program, "vadd", &err);
	if(err != CL_SUCCESS)
	{
		printf("Unable to create a kernel \n");
		exit(0);		
	}

	//check error after every step- I am skipping that for now
	d_a  = clCreateBuffer(context,  CL_MEM_READ_ONLY ,  sizeof(float) * count, NULL, &err);
    d_b  = clCreateBuffer(context,  CL_MEM_READ_ONLY ,  sizeof(float) * count, NULL, &err);
    d_c  = clCreateBuffer(context,  CL_MEM_READ_WRITE, sizeof(float) * count, NULL, &err);
    d_d  = clCreateBuffer(context,  CL_MEM_READ_WRITE, sizeof(float) * count, NULL, &err);
    d_e  = clCreateBuffer(context,  CL_MEM_READ_ONLY , sizeof(float) * count, NULL, &err);
    d_f  = clCreateBuffer(context,  CL_MEM_WRITE_ONLY, sizeof(float) * count, NULL, &err);
    d_g  = clCreateBuffer(context,  CL_MEM_READ_ONLY, sizeof(float) * count, NULL, &err);

    err = clEnqueueWriteBuffer(commands, d_a, CL_TRUE, 0, sizeof(float) * count, h_a, 0, NULL, NULL);
    err = clEnqueueWriteBuffer(commands, d_b, CL_TRUE, 0, sizeof(float) * count, h_b, 0, NULL, NULL);
    err = clEnqueueWriteBuffer(commands, d_e, CL_TRUE, 0, sizeof(float) * count, h_e, 0, NULL, NULL);
    err = clEnqueueWriteBuffer(commands, d_g, CL_TRUE, 0, sizeof(float) * count, h_g, 0, NULL, NULL);

    //C=A+B
    err = clSetKernelArg(ko_vadd, 0, sizeof(cl_mem), &d_a);
    err = clSetKernelArg(ko_vadd, 1, sizeof(cl_mem), &d_b);
    err = clSetKernelArg(ko_vadd, 2, sizeof(cl_mem), &d_c);
    err = clSetKernelArg(ko_vadd, 3, sizeof(unsigned int), &count);

    global=count;
    err = clEnqueueNDRangeKernel(commands, ko_vadd, 1, NULL, &global, NULL, 0, NULL, NULL);

    //D=C+E
    err = clSetKernelArg(ko_vadd, 0, sizeof(cl_mem), &d_c);
    err = clSetKernelArg(ko_vadd, 1, sizeof(cl_mem), &d_e);
    err = clSetKernelArg(ko_vadd, 2, sizeof(cl_mem), &d_d);
    err = clSetKernelArg(ko_vadd, 3, sizeof(unsigned int), &count);

    global=count;
    err = clEnqueueNDRangeKernel(commands, ko_vadd, 1, NULL, &global, NULL, 0, NULL, NULL);   

    //F=D+G
    err = clSetKernelArg(ko_vadd, 0, sizeof(cl_mem), &d_d);
    err = clSetKernelArg(ko_vadd, 1, sizeof(cl_mem), &d_g);
    err = clSetKernelArg(ko_vadd, 2, sizeof(cl_mem), &d_f);
    err = clSetKernelArg(ko_vadd, 3, sizeof(unsigned int), &count);

    global=count;
    err = clEnqueueNDRangeKernel(commands, ko_vadd, 1, NULL, &global, NULL, 0, NULL, NULL);    


    err = clFinish(commands);

    err = clEnqueueReadBuffer( commands, d_f, CL_TRUE, 0, sizeof(float) * count, h_f, 0, NULL, NULL );

    for(i=0;i<10;i++)
    	printf("%f \n", h_f[i]);

    clReleaseMemObject(d_a);
    clReleaseMemObject(d_b);
    clReleaseMemObject(d_c);
    clReleaseMemObject(d_d);
    clReleaseMemObject(d_e);
    clReleaseMemObject(d_f);
    clReleaseMemObject(d_g);
    clReleaseProgram(program);
    clReleaseKernel(ko_vadd);
    clReleaseCommandQueue(commands);
    clReleaseContext(context);

    free(h_a);
    free(h_b);
    free(h_c);
    free(h_d);
    free(h_e);
    free(h_f);
    free(h_g);


    return 0;
}