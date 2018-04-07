#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <math.h>
#include <OpenCl/opencl.h>

#define pi 3.141592653589
#define MAX_SOURCE_SIZE (0x100000)

#define LENGTH 10

int main(int argc, char** argv)
{
	cl_int err;

	float* h_a = (float*) calloc (LENGTH, sizeof(float));
	float* h_c = (float*) calloc (LENGTH, sizeof(float));
	float* sum = (float*) calloc (1, sizeof(float));
	sum[0]=0;

	float h =pi/LENGTH; 

	size_t global; //global domain size

	cl_device_id device_id;
	cl_context context;
	cl_command_queue commands;
	cl_program program;

	cl_kernel ko_vadd;

	cl_uint numPlatforms;

	cl_mem d_a; //device memory used for vector a
	cl_mem d_c;
	cl_mem d_sum;

	//Loading the kernel
	FILE *fp;
	const char fileName[]="./kernel.cl";
	size_t source_size;
	char* source_str;
	cl_int i;

	fp = fopen(fileName, "r");
	if(!fp)
	{
		fprintf(stderr, "Failed to load kernel.\n");
		exit(1);
	}
	source_str = (char*) malloc(MAX_SOURCE_SIZE);
	source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
	fclose(fp);

	//int i=0;
	int count = LENGTH;

	for(i=0;i<count;i++)
	{
		h_a[i]=i*h;
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

	program = clCreateProgramWithSource(context, 1, (const char **) & source_str, NULL, &err);
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
	d_a  = clCreateBuffer(context,  CL_MEM_READ_ONLY,  sizeof(float) * count, NULL, &err);
    d_c  = clCreateBuffer(context,  CL_MEM_WRITE_ONLY, sizeof(float) * count, NULL, &err);
    d_sum  = clCreateBuffer(context,  CL_MEM_WRITE_ONLY , sizeof(float) , NULL, &err);

    err = clEnqueueWriteBuffer(commands, d_a, CL_TRUE, 0, sizeof(float) * count, h_a, 0, NULL, NULL);

    err = clSetKernelArg(ko_vadd, 0, sizeof(cl_mem), &d_a);
    err = clSetKernelArg(ko_vadd, 1, sizeof(cl_mem), &d_c);
    err = clSetKernelArg(ko_vadd, 2, sizeof(cl_mem), &d_sum);
    err = clSetKernelArg(ko_vadd, 3, sizeof(unsigned int), &count);
   

    global=count;
    err = clEnqueueNDRangeKernel(commands, ko_vadd, 1, NULL, &global, NULL, 0, NULL, NULL);

    err = clFinish(commands);

    err = clEnqueueReadBuffer( commands, d_c, CL_TRUE, 0, sizeof(float) * count, h_c, 0, NULL, NULL );
    err = clEnqueueReadBuffer( commands, d_sum, CL_TRUE, 0, sizeof(float) , sum, 0, NULL, NULL );

    sum[0]=sum[0];

    for(i=0;i<10;i++)
    	printf("%f \n", h_c[i]);

    printf("SUM = %f \n", sum[0]);

    clReleaseMemObject(d_a);
    clReleaseMemObject(d_c);
    clReleaseProgram(program);
    clReleaseKernel(ko_vadd);
    clReleaseCommandQueue(commands);
    clReleaseContext(context);

    free(h_a);
    free(h_c);

    return 0;
}