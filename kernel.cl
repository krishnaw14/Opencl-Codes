__kernel void vadd(__global float* a, __global float* c, __global float* sum, const unsigned int count)
{
	int i = get_global_id(0);                                  
 	float flag=1.0f;                                    
    if(i < count)                                              
	{                                                          
       c[i] = sin(a[i]);                                    
       flag+=c[i];                                     
	   if(i==(count-1))			
	       sum[0] =flag;			
	}                                                          
}