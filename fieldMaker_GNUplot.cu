
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <unistd.h>
#include <cuda.h>
#include <string.h>
#include <math.h>

// consider making wfs into a float4 array, with the wfs from 4 els stacked side-by-side -> less loop iters

// command to check gpu device info
/*** ./usr/local/cuda-9.2/extras/demo_suite/deviceQuery ***/

// compile command
/*** nvcc -Xptxas=-O3,-v -use_fast_math -maxrregcount=20 fieldMaker_paramTester.cu ***/

// physical constants
#define CW				1.482	// water sound speed (mm/us)

// receive system specifics
#define RECLEN			8191	// length of acquired signal
#define NELS			160		// number of receiver elements
#define NLOCS			5
#define NPULSES			4
#define ADC_CLOCK		20		// digitizer clock (MHz)

// prescribed constant
#define DT				105.0	// time-of-flight = ( element-to-origin (100 us) + transmit-system-specific delays (10.5 us))

// derived constants to take some load off gpu
#define DT_TIMES_ADC_CLOCK	( DT * ADC_CLOCK )
#define ADC_CLOCK_OVER_CW	( ADC_CLOCK / CW )

//~ const float ADC_CLOCK_OVER_CW = ADC_CLOCK/CW;
//~ const float DT_TIMES_ADC_CLOCK = DT*ADC_CLOCK;

// center point of the pressure field being calculated
#define FIELD_X0 		0.0
#define FIELD_Y0 		0.0
#define FIELD_Z0 		0.0

// size of the pressure field to calculate ( units = mm ) 
#define FIELD_DIM_X 	30.0
#define FIELD_DIM_Y 	30.0
#define FIELD_DIM_Z 	30.0

// constants for gpu
#define BLOCK_DIM_X		4
#define BLOCK_DIM_Y		8
#define BLOCK_DIM_Z		8

// max threads = 1024 (GPU specific, this is for laptop) [8*8*16=1024]
#define THREAD_DIM_X 	16
#define THREAD_DIM_Y 	8
#define THREAD_DIM_Z	8 //works best if THREAD_DIM_Z < THREAD_DIM_X/Y
#define BLOCK_SIZE		(THREAD_DIM_X*THREAD_DIM_Y*THREAD_DIM_Z)

// constant to calculate indices of calculation grid
#define IE (BLOCK_DIM_X*THREAD_DIM_X)
#define JE (BLOCK_DIM_Y*THREAD_DIM_Y)
#define KE (BLOCK_DIM_Z*THREAD_DIM_Z)

// variable to hold coordinate locations of array elements
__constant__ float4 arxyz[NELS];

// projects the measured signals from the array elements back into the field, sums them together
__global__ void getSignalIntensityField(float *si_field, float4 *wfs, float4 *XYZ, float dt){
	
	int x = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;
	int y = __mul24(blockIdx.y,blockDim.y) + threadIdx.y;
	int z = __mul24(blockIdx.z,blockDim.z) + threadIdx.z;	
	int sigint_idx = (z*JE + y)*IE + x; // index of field location being calculated
		
	int tidx; 						// index of 'wfs' vector
	float dx,dy,dz;					// unit distances from array element to location in calculation field 
	float4 xyz = XYZ[sigint_idx];	// makes calculation faster to load value stored in global mem into local mem 

	#pragma unroll 20
	for(int eln = 0; eln<NELS; eln++){
		dx =  __fsub_rn(xyz.x,arxyz[eln].x);
		dy =  __fsub_rn(xyz.y,arxyz[eln].y);
		dz =  __fsub_rn(xyz.z,arxyz[eln].z);
				
		tidx = __float2int_rn( __fadd_rn( __fmul_rn( __fsqrt_rn( __fadd_rn( __fadd_rn( __fmul_rn(dx,dx) , __fmul_rn(dy,dy) ) , __fmul_rn(dz,dz)) ) , ADC_CLOCK_OVER_CW), (DT+dt)*ADC_CLOCK)) + __mul24(eln,RECLEN);


		// this script loads 4 waveforms at a time, this adds the signal values from all of them together at once
		si_field[sigint_idx] += __fadd_rn(__fsub_rn(wfs[ tidx ].x, wfs[ tidx ].y),__fsub_rn(wfs[ tidx ].z, wfs[ tidx ].w));	
	}	
}


// function to set the field to 0
__global__ void resetSignalIntensityField(float *pfield){
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	int z = blockIdx.z*blockDim.z + threadIdx.z;	
	int sigint_idx = z*IE*JE + y*IE + x;	
	pfield[sigint_idx] = 0;
}

// loads the signals from file
void loadWF(float4 *wf){
	float *wftmp;
	wftmp = (float *)malloc(RECLEN*NELS*NPULSES*sizeof(float));
	FILE *fname;			
	//~ fname = fopen("hiampwfHilb_sk","rb");
	fname = fopen("hiampwf_sk","rb");
	fread(wftmp,sizeof(wftmp),RECLEN*NELS*NPULSES,fname);
	fclose(fname);
	
	for(int rl=1;rl<(NPULSES*NELS*RECLEN);rl++){
		wf[rl].x=wftmp[rl];
		wf[rl].y=wftmp[rl+NELS*RECLEN]*0;
		wf[rl].z=wftmp[rl+(NELS*RECLEN)*2]*0;
		wf[rl].w=wftmp[rl+3*NELS*RECLEN]*0;
	}
	
	for(int n=0;n<NELS;n++){
		wf[n*RECLEN].x=0;
		wf[n*RECLEN].y=0;
		wf[n*RECLEN].z=0;
		wf[n*RECLEN].w=0;		
	}
	free(wftmp);
}

// loads the coordinates of the receiver elements of the array
void loadArray(){	
	float *array_tmp;
	array_tmp = (float *)malloc(256*3*sizeof(float));
	float4 arxyz_h[NELS];
	FILE *fname;				
	fname = fopen("./dataFiles/arrayCoords_bin","rb");	
	fread(array_tmp,sizeof(array_tmp),RECLEN*NELS,fname);
	fclose(fname);	
	
	int cntr1, cntr2;
	cntr1 = 0;
	for(cntr2=0;cntr2<256;cntr2++){		
		if( (cntr2<128)  && (cntr2%4 == 0) ){
			arxyz_h[cntr1].x = array_tmp[cntr2];
			arxyz_h[cntr1].y = array_tmp[cntr2+256];
			arxyz_h[cntr1].z = array_tmp[cntr2+512];
			cntr1++;
		} else if (cntr2>=128) {
			arxyz_h[cntr1].x = array_tmp[cntr2];
			arxyz_h[cntr1].y = array_tmp[cntr2+256];
			arxyz_h[cntr1].z = array_tmp[cntr2+512];
			cntr1++;
		}	
	}
	cudaMemcpyToSymbol(arxyz,arxyz_h,NELS*sizeof(float4));
	free(array_tmp);
}

// generates a 4-vector storing the locations of all points in the field we are trying to calculate
// the 4 field of the vector is unused, but in my testing it looked like it was faster to use float4 than float3 and I wasn't running up against any memory limits so I stuck with it
void setFieldLocsXYZ(float4 *xyz){
	int cntr1,cntr2,cntr3;
	int xsteps,ysteps,zsteps;

	xsteps = (BLOCK_DIM_X*THREAD_DIM_X);
	ysteps = (BLOCK_DIM_Y*THREAD_DIM_Y);
	zsteps = (BLOCK_DIM_Z*THREAD_DIM_Z);
	
	float x0,y0,z0;
	x0 = FIELD_X0 - FIELD_DIM_X/2.0;
	y0 = FIELD_Y0 - FIELD_DIM_Y/2.0;
	z0 = FIELD_Z0 - FIELD_DIM_Z/2.0;
	
	float dx,dy,dz;
	dx = FIELD_DIM_X/xsteps;
	dy = FIELD_DIM_Y/ysteps;
	dz = FIELD_DIM_Z/zsteps;
	
	for(cntr1=0;cntr1<xsteps;cntr1++){	
		for(cntr2=0;cntr2<ysteps;cntr2++){
			for(cntr3=0;cntr3<ysteps;cntr3++){
				xyz[(cntr1*ysteps+cntr2)*zsteps+cntr3].x = x0+cntr3*dx;
				xyz[(cntr1*ysteps+cntr2)*zsteps+cntr3].y = y0+cntr2*dy;
				xyz[(cntr1*ysteps+cntr2)*zsteps+cntr3].z = z0+cntr1*dz;
			}
		}
	}	
}


int plotData(float *siField, FILE *pipe, float dt){
	//~ printf("HELLLO\n");
	FILE *fname;			
	fname = fopen("fdata.dat","w");
	
	int nz = 3;
	int N = 5;
	int xx,yy,zz;
	xx=0; yy=0;
	
	for(int nn=0;nn<nz;nn++){
		for(yy=0;yy<BLOCK_DIM_Y*THREAD_DIM_Y;yy++){
			for(zz=0;zz<N;zz++){
				for(xx=0;xx<BLOCK_DIM_X*THREAD_DIM_X;xx++){	
					fprintf(fname,"%f ",siField[(((zz+nn*N)*BLOCK_DIM_Z*THREAD_DIM_Z/(N*nz))*JE + yy)*IE + xx]);
				}			
			}
			fprintf(fname,"\n");	
		}
	}
	
	fclose(fname);
	usleep(35000);
	fprintf(pipe, "set term x11 size 3000,1800 font 'Helvetica,80'\n");
	fprintf(pipe, "set lmargin at screen 0.02; set rmargin at screen 0.9; set bmargin at screen 0.02; set tmargin at screen 0.98\n");
	fprintf(pipe, "set pm3d map\n");
	fprintf(pipe, "set xrange [%.2f:%.2f]; set yrange [%.2f:%.2f];\n", (0.0-FIELD_DIM_X*N*0.5),(FIELD_DIM_X*N*0.5),(0.0-FIELD_DIM_Y*nz*0.5),(FIELD_DIM_Y*nz*0.5));
	//~ fprintf(pipe, "set cbrange[-300:500]\n");
	fprintf(pipe, "set grid\n");
	fprintf(pipe, "set title 't=%.2f'\n",dt+DT);
	fprintf(pipe, "splot 'fdata.dat' u (($1-%f)*%f):(($2-%f)*%f):3 matrix with image noti\n",N*BLOCK_DIM_X*THREAD_DIM_X*0.5,FIELD_DIM_X/(BLOCK_DIM_X*THREAD_DIM_X),nz*BLOCK_DIM_Y*THREAD_DIM_Y*0.5,FIELD_DIM_Y/(BLOCK_DIM_Y*THREAD_DIM_Y));
	fflush(pipe);
	usleep(50000);
    return(0);

}





int main(){
	
	// timer variables	
	struct timeval t0,t1;
	
	// load waveforms
	float4 *wf, *wf_d;
	wf = (float4 *)malloc( NPULSES*RECLEN*NELS*sizeof(float4) );
	loadWF(wf); 
	cudaMalloc( &wf_d, RECLEN*NELS*sizeof(float4) );
	cudaMemcpy( wf_d, &wf[RECLEN*NELS], RECLEN*NELS*sizeof(float4), cudaMemcpyHostToDevice ); // copy to gpu
	
	// load array coords to __constant__ memory on GPU
	loadArray();
	
	// allocate memory for signal intensity field calcs, initialize to 0, and put it on the GPU
	float *sig_field_host, *sig_field_pinned, *sig_field_d;
	int nmemb = (BLOCK_DIM_X*THREAD_DIM_X)*(BLOCK_DIM_Y*THREAD_DIM_Y)*(BLOCK_DIM_Z*THREAD_DIM_Z);
	unsigned int sig_field_size = nmemb*sizeof(float);
	sig_field_host = (float *)malloc(sig_field_size);
	cudaMallocHost((void **)&sig_field_pinned,sig_field_size);
	cudaMalloc((void **)&sig_field_d,sig_field_size);	
	memset(sig_field_host,0,sig_field_size); // initialize sig_field to 0
	memcpy(sig_field_pinned,sig_field_host,sig_field_size); // copy to pinned memory
	cudaMemcpy(sig_field_d,sig_field_host,sig_field_size,cudaMemcpyHostToDevice); // copy to gpu

	// allocate memory for the spatial coordinates of the calculation grid, populate it, and put it on the GPU
	float4 *xyz_h, *xyz_d;
	int nmembxyz = (BLOCK_DIM_X*THREAD_DIM_X)*(BLOCK_DIM_Y*THREAD_DIM_Y)*(BLOCK_DIM_Z*THREAD_DIM_Z);
	unsigned int xyz_size = nmembxyz*sizeof(float4);
	xyz_h = (float4 *)malloc(xyz_size);
	cudaMalloc((void **)&xyz_d,xyz_size);
	setFieldLocsXYZ(xyz_h);
	cudaMemcpy(xyz_d,xyz_h,xyz_size,cudaMemcpyHostToDevice); // copy to gpu
	
	// setup cuda blocks/threads
	dim3 num_blocks(BLOCK_DIM_X, BLOCK_DIM_Y, BLOCK_DIM_Z);
	dim3 num_threads(THREAD_DIM_X, THREAD_DIM_Y, THREAD_DIM_Z);
	
	FILE *pipe = popen("gnuplot -persist","w");
	// calculate the signal intenisty in the field
	gettimeofday(&t0,NULL);
	for(float dt=-25.0;dt<5.1;dt+=0.2){
		resetSignalIntensityField <<<num_blocks, num_threads>>> ( sig_field_d );	
		getSignalIntensityField <<<num_blocks, num_threads>>> ( sig_field_d , wf_d, xyz_d, dt);
		cudaMemcpy(sig_field_host,sig_field_d,sig_field_size,cudaMemcpyDeviceToHost);	
		plotData(sig_field_host, pipe, dt);
	}
	gettimeofday(&t1,NULL);
	printf("calc time = %d us\n",t1.tv_usec-t0.tv_usec);
	
	sleep(2);
	fclose(pipe);
	// write the results to file
	FILE *fname;
	fname = fopen("sig_field_bin","wb");	
	fwrite(sig_field_host,sig_field_size,nmemb,fname);
	fclose(fname);

	// free allocated memory
	cudaFree(wf_d);  free(wf);	
	cudaFree(sig_field_d); 
	cudaFreeHost(sig_field_pinned);
	free(sig_field_host);
	cudaFree(xyz_d); free(xyz_h);
	
	return 0;
}





