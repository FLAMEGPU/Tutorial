
/*
* FLAME GPU v 1.4.0 for CUDA 7.5
* Copyright 2015 University of Sheffield.
* Author: Dr Paul Richmond
* Contact: p.richmond@sheffield.ac.uk (http://www.paulrichmond.staff.shef.ac.uk)
*
* University of Sheffield retain all intellectual property and
* proprietary rights in and to this software and related documentation.
* Any use, reproduction, disclosure, or distribution of this software
* and related documentation without an express license agreement from
* University of Sheffield is strictly prohibited.
*
* For terms of licence agreement please attached licence or view licence
* on www.flamegpu.com website.
*
*/
#include <cuda_runtime.h>
#include <stdio.h>
#ifdef VISUALISATION
#include <GL/glew.h>
#include <GL/glut.h>
#endif
#include "header.h"

/* IO Variables*/
char inputfile[100];          /**< Input path char buffer*/
char outputpath[1000];         /**< Output path char buffer*/

#define OUTPUT_TO_XML 0


/** checkUsage
 * Function to check the correct number of arguments
 * @param arc	main argument count
 * @param argv	main argument values
 * @return true if usage is correct, otherwise false
 */
int checkUsage( int argc, char** argv){
	//Check usage
#ifdef VISUALISATION
	printf("FLAMEGPU Visualisation mode\n");
	if(argc < 2)
	{
		printf("Usage: main [XML model data] [Optional CUDA device ID]\n");
		return false;
	}
#else
	printf("FLAMEGPU Console mode\n");
	if(argc < 3)
	{
		printf("Usage: main [XML model data] [Iterations] [Optional CUDA device ID]\n");
		return false;
	}
#endif
	return true;
}

const char* getOutputDir(){
  return outputpath;
}


/** setFilePaths
 * Function to set global variables for the input XML file and its directory location
 *@param input input path of model xml file
 */
void setFilePaths(char* input){
	//Copy input file
	strcpy(inputfile, input);
	printf("Initial states: %s\n", inputfile);

	//Calculate the output path from the path of the input file
	int i = 0;
	int lastd = -1;
	while(inputfile[i] != '\0')
	{
		/* For windows directories */
		if(inputfile[i] == '\\') lastd=i;
		/* For unix directories */
		if(inputfile[i] == '/') lastd=i;
		i++;
	}
	strcpy(outputpath, inputfile);
	outputpath[lastd+1] = '\0';
	printf("Output dir: %s\n", outputpath);
}


void initCUDA(int argc, char** argv){
	cudaError_t cudaStatus;
	int device;
	int device_count;

	//default device
	device = 0;
	cudaStatus = cudaGetDeviceCount(&device_count);

	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Error finding CUDA devices!  Do you have a CUDA-capable GPU installed?\n");
		exit(EXIT_FAILURE);
	}
	if (device_count == 0){
		fprintf(stderr, "Error no CUDA devices found!\n");
		exit(EXIT_FAILURE);
	}

#ifdef VISUALISATION
	if (argc == 3){
		device = atoi(argv[2]);
	}
#else
	if (argc == 4){
		device = atoi(argv[3]);
	}
#endif

	if (device >= device_count){
		fprintf(stderr, "Error selecting CUDA device! Device id '%d' is not found?\n", device);
		exit(EXIT_FAILURE);
	}

	// Select device
	cudaStatus = cudaSetDevice(device);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Error setting CUDA device!\n");
		exit(EXIT_FAILURE);
	}
}


/**
 * Program main (Handles arguments)
 */
int main( int argc, char** argv) 
{
	cudaError_t cudaStatus;
	//check usage mode
	if (!checkUsage(argc, argv))
		exit(EXIT_FAILURE);

	//get the directory paths
	setFilePaths(argv[1]);

	//initialise CUDA
	initCUDA(argc, argv);

#ifdef VISUALISATION
	//Init visualisation must be done before simulation init
	initVisualisation();
#endif

	//initialise the simulation
	initialise(inputfile);

    
#ifdef VISUALISATION
	runVisualisation();
	exit(EXIT_SUCCESS);
#else	
	//Benchmark simulation
	cudaEvent_t start, stop;
	float milliseconds = 0;
	
	//create timing events
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	//Get the number of iterations
	int iterations = atoi(argv[2]);
	if (iterations == 0)
	{
		printf("Second argument must be an integer (Number of Iterations)\n");
		exit(EXIT_FAILURE);
	}
  
	//start timing
	cudaEventRecord(start);

	for (int i=0; i< iterations; i++)
	{
		printf("Processing Simulation Step %i\n", i+1);

		//single simulation iteration
		singleIteration();

		if (OUTPUT_TO_XML)
		{
			saveIterationData(outputpath, i+1, 
				//default1 state prey agents
				get_host_prey_default1_agents(), get_device_prey_default1_agents(), get_agent_prey_default1_count(),
				//default2 state predator agents
				get_host_predator_default2_agents(), get_device_predator_default2_agents(), get_agent_predator_default2_count(),
				//default3 state grass agents
				get_host_grass_default3_agents(), get_device_grass_default3_agents(), get_agent_grass_default3_count());
			
				printf("Iteration %i Saved to XML\n", i+1);
		}

	}

	//CUDA stop timing
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&milliseconds, start, stop);
	printf( "Total Processing time: %f (ms)\n", milliseconds);
#endif

	cleanup();
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Error resetting the device!\n");
		return EXIT_FAILURE;
	}
	return EXIT_SUCCESS;
}
