
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <cstdlib>
#include <iostream>
#include "lodepng.h"

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

void getError(cudaError_t err) {
	if (err != cudaSuccess) {
		std::cout << "CUDA Error " << cudaGetErrorString(err) << std::endl;
	}
}

__global__ void blur(unsigned char* input_image, unsigned char* output_image, int width, int height) {

	const unsigned int offset = blockIdx.x*blockDim.x + threadIdx.x;
	int x = offset % width;
	int y = (offset - x) / width;
	int fsize = 5; // Filter size
	if (offset < width*height) {

		float output_red = 0;
		float output_green = 0;
		float output_blue = 0;
		int hits = 0;
		for (int ox = -fsize; ox < fsize + 1; ++ox) {
			for (int oy = -fsize; oy < fsize + 1; ++oy) {
				if ((x + ox) > -1 && (x + ox) < width && (y + oy) > -1 && (y + oy) < height) {
					const int currentoffset = (offset + ox + oy*width) * 3;
					output_red += input_image[currentoffset];
					output_green += input_image[currentoffset + 1];
					output_blue += input_image[currentoffset + 2];
					hits++;
				}
			}
		}
		output_image[offset * 3] = output_red / hits;
		output_image[offset * 3 + 1] = output_green / hits;
		output_image[offset * 3 + 2] = output_blue / hits;
	}
}

void filter(unsigned char* input_image, unsigned char* output_image, int width, int height) {

	unsigned char* dev_input;
	unsigned char* dev_output;
	getError(cudaMalloc((void**)&dev_input, width*height * 3 * sizeof(unsigned char)));
	getError(cudaMemcpy(dev_input, input_image, width*height * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice));

	getError(cudaMalloc((void**)&dev_output, width*height * 3 * sizeof(unsigned char)));

	dim3 blockDims(512, 1, 1);
	dim3 gridDims((unsigned int)ceil((double)(width*height * 3 / blockDims.x)), 1, 1);

	blur<<<gridDims, blockDims>>>(dev_input, dev_output, width, height);

	getError(cudaMemcpy(output_image, dev_output, width*height * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost));

	getError(cudaFree(dev_input));
	getError(cudaFree(dev_output));

}

int main(int argc, char** argv)
{
	const int arraySize = 5;
	const int a[arraySize] = { 1, 2, 3, 4, 5 };
	const int b[arraySize] = { 10, 20, 30, 40, 50 };
	int c[arraySize] = { 0 };

	// Add vectors in parallel.
	cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addWithCuda failed!");
		return 1;
	}

	if (argc != 3) {
		std::cout << "Enter two command line params: input.png output.png" << std::endl;
		return 0;
	}

	const char* input_file = argv[1];
	const char* output_file = argv[2];

	std::vector<unsigned char> in_image;
	unsigned int width, height;

	// Load the data
	unsigned error = lodepng::decode(in_image, width, height, input_file);
	if (error) {
		std::cout << "PNG decoder error " << error << ": " << lodepng_error_text(error) << std::endl;
		return 1;
	}

	int size_rgb = (in_image.size() / 4) * 3;

	// Prepare data for CUDA 
	unsigned char* input_image = new unsigned char[size_rgb];
	unsigned char* output_image = new unsigned char[size_rgb];
	int index = 0;
	for (int i = 0; i < in_image.size(); ++i) {
		if (i % 4 != 0) {
			input_image[index] = in_image.at(i);
			index++;
		}
	}

	// Filtering
	filter(input_image, output_image, width, height);

	// Prepare data for output
	std::vector<unsigned char> out_image;
	for (int i = 0; i < size_rgb; ++i) {
		if (i % 3 == 0) {
			out_image.push_back(255);
		}
		out_image.push_back(output_image[i]);
	}
	out_image.push_back(255);

	// Output the data
	error = lodepng::encode(output_file, out_image, width, height);

	//if there's an error, display it
	if (error) {
		std::cout << "PNG encoder error " << error << ": " << lodepng_error_text(error) << std::endl;
	}

	// Clean up
	delete[] input_image;
	delete[] output_image;

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
