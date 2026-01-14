#include "opencl.hpp"

int main() {
	Device device(select_device_with_most_flops()); // compile OpenCL C code for the fastest available device

	const uint N = 256u; // size of vectors
	Memory<float> A(device, N); // allocate memory on both host and device
	Memory<float> B(device, N);
	Memory<float> C(device, N);

	Kernel add_kernel(device, N, "add_kernel", A, B, C); // kernel that runs on the device

	for(uint n=0u; n<N; n++) {
		A[n] = static_cast<float>(n); // initialize memory
		B[n] = static_cast<float>(n*n);
		C[n] = 0.0f;
	}

	println("Value before kernel execution: C[0] = "+to_string(C[0]));

	A.write_to_device(); // copy data from host memory to device memory
	B.write_to_device();
	add_kernel.run(); // run add_kernel on the device
	C.read_from_device(); // copy data from device memory to host memory

	for (uint n = 0u; n < N; n++)
	{
		println(to_string(C[n]));
	}

	wait();
	return 0;
}