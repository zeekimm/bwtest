#include <iostream>
#include <chrono>
#include <vector>
#include <memory>
#include <string>
#include <cstring>
#include "CL/opencl.hpp"

static inline float GetDurationMs(const std::chrono::steady_clock::time_point &base,
    const std::chrono::steady_clock::time_point &curr) {
    return std::chrono::duration<float, std::milli>(curr - base).count();
}

static inline void ShowTime(const std::vector<float> &timeList, const std::string &tag = "") {
    float tTotal = 0.f, tMin = 10000.f, tMax = 0.f, tAvg = 0.f, tCount = 0;
    for (auto &t : timeList) {
        tMin = std::min(t, tMin);
        tMax = std::max(t, tMax);
        tTotal += t;
        tCount ++;
    }

    for (auto &t : timeList) {
        std::cout << tag << " cur: " << t << "ms,"
                         << " avg: " << tTotal/tCount << "ms,"
                         << " min: " << tMin << "ms,"
                         << " max: " << tMax << "ms" << std::endl; 
    }
}

static const int testWidthInBytes = 360 * 96 * 2;
static const int testHeight = 360;

static void TestCpu();
static void TestGpu();

int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cout << "usage:" << argv[0] << " type, eg: " << std::endl;
        std::cout << "--" << argv[0] << " cpu " << std::endl;
        std::cout << "--" << argv[0] << " gpu " << std::endl;
        exit(-22);
    }

    if (std::string("cpu") == argv[1]) {
        TestCpu();
    } else {
        TestGpu();
    }

    return 0;
}

static void TestCpu() {
    const int totalBytes = testWidthInBytes * testHeight;
    int *src = static_cast<int*>(malloc(totalBytes));
    int *dst = static_cast<int*>(malloc(totalBytes));

    // TODO: warnup?
    memcpy(src, dst, totalBytes);

    std::vector<float> tUsed;
    
    for (int i = 0; i < 50; i++) {
        auto t1 = std::chrono::steady_clock::now();
        memcpy(src, dst, totalBytes);
        auto t2 = std::chrono::steady_clock::now();
        tUsed.push_back(GetDurationMs(t1, t2));
    }

    ShowTime(tUsed, "cpubw");
}

std::string gOpenclDemoSource = R"""(
#ifndef DATA_TYPE
    #define DATA_TYPE int
#endif
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void benchmark(__global const DATA_TYPE* src, __global DATA_TYPE* dst, int size) {
    int gid = get_global_id(0);

    if (gid >= size) return;

    dst[gid] = src[gid];
}
)""";

std::vector<std::pair<int, std::string>> cfgs = {
    // TODO: cl_TYPE3 may has wrong size

    // {sizeof(cl_char),  "-DDATA_TYPE=char"},
    // {sizeof(cl_char2), "-DDATA_TYPE=char2"},
    // {sizeof(cl_char3), "-DDATA_TYPE=char3"},
    // {sizeof(cl_char4), "-DDATA_TYPE=char4"},

    // {sizeof(cl_uchar), "-DDATA_TYPE=uchar"},
    // {sizeof(cl_uchar2), "-DDATA_TYPE=uchar2"},
    // {sizeof(cl_uchar3), "-DDATA_TYPE=uchar3"},
    // {sizeof(cl_uchar4), "-DDATA_TYPE=uchar4"},

    // {sizeof(cl_short), "-DDATA_TYPE=short"},
    // {sizeof(cl_short2), "-DDATA_TYPE=short2"},
    // {sizeof(cl_short3), "-DDATA_TYPE=short3"},
    // {sizeof(cl_short4), "-DDATA_TYPE=short4"},

    // {sizeof(cl_ushort), "-DDATA_TYPE=ushort"},
    {sizeof(cl_ushort2), "-DDATA_TYPE=ushort2"},
    // {sizeof(cl_ushort3), "-DDATA_TYPE=ushort3"},
    // {sizeof(cl_ushort4), "-DDATA_TYPE=ushort4"},

    // {sizeof(cl_half), "-DDATA_TYPE=half"},
    // {sizeof(cl_half2), "-DDATA_TYPE=half2"},
    // {sizeof(cl_half3), "-DDATA_TYPE=half3"},
    // {sizeof(cl_half4), "-DDATA_TYPE=half4"},

    // {sizeof(cl_float), "-DDATA_TYPE=float"},
    // {sizeof(cl_float2), "-DDATA_TYPE=float2"},
    // {sizeof(cl_float3), "-DDATA_TYPE=float3"},
    // {sizeof(cl_float4), "-DDATA_TYPE=float4"},

    // { sizeof(cl_int), "-DDATA_TYPE=int" },
    // {sizeof(cl_int2), "-DDATA_TYPE=int2"},
    // {sizeof(cl_int3), "-DDATA_TYPE=int3"},
    // {sizeof(cl_int4), "-DDATA_TYPE=int4"},

    // {sizeof(cl_uint), "-DDATA_TYPE=uint"},
    // {sizeof(cl_uint2), "-DDATA_TYPE=uint2"},
    // {sizeof(cl_uint3), "-DDATA_TYPE=uint3"},
    // {sizeof(cl_uint4), "-DDATA_TYPE=uint4"},
};

static float GetCLRunningTime(cl_event event) {
    // CL_PROFILING_COMMAND_QUEUED                 
    // CL_PROFILING_COMMAND_SUBMIT                 
    // CL_PROFILING_COMMAND_START          GPU start
    // CL_PROFILING_COMMAND_END            GPU end
    // CL_PROFILING_COMMAND_COMPLETE               

    cl_ulong tpStart = 0, tpEnd = 0;
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &tpStart, nullptr);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &tpEnd, nullptr);

    return (tpEnd - tpStart) / 1000000.f;
}

static void TestGpu() {
    cl_uint numPlatform = 0, numDevice = 0;
    cl_uint status = clGetPlatformIDs(0, NULL, &numPlatform);
    if (status != CL_SUCCESS || numPlatform <= 0) {
        throw std::runtime_error("clGetPlatformIDs fail");
    }

    std::cout << "--Found " << numPlatform << " Opencl Platform" << std::endl;

    std::vector<cl_platform_id> platforms(numPlatform);
    status = clGetPlatformIDs(numPlatform, platforms.data(), nullptr);
    if (status != CL_SUCCESS) {
        std::cout << "--clGetPlatformIDs fail:" << status << std::endl;
        throw std::runtime_error("clGetPlatformIDs fail");
    } else {
        std::cout << "--clGetPlatformIDs successed:" << status << std::endl;
    }

    cl_device_id device = nullptr;
    for (cl_uint i = 0; i < numPlatform; i++) {
        auto platform = platforms[i];
        status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &numDevice);
        if (status != CL_SUCCESS) {
            std::cerr << "--clGetDeviceIDs fail:" << status << std::endl;
            throw std::runtime_error("clGetDeviceIDs fail");
        }

        if (numDevice <= 0) {
            continue;
        }

        std::cout << "--Found " << numDevice << " Opencl device in platform " << i << std::endl;

        std::vector<cl_device_id> devices(numDevice);
        status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, numDevice, devices.data(), nullptr);
        if (status != CL_SUCCESS) {
            std::cout << "--clGetDeviceIDs fail:" << status << std::endl;
            throw std::runtime_error("clGetDeviceIDs fail");
        }

        device = devices[0];
        break;
    }

    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, nullptr);
    if (context == nullptr) {
        std::cerr << "--clCreateContext fail" << std::endl;
        throw std::runtime_error("clCreateContext fail");
    }

    // enable queue profiling to query gpu run time, clFinish contain api calling time?
    cl_queue_properties option[3] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
    cl_int err = CL_SUCCESS;
    cl_command_queue cmdQueue = clCreateCommandQueueWithProperties(context, device, option, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "--clCreateCommandQueueWithProperties fail" << std::endl;
        throw std::runtime_error("clCreateCommandQueueWithProperties fail");
    }

    for (auto &cfg : cfgs) {
        std::cout << "--Test with config:" << cfg.second << std::endl;
        std::string kernelOptions = cfg.second;

        const char *kernelSrc = gOpenclDemoSource.c_str();
        size_t size[] = { gOpenclDemoSource.size() };
        cl_program program = clCreateProgramWithSource(context, 1, &kernelSrc, size, &err);

        err = clBuildProgram(program, 1, &device, kernelOptions.c_str(), nullptr, nullptr);
        if (err != CL_SUCCESS) {
            std::vector<char> errBuf(0x10000);
            err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, errBuf.size(),
                errBuf.data(), nullptr);
            std::cerr << "--Build log:" << errBuf.data() << std::endl;
            throw std::runtime_error("clGetProgramBuildInfo fail");
        }

        cl_kernel kernel = clCreateKernel(program, "benchmark", &err);

        size_t maxWorkGroupSize;
        err = clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_WORK_GROUP_SIZE,
            sizeof(size_t), &maxWorkGroupSize, NULL);

        int totalBytes = testWidthInBytes * testHeight;
        int totalSize  = totalBytes / cfg.first;

        cl_mem srcMem = clCreateBuffer(context, CL_MEM_READ_WRITE, totalBytes, nullptr, &err);
        cl_mem dstMem = clCreateBuffer(context, CL_MEM_READ_WRITE, totalBytes, nullptr, &err);

        clSetKernelArg(kernel, 0, sizeof(cl_mem), &srcMem);
        clSetKernelArg(kernel, 1, sizeof(cl_mem), &dstMem);
        clSetKernelArg(kernel, 2, sizeof(int), &totalSize);

        size_t opos[] = { 0, 0, 0 };
        size_t gdim[] = { static_cast<size_t>(totalSize), 1, 1 };
        size_t ldim[] = { static_cast<size_t>(maxWorkGroupSize), 1, 1 };

        // TODO: warp up?
        clEnqueueNDRangeKernel(cmdQueue, kernel, 1, opos, gdim, ldim, 0, nullptr, nullptr);
        clFinish(cmdQueue);

        std::vector<float> tUsed;
        std::vector<cl_event> events;

        clFinish(cmdQueue);
        for (int i = 0; i < 50; i ++) {
            cl_event event = nullptr;
            err = clEnqueueNDRangeKernel(cmdQueue, kernel, 1, opos, gdim, ldim, 0, nullptr, &event);
            if (err != CL_SUCCESS) {
                throw std::runtime_error("clEnqueueNDRangeKernel fail");
            }
            events.push_back(event);
            clFlush(cmdQueue);
            // clFinish(cmdQueue);
        }

        clFinish(cmdQueue);
        for (auto &event : events) tUsed.push_back(GetCLRunningTime(event));
        ShowTime(tUsed, "gpubw");

        for (auto &event : events) clReleaseEvent(event);

        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseMemObject(srcMem);
        clReleaseMemObject(dstMem);
    }

    clReleaseCommandQueue(cmdQueue);
    clReleaseContext(context);
    clReleaseDevice(device);
}
