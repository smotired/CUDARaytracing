/// Kernel functions for tracing rays
#pragma once
#include "rays.cuh"
#include "renderer.cuh"

// Size of a single ray's queue in a single block.
#define SINGLE_QUEUE_SIZE (RAY_BLOCKDIM*RAY_BLOCKDIM*RAY_QUEUE_SIZE)

// A queue for secondary rays.
struct RayQueue {
    Ray* rays;               // Unified queue of all rays across all blocks.

    unsigned int *readIdxs;  // Indexes that each block is reading from its queue.
    unsigned int *endIdxs;   // Indexes for each block to stop reading at.
    unsigned int *writeIdxs; // Indexes for each block to write rays to.
    unsigned int *maxIdxs;   // Indexes for each block to stop writing at.

    // Enqueues a ray into that block's queue
    __device__ void Enqueue(const uint3 blockIdx, const Ray& ray) const {
        const unsigned int bidx = blockIdx.y * RAY_BLOCKCOUNT + blockIdx.x; // Index of the block itself
        const unsigned int qidx = bidx * 2 * SINGLE_QUEUE_SIZE;             // Start index of the block's first queue

        // Try to add to the queue, with an assertion to stop the kernel gracefully if we run out of room
        const unsigned int ridx = atomicAdd(writeIdxs + bidx, 1);
        assert(ridx < maxIdxs[bidx]);
        rays[qidx + ridx] = ray;
    }

    // Dequeues a ray from that block's queue
    __device__ Ray* Dequeue(const uint3 blockIdx) const {
        const unsigned int bidx = blockIdx.y * RAY_BLOCKCOUNT + blockIdx.x;
        const unsigned int qidx = bidx * 2 * SINGLE_QUEUE_SIZE;

        // Dequeue, or return nullptr if the queue is empty.
        const unsigned int ridx = atomicAdd(readIdxs + bidx, 1);
        if (ridx >= endIdxs[bidx]) return nullptr;
        return rays + qidx + ridx;
    }

    // Check if a block's ray queue is empty. (Not thread safe -- should use __syncthreads() before/after)
    __device__ bool IsEmpty(const uint3 blockIdx) const {
        const unsigned int bidx = blockIdx.y * RAY_BLOCKCOUNT + blockIdx.x;
        return readIdxs[bidx] >= endIdxs[bidx];
    }

    // Swaps a block's queues
    __device__ void SwapQueues(uint3 const blockIdx) const {
        const unsigned int bidx = blockIdx.y * RAY_BLOCKCOUNT + blockIdx.x;

        const bool enqueueSecondHalf = maxIdxs[bidx] == SINGLE_QUEUE_SIZE;
        readIdxs[bidx] = enqueueSecondHalf ? 0 : SINGLE_QUEUE_SIZE;    // Read from the half we don't write to next
        endIdxs[bidx] = writeIdxs[bidx];                                        // Read up to where we last wrote
        writeIdxs[bidx] = enqueueSecondHalf ? SINGLE_QUEUE_SIZE : 0;   // Write to the half we didn't just write from
        maxIdxs[bidx] = SINGLE_QUEUE_SIZE * (enqueueSecondHalf ? 2 : 1); // Write until the end of the new write half
    }

    // Set up a queue.
    void Init(const uint3 blockIdx) const {
        const unsigned int bidx = blockIdx.y * RAY_BLOCKCOUNT + blockIdx.x;
        readIdxs[bidx] = SINGLE_QUEUE_SIZE; // Read from the second half
        endIdxs[bidx] = SINGLE_QUEUE_SIZE;  // Read 0 rays in the second half
        writeIdxs[bidx] = 0;                // Write to the first half
        maxIdxs[bidx] = SINGLE_QUEUE_SIZE;  // Write until the end of the first half
    }
};

__managed__ extern RayQueue rayQueue;

// Fire primary rays from the camera origin to the center of the plane
__global__ void DispatchRows();

// Trace a ray through the scene and calculate the returned color
__device__ void TraceRay(Ray& ray, int hitSide = HIT_FRONT_AND_BACK);

// Trace a ray through the scene and calculate a shadow hit. Return true if it hits an object.
__device__ bool TraceShadowRay(ShadowRay& ray, float3 n, float tMax = BIGFLOAT, int hitSide = HIT_FRONT_AND_BACK);