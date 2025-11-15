#include "bvh.cuh"

#include <cstdio>

#include "mesh.cuh"

void BVH::ConstructBVH(Mesh *newMesh) {
    mesh = newMesh;

    // Set up a managed pointer to each element
    cudaMallocManaged(&elements, mesh->NF() * sizeof(size_t));
    for (int i = 0; i < mesh->NF(); i++)
        elements[i] = i;

    // Allocate enough space for the maximum of 2n-1 nodes
    auto *tempNodes = new BVHNode[2 * mesh->NF()];

    // Create a root node, and then recurse to all elements.
    tempNodes[0] = BVHNode(0, mesh->NF(), mesh->GetBoundingBox());
    nodeCount = 0;
    SplitOrFinish(tempNodes, 0);
    // printf("Final BVH has %lu nodes encompassing %lu elements. Maximum node count would be %lu.\n", nodeCount, mesh->NF(), 2 * mesh->NF());

    // Transfer BVH into managed memory
    cudaMallocManaged(&nodes, nodeCount * sizeof(BVHNode));
    memcpy(nodes, tempNodes, nodeCount * sizeof(BVHNode));

    // Update max depth if necessary
    const size_t depth = CalculateDepth();
    // printf("BVH depth: %lu\n", depth);
    if (depth > BVH_MAX_DEPTH) {
        printf("ERROR: BVH had depth %lu, but maximum depth was %d!\n", depth, BVH_MAX_DEPTH);
        throw std::exception{};
    }
}

void BVH::SplitOrFinish(BVHNode* nodeList, const size_t nodeId) {
    nodeCount++;

    // Setup info
    constexpr int binCount = 32;
    constexpr float binSize = 1.0f / static_cast<float>(binCount);
    struct AreaSplitResult {
        // Definition of this split
        unsigned int axis = 0;
        unsigned int bin = 0;
        // Traversal cost
        float cost = BIGFLOAT;
        // Boxes for this split
        Box lbox{};
        Box rbox{};
    };

    // Information about this node
    BVHNode &node = nodeList[nodeId];
    if (node.elemCount == 1) {
        node.leaf = true;
        node.nextNode = nodeCount;
        return;
    }

    const Box& box = node.box;
    const float outer = 1.0f / SurfaceArea(box);

    const float3 binDim = (node.box.pmax - node.box.pmin) * binSize;

    // Result of our best split so far
    AreaSplitResult best;

    // Loop through each axis and bin, and create a bounding box based on the split
    for (unsigned int axis = 0; axis < 3; axis++) {
        for (int bin = 1; bin < binCount; bin++) {
            // Find the split point
            float splitPoint = ref(box.pmin, axis) + static_cast<float>(bin) * ref(binDim, axis);

            // Reset bounds of other boxes

            // Create a result for this split
            AreaSplitResult result;
            result.axis = axis;
            result.bin = bin;
            result.lbox.Init();
            result.rbox.Init();
            size_t lcount = 0, rcount = 0;

            // Loop through each element
            for (size_t i = 0; i < node.elemCount; i++) {
                // Figure out which split it belongs to and get bounds
                const float center = GetElementCenter(node.elemStartIndex + i, axis);
                const Box bounds = GetElementBounds(node.elemStartIndex + i);

                // Extend either box
                if (center <= splitPoint) {
                    result.lbox += bounds;
                    lcount++;
                } else {
                    result.rbox += bounds;
                    rcount++;
                }
            }

            // Calculate surface area and cost, assuming C_triangle = C_node = 1
            const float larea = SurfaceArea(result.lbox);
            const float rarea = SurfaceArea(result.rbox);

            const float lcost = (larea * outer) * lcount;
            const float rcost = (rarea * outer) * rcount;
            result.cost = 1 + lcost + rcost;

            if (result.cost < best.cost)
                best = result;
        }
    }

    // Calculate cost of keeping this as a leaf node vs converting to a child node
    // assume C_Triangle = C_Node = 1, and C_leaf = # triangles
    // Split if it would be cheaper or if we would have too many elements otherwise
    if (best.cost > node.elemCount && node.elemCount <= BVH_MAX_ELEMENTS) {
        node.leaf = true;
        node.nextNode = nodeCount;
        return; // Do no further splits
    }

    // Otherwise, partition elements based on the chosen split.
    node.leaf = false;
    size_t i = 0, j = node.elemCount;
    const float splitPoint = ref(box.pmin, best.axis) + static_cast<float>(best.bin) * ref(binDim, best.axis);
    while (i < j) {
        const float center = GetElementCenter(node.elemStartIndex + i, best.axis);
        if (center <= splitPoint) {
            i++;
        } else {
            j--;
            SwapElements(node.elemStartIndex + i, node.elemStartIndex + j);
        }
    }

    // I is our split point. Partition into one box with i elements, and one with node.elemCount-i elements.
    // We will partition depth first.
    node.leftChild = nodeCount;
    nodeList[node.leftChild] = BVHNode(node.elemStartIndex, i, best.lbox);
    SplitOrFinish(nodeList, nodeCount); // This will increment nodeCount based on the amount of created nodes

    node.rightChild = nodeCount;
    nodeList[node.rightChild] = BVHNode(node.elemStartIndex + i, node.elemCount - i, best.rbox);
    SplitOrFinish(nodeList, nodeCount);

    node.nextNode = nodeCount;
}


float BVH::GetElementCenter(const size_t elementId, const unsigned int axis) const {
    float sum = 0.0f;
    for (int i = 0; i < 3; i++) {
        sum += ref(*mesh->V(ref(*mesh->F(elements[elementId]), i)), axis);
    }
    return sum * 0.3333333333333333f;
}

Box BVH::GetElementBounds(const size_t elementId) const {
    Box box{};
    box.Init();
    for (int i = 0; i < 3; i++) {
        box += *mesh->V(ref(*mesh->F(elements[elementId]), i));
    }
    return box;
}