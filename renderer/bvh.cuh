// Maximum elements per BVH node
#pragma once
#include "mesh.cuh"
#include "rays.cuh"
#define BVH_MAX_ELEMENTS 8
#define BVH_MAX_DEPTH 64

class Mesh;

/// Class that can represent a mesh BVH
class BVH {
protected:
    struct BVHNode {
        bool leaf;

        // Start and count offset of elements in this node
        size_t elemStartIndex;
        size_t elemCount;

        // Index in node list of child nodes (if this is not a leaf node)
        size_t leftChild;
        size_t rightChild;

        // Index of the node after all descendent nodes. Either the node's sibling, or some ancestor's sibling, or nothing.
        size_t nextNode;

        // Bounding box for this node
        Box box;

        BVHNode(const size_t elemStartIndex, const size_t elemCount, const Box& box)
            : leaf(false), elemStartIndex(elemStartIndex), elemCount(elemCount), leftChild(0), rightChild(0), nextNode(0), box(box) {}

        BVHNode& operator=(const BVHNode & other) = default;
        BVHNode(): leaf(false), elemStartIndex(0), elemCount(0), leftChild(0), rightChild(0), nextNode(0), box() {}
    };

    Mesh* mesh = nullptr; // The mesh we build from
    size_t *elements = nullptr; // Elements are represented with offsets from starting object

    BVHNode* nodes = nullptr;
    size_t nodeCount = 0;

    // Get the center of an element along an axis.
    [[nodiscard]] float GetElementCenter(size_t elementId, unsigned int axis) const;

    // Get the bounds of an element along an axis.
    [[nodiscard]] Box GetElementBounds(size_t elementId) const;

    // Swap two elements
    void SwapElements(const size_t i, const size_t j) const {
        const size_t t = elements[i];
        elements[i] = elements[j];
        elements[j] = t;
    }

    // Get the surface area of a box from its position coords.
    [[nodiscard]] inline static float SurfaceArea(const Box& box) {
        const float l = box.pmax.x - box.pmin.x; const float w = box.pmax.y - box.pmin.y; const float h = box.pmax.z - box.pmin.z;
        return 2 * l * w + 2 * l * h + 2 * w * h;
    };

    // Splits a node if necessary, then recurses to child nodes.
    void SplitOrFinish(BVHNode* nodeList, size_t nodeId);

    // Calculate the maximum depth of the tree
    size_t CalculateDepth(const size_t nodeId = 0) const {
        const BVHNode &node = nodes[nodeId];
        if (node.leaf)
            return 1;
        return 1 + max(CalculateDepth(node.leftChild), CalculateDepth(node.rightChild));
    }

public:
    // Construct a BVH
    void ConstructBVH(Mesh* newMesh);

    [[nodiscard]] __host__ __device__ size_t GetNodeCount() const { return nodeCount; }
    [[nodiscard]] __host__ __device__ bool NodeIsLeaf(const size_t nodeId) const { return nodes[nodeId].leaf; }
    [[nodiscard]] __host__ __device__ size_t* GetNodeElements(const size_t nodeId) const { return elements + nodes[nodeId].elemStartIndex; }
    [[nodiscard]] __host__ __device__ size_t GetNodeElementCount(const size_t nodeId) const { return nodes[nodeId].elemCount; }

    [[nodiscard]] __host__ __device__ size_t GetNodeLeftChild(const size_t nodeId) const { return nodes[nodeId].leftChild; }
    [[nodiscard]] __host__ __device__ size_t GetNodeRightChild(const size_t nodeId) const { return nodes[nodeId].rightChild; }
    [[nodiscard]] __host__ __device__ size_t GetNextNode(const size_t nodeId) const { return nodes[nodeId].nextNode; }

    [[nodiscard]] __host__ __device__ const Box& GetNodeBoundingBox(const size_t nodeId) const { return nodes[nodeId].box; }
};