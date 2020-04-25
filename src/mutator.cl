typedef char           int8_t;
typedef unsigned char  uint8_t;
typedef unsigned short uint16_t;
typedef int            int32_t;
typedef unsigned int   uint32_t;


typedef struct Node
{
    uint8_t  child_mask;
	uint8_t  leaf_mask;
	uint32_t child_offset;
	uint8_t  reflective_mask;
	uint8_t  emissive;
} Node;


__kernel void mutate(
		__global Node* svo,
		uint32_t node_index,
		uint8_t  child_index,
		uint8_t  value
	)
{
	if (value) {
		svo[node_index].emissive |= (1u << child_index);
	} else {
		const uint8_t mask = 255 ^ (1u << child_index);
		svo[node_index].emissive &= mask;
	}
}
