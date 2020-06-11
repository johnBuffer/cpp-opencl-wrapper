typedef char           int8_t;
typedef unsigned char  uint8_t;
typedef unsigned short uint16_t;
typedef int            int32_t;
typedef unsigned int   uint32_t;


__constant uint8_t MAX_STACK_SIZE = 10u;


typedef struct __attribute__ ((packed)) _Mutation {
	uint8_t  needed;
	uint8_t  value;
	uint32_t node_id;
} Mutation;


__kernel void mutate(
		global uint8_t* svo,
		global Mutation* diffs
	)
{
	for (uint8_t i = 0; i < MAX_STACK_SIZE; ++i) {
		const Mutation mut = diffs[i];
		if (mut.needed) {
			svo[mut.node_id] = mut.value;
		}
	}
}
