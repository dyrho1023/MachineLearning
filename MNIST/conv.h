#define	S_I 28				// Image size
#define	N_I 1				// Image channel
#define	N_LB 10				// Number of labels

#define S_CONV_1 28			// Size of feature maps
#define N_CONV_1 32			// Number of feature maps
#define K_CONV_1 5			// Kernel size
//#define ST_CONV_1 1		// Stride

#define S_CONV_2 14
#define N_CONV_2 64
#define K_CONV_2 5
//#define ST_CONV_2 1

#define S_FC_1 1
#define N_FC_1 512
#define K_FC_1 7
//#define ST_FC_1 1

#define S_FC_2 1
#define N_FC_2 10
#define K_FC_2 1
//#define ST_FC_2 1

#define P_IDX(x, y, z, width, area) ( (x) + ((y) * (width)) + ((z) * (area)) )
#define K_IDX(x, y, z, k, width, area, vol) ( (x) + ((y) * (width)) + ((z) * (area)) + ((k) * (vol)) )

// Fractional part
//#define IN_BW 10	// Input 
//#define KE_BW 10		// Kernal
//#define OP_BW 10		// Operation
//#define IM_BW 2   // Image
