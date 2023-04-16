// LeNet-5-like inference c model

#include <stdlib.h>
#include <stdio.h>
#include "conv.h"

// open weight and bias
void get_wb();

// convolution
void convl (double *fm_in, double *kn, double *bi, double *fm_out, int s_fm, int s_kn, int n_fm_in, int n_fm_out);
void convl_valid (double *fm_in, double *kn, double *bi, double *fm_out, int s_fm, int s_kn, int n_fm_in, int n_fm_out);

// Relu
void relu (double *fm_in, double *fm_out, int s_fm, int n_fm_in);
//fc
void fc1(double *value, double *weight, double *bi, double *fm_out);

// subsampling
void subsp (double *fm_in, double *kn, double *fm_out, int s_fm, int n_fm);

double conv1_weight[5][5][1][32];
double conv1_biase[32];
double conv2_weight[5][5][32][64];
double conv2_biase[64];
double fc1_weight[49][64][512];			// 100MB
double temp[3136][512];
double fc1_temp[64][49][512];
double fc1_biase[512];
double fc2_weight[512][10];
double fc2_biase[10];

//Layer 0
double *we_l0;
double *bi_l0;
double *fm_l1;
double *re_l1;
double *ss_l1;
//Layer 1
double *we_l1;
double *bi_l1;
double *fm_l2;
double *re_l2;
double *ss_l2;
//Layer 2
double *we_l2;
double *bi_l2;
double *fm_l3;
double *re_l3;
//Layer 3
double *we_l3;
double *bi_l3;
double *fm_l4;

//Simulation
int KE_BW;		// Kernal	
int OP_BW;		// Operation
int IM_BW;		// Image

int main (int argc, char* argv[]) {
	// input arguments

  // Layer 0
	double *image;

	double max;
	int max_idx;

	// Simulation part
	// OP_BW = 2;
  // KE_BW = 1;
  // IM_BW = 2;

	image = (double *) malloc(sizeof(double)*S_I*S_I);								              // 28 * 28
	
	we_l0 = (double *) malloc(sizeof(double)*K_CONV_1*K_CONV_1*N_I*N_CONV_1);		    // 5 * 5 * 1 * 32
	bi_l0 = (double *) malloc(sizeof(double)*N_CONV_1);	          							    // 32
	fm_l1 = (double *) malloc(sizeof(double)*S_CONV_1*S_CONV_1*N_CONV_1);	  		    // 28 * 28 * 32
	re_l1 = (double *) malloc(sizeof(double)*S_CONV_1*S_CONV_1*N_CONV_1);		  	    // 28 * 28 * 32 
	ss_l1 = (double *) malloc(sizeof(double)*S_CONV_1*S_CONV_1*N_CONV_1/4);			    // ( 28 * 28 * 32 ) / 4
	
	we_l1 = (double *) malloc(sizeof(double)*K_CONV_2*K_CONV_2*N_CONV_1*N_CONV_2);	// 5 * 5 * 32 * 64
	bi_l1 = (double *) malloc(sizeof(double)*N_CONV_2);							              	// 64
	fm_l2 = (double *) malloc(sizeof(double)*S_CONV_2*S_CONV_2*N_CONV_2);			      // 14 * 14 * 64
	re_l2 = (double *) malloc(sizeof(double)*S_CONV_2*S_CONV_2*N_CONV_2);			      // 14 * 14 * 64
	ss_l2 = (double *) malloc(sizeof(double)*S_CONV_2*S_CONV_2*N_CONV_2/4);			    // ( 14 * 14 * 64 ) / 4
	
	we_l2 = (double *) malloc(sizeof(double)*K_FC_1*K_FC_1*N_CONV_2*N_FC_1);		    // 7 * 7 * 64 * 512
	bi_l2 = (double *) malloc(sizeof(double)*N_FC_1);								                // 512
	fm_l3 = (double *) malloc(sizeof(double)*S_FC_1*S_FC_1*N_FC_1);					        // 1 * 1 * 512
	re_l3 = (double *) malloc(sizeof(double)*S_FC_1*S_FC_1*N_FC_1);					        // 1 * 1 * 512
	
	we_l3 = (double *) malloc(sizeof(double)*K_FC_2*K_FC_2*N_FC_1*N_FC_2);			    // 1 * 1 * 512 * 10
	bi_l3 = (double *) malloc(sizeof(double)*N_FC_1);								                // 10
	fm_l4 = (double *) malloc(sizeof(double)*S_FC_2*S_FC_2*N_FC_2);					        // 1 * 1 * 10

	// Image load
	int i;
	unsigned char temp, temp_;

  int imnum = 0;
  char imagenum[100];
  
	for(imnum=0;imnum<20;imnum++)
  {  
    sprintf(imagenum,"../DB/MNIST/image_%d.raw",imnum);
	  
    FILE* fd = fopen(imagenum, "rb");
	  for (i = 0; i < S_I * S_I; i++) 
    {
		  temp_ = fread(&temp, 1, 1, fd);
		  image[i] = (double) temp;
		  image[i] = image[i]-127.5;
		  image[i] = image[i]/255;
	  //image[i] = (double)((int)(image[i]*(double)(1<<IM_BW))) / (double)(1<<IM_BW);
	  }
	  fclose(fd);

	  // Variables
	  i = 0;
	  //printf("start! \n");

	  // Weight
	  get_wb();

	  /////////////
	  // Layer 0 //
	  /////////////
	  convl(image, we_l0, bi_l0, fm_l1, S_CONV_1, K_CONV_1, N_I, N_CONV_1);
	  relu(fm_l1, re_l1, S_CONV_1, N_CONV_1);
	  subsp(re_l1, we_l0, ss_l1, S_CONV_1, N_CONV_1);
  	//printf("layer 0 is finished! \n");

  	/////////////
  	// Layer 1 //
  	/////////////
  	convl(ss_l1, we_l1, bi_l1, fm_l2, S_CONV_2, K_CONV_2, N_CONV_1, N_CONV_2);
    relu(fm_l2, re_l2, S_CONV_2, N_CONV_2);
  	subsp(re_l2, we_l1, ss_l2, S_CONV_2, N_CONV_2);
  	//printf("layer 1 is finished! \n");
	
  	/////////////
  	// Layer 2 //
  	/////////////
  	convl_valid(ss_l2, we_l2, bi_l2, fm_l3, K_FC_1, K_FC_1, N_CONV_2, N_FC_1);
  	relu(fm_l3, re_l3, S_FC_1, N_FC_1);
  	//printf("layer 2 is finished! \n");
	
  	/////////////
  	// Layer 3 //
  	/////////////
  	convl(re_l3, we_l3, bi_l3, fm_l4, S_FC_2, K_FC_2, N_FC_1, N_FC_2);
  	//printf("layer 3 is finished! \n");

  	max = fm_l4[0];
  	max_idx = 0;
	
  	for (i = 1; i < N_FC_2; i++) {
	  	if (fm_l4[i] > max) {
		  	max = fm_l4[i];
			  max_idx = i;
	  	  }
	    }
	  //free(fm_l4);

	  FILE *save = fopen("./MNIST.txt","a");
	  fprintf(save,"%d \n",max_idx);
	  fclose(save);

    printf("%d\n",max_idx);
  }
  free(image);
  free(we_l0);
  free(bi_l0);
  free(fm_l1);
  free(we_l1);
  free(bi_l1);
	free(fm_l2);
  free(we_l2);
  free(bi_l2);
	free(fm_l3);
  free(we_l3);
  free(bi_l3);

	return 0;
}

/////////////////////////
// get weight and bias //
/////////////////////////
void get_wb()
{
	int i,j,k,l;
  int ge;

	FILE *conv1_w;
	conv1_w = fopen("co1_we.txt", "r");
	for (i = 0; i < (K_CONV_1*K_CONV_1*N_I*N_CONV_1); i++)
	{
		ge=fscanf(conv1_w, "%lf", &we_l0[i]);
	}
	fclose(conv1_w);
	
	FILE *conv1_b;
	conv1_b = fopen("conv1_biases.txt", "r");
	for (i = 0; i < 32; i++)
	{
		ge=fscanf(conv1_b, "%lf", &bi_l0[i]);
	}
	fclose(conv1_b);

	FILE *conv2_w;
	conv2_w = fopen("co2_we.txt", "r");
	for (i = 0; i < (K_CONV_2*K_CONV_2*N_CONV_1*N_CONV_2); i++)
	{
		ge=fscanf(conv2_w, "%lf", &we_l1[i]);
	}
	fclose(conv2_w);

	FILE *conv2_b;
	conv2_b = fopen("conv2_biases.txt", "r");
	for (i = 0; i < 64; i++)
	{
		ge=fscanf(conv2_b, "%lf", &bi_l1[i]);
	}
	fclose(conv2_b);

	FILE *fc1_w;
	fc1_w = fopen("fc1_we.txt", "r");
  for (i = 0; i < (K_FC_1*K_FC_1*N_CONV_2*N_FC_1); i++)
	{
		ge= fscanf(fc1_w, "%lf", &we_l2[i]);
	}
	fclose(fc1_w);

	FILE *fc1_b;
	fc1_b = fopen("fc1_biases.txt", "r");
	for (i = 0; i < 512; i++)
	{
		ge=fscanf(fc1_b, "%lf", &bi_l2[i]);
	}
	fclose(fc1_b);

  FILE *fc2_w;
	fc2_w = fopen("fc2_we.txt", "r");
	for (i = 0; i < 5120; i++)
	{
			ge=fscanf(fc2_w, "%lf", &we_l3[i]);
	}
	fclose(fc2_w);

	FILE *fc2_b;
	fc2_b = fopen("fc2_biases.txt", "r");
	for (i = 0; i < 10; i++)
	{
		ge=fscanf(fc2_b, "%lf", &bi_l3[i]);
	}
	fclose(fc2_b);
}

/////////////////
// Convolution //
/////////////////
void convl (double *fm_in, double *kn, double *bi, double *fm_out, int s_fm, int s_kn, int n_fm_in, int n_fm_out) {
	double max =0;
	double min =0;
	double p_fm;			// feature map value corresponding to a pixel
	double weight;			// kernel weight
	double prod;			// product
	double p_sum;			// partial sum

	int i, j, k, x, y, z;
	int xt, yt;
	int temp=0;
  int jump=0;
	//printf("conv starts!\n");
	for (k = 0; k < n_fm_out; k++) {
		for (j = 0; j < s_fm; j++) {
			for (i = 0; i < s_fm; i++) {
				p_sum = 0;
				for (z = 0; z < n_fm_in; z++) {
					for (y = 0; y < s_kn; y++) {
						for (x = 0; x < s_kn; x++) {
							// load pixel value
								xt = i - ((s_kn-1)/2) + x;
							    yt = j - ((s_kn-1)/2) + y;
							if ((xt < 0) || (yt < 0) || (xt > s_fm-1) || (yt > s_fm-1)) {
								p_fm = 0;		// zero padding
							} else {
								p_fm = fm_in[P_IDX(xt, yt, z, s_fm, s_fm*s_fm)];
							}
								// load kernel value
								weight = kn[K_IDX(x, y, z, k, s_kn, s_kn*s_kn, s_kn*s_kn*n_fm_in)];
							/* Truncation*/
							//	p_fm = (double)((int)(p_fm*(double)(1<<IN_BW))) / (double)(1<<IN_BW);
							//	weight = (double)((int)(weight*(double)(1<<KE_BW))) / (double)(1<<KE_BW);
							// compute product
							prod = weight * p_fm;
							p_sum += prod;

							/* Truncation*/
							// p_sum = (double)((int)(p_sum*(double)(1<<OP_BW))) / (double)(1<<OP_BW);
						}
					}
				}

		//		bi[k] = (double)((int)(bi[k]*(double)(1<<KE_BW))) / (double)(1<<KE_BW);
		     	p_sum = bi[k]+p_sum;
		//		p_sum = (double)((int)(p_sum*(double)(1<<OP_BW))) / (double)(1<<OP_BW);
				fm_out[P_IDX(i, j, k, s_fm, s_fm*s_fm)] = p_sum;
				
			}
		}
	}

}

//////////
// Relu //
//////////
void relu (double *fm_in, double *fm_out, int s_fm, int n_fm_in)
{
	int i,j,k;

	for (k=0; k<n_fm_in; k++)
	{
		for (j=0; j<s_fm; j++)
		{
			for (i=0; i<s_fm; i++)
			{
				if (fm_in[P_IDX(i,j,k, s_fm, s_fm*s_fm)]<=0)
				{
					fm_out[P_IDX(i,j,k, s_fm, s_fm*s_fm)] = 0;
				}
				else
        {
					fm_out[P_IDX(i,j,k, s_fm, s_fm*s_fm)]=fm_in[P_IDX(i,j,k, s_fm, s_fm*s_fm)];
				}
			}
		}
	}
}

void convl_valid (double *fm_in, double *kn, double *bi, double *fm_out, int s_fm, int s_kn, int n_fm_in, int n_fm_out) 
{
	double p_fm;			// feature map value corresponding to a pixel
	double weight;	   		// kernel weight
	double prod;			// product
	double p_sum;			// partial sum

	int i, j, k, x, y, z;
	int xt, yt;

  //printf("conv starts!\n");
	for (k = 0; k < n_fm_out; k++) {
		for (j = 0; j < (s_fm-s_kn+1); j++) {
			for (i = 0; i < (s_fm-s_kn+1); i++) {
				p_sum = 0;
				for (z = 0; z < n_fm_in; z++) {
					for (y = 0; y < s_kn; y++) {
						for (x = 0; x < s_kn; x++) {
							// load pixel value
							xt = i + x;
							yt = j + y;
							//printf("ijk,xyz: %d %d %d, %d %d %d ", i, j, k, x, y, z);
							//printf("good %d  ", P_IDX(xt, yt, z, s_fm, s_fm*s_fm));
							p_fm = fm_in[P_IDX(xt, yt, z, s_fm, s_fm*s_fm)];
							//p_fm = fm_in[K_IDX(x, y, z, 0, s_kn, s_kn*s_kn, s_kn*s_kn*n_fm_in)];
							//printf("1");
							// load kernel value
							weight = kn[K_IDX(x, y, z, k, s_kn, s_kn*s_kn, s_kn*s_kn*n_fm_in)];
              //printf("2");

							// compute product
							prod = weight * p_fm;
							p_sum += prod;
							//printf("3\n");
						}
					}
				}
        p_sum = p_sum+bi[k];
				fm_out[P_IDX(i, j, k, (s_fm-s_kn+1), ((s_fm-s_kn+1)*(s_fm-s_kn+1)))] = p_sum;
      }
		}
	}
}

/////////////////
// Subsampling //
/////////////////
void subsp (double *fm_in, double *kn, double *fm_out, int s_fm, int n_fm) {
	double p00, p01, p10, p11;
	double max;

	int i, j, k;
	int xt0, yt0, xt1, yt1;
	for (k = 0; k < n_fm; k++) {
		for (j = 0; j < s_fm; ) {
			for (i = 0; i < s_fm; ) {
				xt0 = i; xt1 = i+1;
				yt0 = j; yt1 = j+1;

				// p00 load
				p00 = fm_in[P_IDX(xt0, yt0, k, s_fm, s_fm*s_fm)];

				// p01 load
				if (yt1 == s_fm) p01 = 0;
				else p01 = fm_in[P_IDX(xt0, yt1, k, s_fm, s_fm*s_fm)];

				// p10 load
				if (xt1 == s_fm) p10 = 0;
				else p10 = fm_in[P_IDX(xt1, yt0, k, s_fm, s_fm*s_fm)];

				// p11 load
				if ((xt1 == s_fm) || (yt1 == s_fm)) p11 = 0;
				else p11 = fm_in[P_IDX(xt1, yt1, k, s_fm, s_fm*s_fm)];

				// max pooling
				max = p00;
				if (p01 > max) max = p01;
				if (p10 > max) max = p10;
				if (p11 > max) max = p11;

				// save
				fm_out[P_IDX(i/2, j/2, k, s_fm/2, s_fm*s_fm/4)] = max;

				i += 2;
			}
			j += 2;
		}
	}
}
