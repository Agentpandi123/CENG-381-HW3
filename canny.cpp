/*******************************************************************************
* --------------------------------------------
*(c) 2001 University of South Florida, Tampa
* Use, or copying without permission prohibited.
* PERMISSION TO USE
* In transmitting this software, permission to use for research and
* educational purposes is hereby granted.  This software may be copied for
* archival and backup purposes only.  This software may not be transmitted
* to a third party without prior permission of the copyright holder. This
* permission may be granted only by Mike Heath or Prof. Sudeep Sarkar of
* University of South Florida (sarkar@csee.usf.edu). Acknowledgment as
* appropriate is respectfully requested.
*
*  Heath, M., Sarkar, S., Sanocki, T., and Bowyer, K. Comparison of edge
*    detectors: a methodology and initial study, Computer Vision and Image
*    Understanding 69 (1), 38-54, January 1998.
*  Heath, M., Sarkar, S., Sanocki, T. and Bowyer, K.W. A Robust Visual
*    Method for Assessing the Relative Performance of Edge Detection
*    Algorithms, IEEE Transactions on Pattern Analysis and Machine
*    Intelligence 19 (12),  1338-1359, December 1997.
*  ------------------------------------------------------
*
* PROGRAM: canny_edge
* PURPOSE: This program implements a "Canny" edge detector. The processing
* steps are as follows:
*
*   1) Convolve the image with a separable gaussian filter.
*   2) Take the dx and dy the first derivatives using [-1,0,1] and [1,0,-1]'.
*   3) Compute the magnitude: sqrt(dx*dx+dy*dy).
*   4) Perform non-maximal suppression.
*   5) Perform hysteresis.
*
* The user must input three parameters. These are as follows:
*
*   sigma = The standard deviation of the gaussian smoothing filter.
*   tlow  = Specifies the low value to use in hysteresis. This is a
*           fraction (0-1) of the computed high threshold edge strength value.
*   thigh = Specifies the high value to use in hysteresis. This fraction (0-1)
*           specifies the percentage point in a histogram of the gradient of
*           the magnitude. Magnitude values of zero are not counted in the
*           histogram.
*
* NAME: Mike Heath
*       Computer Vision Laboratory
*       University of South Florida
*       heath@csee.usf.edu
*
* DATE: 2/15/96
*
* Modified: 5/17/96 - To write out a floating point RAW headerless file of
*                     the edge gradient "up the edge" where the angle is
*                     defined in radians counterclockwise from the x direction.
*                     (Mike Heath)
*******************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define VERBOSE 0

#define NOEDGE 255
#define POSSIBLE_EDGE 128
#define EDGE 0
#define BOOSTBLURFACTOR 90.0
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

int read_pgm_image(char *infilename, unsigned char **image, int *rows, int *cols);
int write_pgm_image(char *outfilename, unsigned char *image, int rows, int cols, char *comment, int maxval);

void canny(unsigned char *image, int rows, int cols, float sigma, float tlow, float thigh, unsigned char **edge, char *fname);
void gaussian_smooth(unsigned char *image, int rows, int cols, float sigma, short int *smoothedim);
void make_gaussian_kernel(float sigma, float *kernel, int *windowsize);
void derrivative_x_y(short int *smoothedim, int rows, int cols, short int *delta_x, short int *delta_y);
void magnitude_x_y(short int *delta_x, short int *delta_y, int rows, int cols, short int *magnitude);
void apply_hysteresis(short int *mag, unsigned char *nms, int rows, int cols, float tlow, float thigh, unsigned char *edge);

int main(int argc, char *argv[])
{
   char *infilename = NULL;  /* Name of the input image */
   char outfilename[128];    /* Name of the output "edge" image */
   unsigned char *image;     /* The input image */
   unsigned char *edge;      /* The output edge image */
   int rows, cols;           /* The dimensions of the image. */
   float sigma,              /* Standard deviation of the gaussian kernel. */
	 tlow,               /* Fraction of the high threshold in hysteresis. */
	 thigh;              /* High hysteresis threshold control. The actual
			        threshold is the (100 * thigh) percentage point
			        in the histogram of the magnitude of the
			        gradient image that passes non-maximal
			        suppression. */

   /****************************************************************************
   * Get the command line arguments.
   ****************************************************************************/
   if(argc < 5){
   fprintf(stderr,"\n<USAGE> %s image sigma tlow thigh [writedirim]\n",argv[0]);
      fprintf(stderr,"\n      image:      An image to process. Must be in ");
      fprintf(stderr,"PGM format.\n");
      fprintf(stderr,"      sigma:      Standard deviation of the gaussian");
      fprintf(stderr," blur kernel.\n");
      fprintf(stderr,"      tlow:       Fraction (0.0-1.0) of the high ");
      fprintf(stderr,"edge strength threshold.\n");
      fprintf(stderr,"      thigh:      Fraction (0.0-1.0) of the distribution");
      fprintf(stderr," of non-zero edge\n                  strengths for ");
      fprintf(stderr,"hysteresis. The fraction is used to compute\n");
      fprintf(stderr,"                  the high edge strength threshold.\n");
      fprintf(stderr,"      writedirim: Optional argument to output ");
      fprintf(stderr,"a floating point");
      fprintf(stderr," direction image.\n\n");
      exit(1);
   }

   infilename = argv[1];
   sigma = atof(argv[2]);
   tlow = atof(argv[3]);
   thigh = atof(argv[4]);

   /****************************************************************************
   * Read in the image. This read function allocates memory for the image.
   ****************************************************************************/
   if(VERBOSE) printf("Reading the image %s.\n", infilename);
   if(read_pgm_image(infilename, &image, &rows, &cols) == 0){
      fprintf(stderr, "Error reading the input image, %s.\n", infilename);
      exit(1);
   }

   /****************************************************************************
   * Perform the edge detection. All of the work takes place here.
   ****************************************************************************/
   if(VERBOSE) printf("Starting Canny edge detection.\n");
   canny(image, rows, cols, sigma, tlow, thigh, &edge, NULL);

   /****************************************************************************
   * Write out the edge image to a file.
   ****************************************************************************/
   sprintf(outfilename, "%s_s_%3.2f_l_%3.2f_h_%3.2f.pgm", infilename,
      sigma, tlow, thigh);
   if(VERBOSE) printf("Writing the edge in the file %s.\n", outfilename);
   char emptyString[] = "";
   if(write_pgm_image(outfilename, edge, rows, cols, emptyString, 255) == 0)
   {
      fprintf(stderr, "Error writing the edge image, %s.\n", outfilename);
      exit(1);
   }

   /****************************************************************************
   * Free dynamically allocated memory.
   ****************************************************************************/
   free(image);
   free(edge);

   return(0); /* exit cleanly */
}

/*******************************************************************************
* PROCEDURE: canny
* PURPOSE: To perform canny edge detection.
* NAME: Mike Heath
* DATE: 2/15/96
*******************************************************************************/
void canny(unsigned char *image, int rows, int cols, float sigma,
         float tlow, float thigh, unsigned char **edge, char *fname)
{
   unsigned char *nms = (unsigned char *) malloc(rows * cols * sizeof(unsigned char));   /* Points that are local maximal magnitude. */
   short int *smoothedim = (short int *) malloc(rows * cols * sizeof(short int));    /* The image after gaussian smoothing. */
   short int *delta_x = (short int *) malloc(rows * cols * sizeof(short int));       /* The first derivative image, x-direction. */
   short int *delta_y = (short int *) malloc(rows * cols * sizeof(short int));       /* The first derivative image, y-direction. */
   short int *magnitude = (short int *) malloc(rows * cols * sizeof(short int));     /* The magnitude of the gradient image. */
   unsigned char *edge_image = (unsigned char *) malloc(rows * cols * sizeof(unsigned char)); /* The final edge image. */

   /****************************************************************************
   * Perform gaussian smoothing on the image using the input standard
   * deviation.
   ****************************************************************************/
   if(VERBOSE) printf("Smoothing the image using a gaussian kernel.\n");
   gaussian_smooth(image, rows, cols, sigma, smoothedim);

   /****************************************************************************
   * Compute the first derivative in the x and y directions.
   ****************************************************************************/
   if(VERBOSE) printf("Computing the X and Y first derivatives.\n");
   derrivative_x_y(smoothedim, rows, cols, delta_x, delta_y);

   /****************************************************************************
   * Compute the magnitude of the gradient.
   ****************************************************************************/
   if(VERBOSE) printf("Computing the magnitude of the gradient.\n");
   magnitude_x_y(delta_x, delta_y, rows, cols, magnitude);

   /****************************************************************************
   * Perform non-maximal suppression.
   ****************************************************************************/
   if(VERBOSE) printf("Doing the non-maximal suppression.\n");

   /****************************************************************************
   * Use hysteresis to mark the edge pixels.
   ****************************************************************************/
   if(VERBOSE) printf("Doing hysteresis thresholding.\n");
   apply_hysteresis(magnitude, nms, rows, cols, tlow, thigh, edge_image);

   /****************************************************************************
   * Assign the output edge image.
   ****************************************************************************/
   *edge = edge_image;  // Point the output to the dynamically allocated edge_image array

   /****************************************************************************
   * Free dynamically allocated memory.
   ****************************************************************************/
   free(nms);
   free(smoothedim);
   free(delta_x);
   free(delta_y);
   free(magnitude);
}

/*******************************************************************************
* PROCEDURE: gaussian_smooth
* PURPOSE: Blur an image with a gaussian filter.
* NAME: Mike Heath
* DATE: 2/15/96
*******************************************************************************/
void gaussian_smooth(unsigned char *image, int rows, int cols, float sigma,
        short int *smoothedim)
{
   int r, c, rr, cc,     /* Counter variables. */
      windowsize,        /* Dimension of the gaussian kernel. */
      center;            /* Half of the windowsize. */
   float *tempim = (float *) malloc(rows * cols * sizeof(float)); /* Buffer for separable filter gaussian smoothing. */
   float kernel[21];     /* A one-dimensional gaussian kernel. Fixed size for sigma up to 4.0. */
   float dot,            /* Dot product summing variable. */
         sum;            /* Sum of the kernel weights variable. */

   /****************************************************************************
   * Create a 1-dimensional gaussian smoothing kernel.
   ****************************************************************************/
   if(VERBOSE) printf("   Computing the gaussian smoothing kernel.\n");
   make_gaussian_kernel(sigma, kernel, &windowsize);
   center = windowsize / 2;

   /****************************************************************************
   * Blur in the x - direction.
   ****************************************************************************/
   if(VERBOSE) printf("   Blurring the image in the X-direction.\n");
   for(r = 0; r < rows; r++){
      for(c = 0; c < cols; c++){
         dot = 0.0;
         sum = 0.0;
         for(cc = (-center); cc <= center; cc++){
            if(((c + cc) >= 0) && ((c + cc) < cols)){
               dot += (float)image[r*cols + (c+cc)] * kernel[center + cc];
               sum += kernel[center + cc];
            }
         }
         tempim[r*cols + c] = dot / sum;
      }
   }

   /****************************************************************************
   * Blur in the y - direction.
   ****************************************************************************/
   if(VERBOSE) printf("   Blurring the image in the Y-direction.\n");
   for(c = 0; c < cols; c++){
      for(r = 0; r < rows; r++){
         sum = 0.0;
         dot = 0.0;
         for(rr = (-center); rr <= center; rr++){
            if(((r + rr) >= 0) && ((r + rr) < rows)){
               dot += tempim[(r + rr) * cols + c] * kernel[center + rr];
               sum += kernel[center + rr];
            }
         }
         smoothedim[r*cols + c] = (short int)(dot*BOOSTBLURFACTOR/sum + 0.5);
      }
   }

   free(tempim);
}

/*******************************************************************************
* PROCEDURE: make_gaussian_kernel
* PURPOSE: Create a one dimensional gaussian kernel.
* NAME: Mike Heath
* DATE: 2/15/96
*******************************************************************************/
void make_gaussian_kernel(float sigma, float *kernel, int *windowsize)
{
   int i, center;
   float x, fx, sum = 0.0;

   *windowsize = 1 + 2 * ceil(2.5 * sigma);
   if (*windowsize > 21) {
      *windowsize = 21;  // Cap the window size to avoid index out of bounds
   }
   center = (*windowsize) / 2;

   if(VERBOSE) printf("      The kernel has %d elements.\n", *windowsize);

   for(i = 0; i < (*windowsize); i++){
      x = (float)(i - center);
      fx = exp(-0.5 * x * x / (sigma * sigma)) / (sigma * sqrt(2.0 * M_PI));
      kernel[i] = fx;
      sum += fx;
   }

   for(i = 0; i < (*windowsize); i++) kernel[i] /= sum;

   if(VERBOSE){
      printf("The filter coefficients are:\n");
      for(i = 0; i < (*windowsize); i++)
         printf("kernel[%d] = %f\n", i, kernel[i]);
   }
}

/*******************************************************************************
* FUNCTION: derrivative_x_y
* PURPOSE: Compute the first derivative of the image in both the x and y
* directions.
* NAME: Mike Heath
* DATE: 2/15/96
*******************************************************************************/
void derrivative_x_y(short int *smoothedim, int rows, int cols, short int *delta_x, short int *delta_y) {
    // Add logic to compute the derivative in x and y directions
    if (VERBOSE) printf("Computing the first derivative in the X and Y directions.\n");
    // Dummy implementation: Fill delta_x and delta_y with zeros
    memset(delta_x, 0, rows * cols * sizeof(short int));
    memset(delta_y, 0, rows * cols * sizeof(short int));
}

/*******************************************************************************
* FUNCTION: magnitude_x_y
* PURPOSE: Compute the magnitude of the gradient image.
* NAME: Mike Heath
* DATE: 2/15/96
*******************************************************************************/
void magnitude_x_y(short int *delta_x, short int *delta_y, int rows, int cols, short int *magnitude) {
    // Add logic to compute the magnitude from the x and y derivatives
    if (VERBOSE) printf("Computing the magnitude of the gradient.\n");
    // Dummy implementation: Fill magnitude with zeros
    memset(magnitude, 0, rows * cols * sizeof(short int));
}

/*******************************************************************************
* FUNCTION: apply_hysteresis
* PURPOSE: Perform hysteresis thresholding on the gradient magnitude image.
* NAME: Mike Heath
* DATE: 2/15/96
*******************************************************************************/
void apply_hysteresis(short int *mag, unsigned char *nms, int rows, int cols, float tlow, float thigh, unsigned char *edge) {
    // Add logic for edge detection hysteresis
    if (VERBOSE) printf("Performing hysteresis thresholding.\n");
    // Dummy implementation: Fill edge with NOEDGE
    memset(edge, NOEDGE, rows * cols * sizeof(unsigned char));
}

/*******************************************************************************
* FUNCTION: read_pgm_image
* PURPOSE: Read a PGM image from a file.
* NAME: Mike Heath
* DATE: 2/15/96
*******************************************************************************/
int read_pgm_image(char *infilename, unsigned char **image, int *rows, int *cols) {
    // Add logic to read a PGM image from file
    if (VERBOSE) printf("Reading the PGM image from file.\n");
    // Dummy implementation: Return failure
    return 0; // Failure, needs actual implementation
}

/*******************************************************************************
* FUNCTION: write_pgm_image
* PURPOSE: Write a PGM image to a file.
* NAME: Mike Heath
* DATE: 2/15/96
*******************************************************************************/
int write_pgm_image(char *outfilename, unsigned char *image, int rows, int cols, char *comment, int maxval) {
    // Add logic to write a PGM image to file
    if (VERBOSE) printf("Writing the PGM image to file.\n");
    // Dummy implementation: Return failure
    return 0; // Failure, needs actual implementation
}
