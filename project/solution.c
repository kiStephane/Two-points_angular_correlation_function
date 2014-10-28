/*
  mpicc -O3 -std=c99 solution.c -o galaxyz
  Run sequentially with srun -n 5 galaxyz test.txt test_rand.txt outfile.txt
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <mpi.h>

#define binsperdegree 4                      /* Nr of bins per degree */
#define totaldegrees 64                       /* Nr of degrees */


/* Count how many lines the input file has */
int count_lines (FILE *infile) {
  char readline[80];      /* Buffer for file input */
  int lines=0;
  while( fgets(readline,80,infile) != NULL ) lines++;
  rewind(infile);  /* Reset the file to the beginning */
  return(lines);
}

/* Read input data from the file, convert to cartesian coordinates 
   and write them to arrays x, y and z */
void read_data(FILE *infile, int n, float *x, float *y, float *z) {
  char readline[80];      /* Buffer for file input */
  float ra, dec, theta, phi, dpi;
  int i=0;
  dpi = acos(-1.0);
  while( fgets(readline,80,infile) != NULL )  /* Read a line */
    {
      sscanf(readline,"%f %f",&ra, &dec);  /* Read a coordinate pair */
      /* Debug */
      /*if ( i == 0 ) printf("     first item: %3.6f %3.6f\n",ra,dec); */
      /* Convert to cartesian coordinates */
      phi   = ra * dpi/180.0;
      theta = (90.0-dec)*dpi/180;
      x[i] = sinf(theta)*cosf(phi);
      y[i] = sinf(theta)*sinf(phi);
      z[i] = cosf(theta);
      /*
	NOTE: there was a bug in the code here. The following ststement
	is not correct.
	z[i] = sinf(cosf(theta));
      */
      ++i;
    }
  /* Debug */
  /* printf("      last item: %3.6f %3.6f\n\n",ra,dec); */
  fclose(infile);
}

/*
  Compute the angle between two observations p and q and add it to the histogram
*/
void add_histogram (float px, float py, float pz, 
		    float qx, float qy, float qz, long int *histogram, 
		    const float pi, const float costotaldegrees) {
  float theta;
    float degreefactor = 180.0/pi*binsperdegree;
    int bin;
  theta = px*qx + py*qy + pz*qz;
  if ( theta >= costotaldegrees ) {   /* Skip if theta < costotaldegrees */
    if ( theta > 1.0 ) theta = 1.0;
    /* Calculate which bin to increment */
    /* histogram [(int)(acos(theta)*180.0/pi*binsperdegree)] += 1L; */
    bin = (int)(acosf(theta)*degreefactor); 
    histogram[bin]++;
  }
}



int main(int argc, char *argv[])
{
    int size_real;
    int i,j;
    int np, id, err;
    MPI_Status status;
    int tag = 777;
    long int TotalCountDD, TotalCountDR, TotalCountRR; /* Counters */
    float pi, costotaldegrees;
    int nr_of_bins = binsperdegree*totaldegrees;  /* Total number of bins */
    double starttime, stoptime;
    long int *histogramDD, *histogramDR, *histogramRR; /* Arrays for histograms */
    float *xd_real, *yd_real, *zd_real;         /* Arrays for real data */
    
    /* Check that we have 4 command line arguments */
    if ( argc != 4 ) {
        printf("Usage: %s real_data sim_data output_file\n", argv[0]);
        return(0);
    }
    
    histogramDD = (long int *)calloc( nr_of_bins+1, sizeof(long int));
    histogramDR = (long int *)calloc( nr_of_bins+1, sizeof(long int));
    histogramRR = (long int *)calloc( nr_of_bins+1, sizeof(long int));
    
    /* Initialize the histograms to zero */
    for ( int i = 0; i <= nr_of_bins; ++i )
    {
        histogramDD[i] = 0L;
        histogramDR[i] = 0L;
        histogramRR[i] = 0L;
    }
    
    err = MPI_Init(&argc, &argv);
    if (err != MPI_SUCCESS) {
        printf("MPI_init failed!\n");
        exit(1);
    }
    
    err = MPI_Comm_size(MPI_COMM_WORLD, &np);
    if (err != MPI_SUCCESS) {
        printf("MPI_Comm_size failed!\n");
        exit(1);
    }
    
    err = MPI_Comm_rank(MPI_COMM_WORLD, &id);
    if (err != MPI_SUCCESS) {
        printf("MPI_Comm_rank failed!\n");
        exit(1);
    }
    
    /* Check that we run on at least two processors */
    if (np < 2) {
        printf("You have to use at least 2 processes to run this program\n");
        MPI_Finalize();	       /* Quit if there is only one processor */
        exit(0);
    }
    
    pi = acosf(-1.0);
    costotaldegrees = (float)(cos(totaldegrees/180.0*pi));
    
    if (id == 0) {
        int Nooflines_Real;  /* Nr of lines in real data */
        FILE *infile,*outfile;  /* Input and output files */
        
        starttime = MPI_Wtime();
        
        /* Open the real data input file */
        infile = fopen(argv[1],"r");
        if ( infile == NULL ) {
            printf("Unable to open %s\n",argv[1]);
            return(0);
        }
        
        /* Count how many lines the input file has */
        Nooflines_Real = count_lines(infile);
        printf("%s contains %d lines\n", argv[1], Nooflines_Real);
        
        /* Allocate arrays for x, y and z values */
        xd_real = (float *)calloc( Nooflines_Real, sizeof(float) );
        yd_real = (float *)calloc( Nooflines_Real, sizeof(float) );
        zd_real = (float *)calloc( Nooflines_Real, sizeof(float) );
        
        /* Read the file with real input data */
        read_data(infile, Nooflines_Real, xd_real, yd_real, zd_real);
        
        int amount_to_share = Nooflines_Real/np;
        int remains= Nooflines_Real%np;
        int size;
        int remains_shared;
        for (int i=1; i< np; i++) {
            if (remains>0){
                remains_shared=1;
            }
            else{
                remains_shared=0;
            }
            size = amount_to_share + remains_shared;
            MPI_Send(&size, 1, MPI_INT, i, tag, MPI_COMM_WORLD);
            
            MPI_Send(xd_real, size, MPI_FLOAT, i,tag,MPI_COMM_WORLD);
            MPI_Send(yd_real, size, MPI_FLOAT, i,tag,MPI_COMM_WORLD);
            MPI_Send(zd_real, size, MPI_FLOAT, i,tag,MPI_COMM_WORLD);
            remains--;
        }
        
        
        
        stoptime = MPI_Wtime();
        printf("Time: %f s\n", stoptime-starttime);
    }
    else {
        err=MPI_Recv (&size_real, 1, MPI_INT, 0, tag, MPI_COMM_WORLD, &status);
        if(err==MPI_SUCCESS){
            //printf("Process %d receive %d\n", id,size_real);
        }
        /* Allocate arrays for x, y and z values */
        xd_real = (float *)calloc( size_real, sizeof(float) );
        yd_real = (float *)calloc( size_real, sizeof(float) );
        zd_real = (float *)calloc( size_real, sizeof(float) );
        MPI_Recv (xd_real, size_real, MPI_FLOAT, 0, tag, MPI_COMM_WORLD, &status);
        MPI_Recv (yd_real, size_real, MPI_FLOAT, 0, tag, MPI_COMM_WORLD, &status);
        MPI_Recv (zd_real, size_real, MPI_FLOAT, 0, tag, MPI_COMM_WORLD, &status);
        
        for ( i = 0; i < size_real; ++i )
        {
            for ( j = i+1; j < size_real; ++j )
            {
                add_histogram (xd_real[i], yd_real[i], zd_real[i],
                               xd_real[j], yd_real[j], zd_real[j], histogramDD, pi, costotaldegrees);
            }
        }
        
        /* Multiply DD histogram with 2 since we only calculate (i,j) pair, not (j,i) */
        for ( i = 0; i <= nr_of_bins; ++i )
            histogramDD[i] *= 2L;
        histogramDD[0] += ((long)(size_real));
        
        /* Count the total nr of values in the DD histograms */
        TotalCountDD = 0L;
        for ( i = 0; i <= nr_of_bins; ++i )
            TotalCountDD += (long)(histogramDD[i]);
        printf("%d:  histogram count = %ld\n\n",id, TotalCountDD);
        
        
        int u = np/2;
        int to;
        if (np%2 == 0) { //pair
            for (int k=1; k<=u ; k++) {
                to=(id - k - 1) % np;
                MPI_Send(&size_real, 1, MPI_INT, to, tag, MPI_COMM_WORLD);
            }
            
            //MPI_Send(xd_real, size, MPI_FLOAT, i,tag,MPI_COMM_WORLD);
            //MPI_Send(yd_real, size, MPI_FLOAT, i,tag,MPI_COMM_WORLD);
            //MPI_Send(zd_real, size, MPI_FLOAT, i,tag,MPI_COMM_WORLD);
        }

        
        
    }
    
    MPI_Finalize();
    exit(0);
}
