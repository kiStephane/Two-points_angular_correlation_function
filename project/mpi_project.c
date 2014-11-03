//
//  mpi_project.c
//  
//
//  Created by Stephane KI on 01/11/2014.
//  Compile : mpicc -O3 -std=c99 mpi_project.c -o mpi_project
//  Run : srun -n 3 mpi_project real_data sim_data output_file
//

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


int main(int argc, char *argv[]){
    int i,j,k;
    int np, id, err;
    MPI_Status status;
    int tag = 777;
    int Nooflines_Real;  /* Nr of lines in real data */
    int Nooflines_Sim;   /* Nr of lines in random data */
    long int TotalCountDD, TotalCountDR, TotalCountRR; /* Counters */
    float pi, costotaldegrees;
    int nr_of_bins = binsperdegree*totaldegrees;  /* Total number of bins */
    double starttime, stoptime;
    
    long int *histogramDD, *histogramDR, *histogramRR; /* Arrays for histograms */
    long int *histogramDD_total, *histogramDR_total, *histogramRR_total; /* SUM Arrays for histograms */
    float *xd_real, *yd_real, *zd_real;         /* Arrays for real data */
    float *xd_sim , *yd_sim , *zd_sim;          /* Arrays for random data */
    double NSimdivNReal, w;
    double *W, *W_sum;
    double startTime, endTime;
    
    
    /* Check that we have 4 command line arguments */
    if ( argc != 4 ) {
        printf("Usage: %s real_data sim_data output_file\n", argv[0]);
        return(0);
    }
    
    histogramDD = (long int *)calloc( nr_of_bins+1, sizeof(long int));
    histogramDD_total = (long int *)calloc( nr_of_bins+1, sizeof(long int));
    
    histogramDR = (long int *)calloc( nr_of_bins+1, sizeof(long int));
    histogramDR_total = (long int *)calloc( nr_of_bins+1, sizeof(long int));
    
    histogramRR = (long int *)calloc( nr_of_bins+1, sizeof(long int));
    histogramRR_total = (long int *)calloc( nr_of_bins+1, sizeof(long int));
    
    W=(double*)calloc(nr_of_bins, sizeof(double));
    
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
    
    FILE *infile, *outfile;  /* Input and output files */
    
    
    if (id == 0) { /* Process 0 does this */
        
        //=========Read and send real data====================
        /* Open the real data input file */
        infile = fopen(argv[1],"r");
        if ( infile == NULL ) {
            printf("Unable to open %s\n",argv[1]);
            MPI_Finalize();
            exit(0);
        }
        Nooflines_Real = count_lines(infile);
        /* Allocate arrays for x, y and z values */
        xd_real = (float *)calloc( Nooflines_Real, sizeof(float) );
        yd_real = (float *)calloc( Nooflines_Real, sizeof(float) );
        zd_real = (float *)calloc( Nooflines_Real, sizeof(float) );
        
        MPI_Bcast(&Nooflines_Real,1, MPI_INT, 0,MPI_COMM_WORLD );
        
        /* Read the file with real input data */
        read_data(infile, Nooflines_Real, xd_real, yd_real, zd_real);
        
        MPI_Bcast(xd_real,Nooflines_Real, MPI_FLOAT, 0,MPI_COMM_WORLD );
        MPI_Bcast(yd_real,Nooflines_Real, MPI_FLOAT, 0,MPI_COMM_WORLD );
        MPI_Bcast(zd_real,Nooflines_Real, MPI_FLOAT, 0,MPI_COMM_WORLD );
        //=====================================================
        
        //========Read and send simulated data======================
        /* Open the file with random (simulated) data */
        infile = fopen(argv[2],"r");
        if ( infile == NULL ) {
            printf("Unable to open %s\n",argv[2]);
            MPI_Finalize();
            exit(0);
        }
        /* Count how many lines the file has */
        Nooflines_Sim = count_lines(infile);
        xd_sim = (float *)calloc( Nooflines_Sim, sizeof(float) );
        yd_sim = (float *)calloc( Nooflines_Sim, sizeof(float) );
        zd_sim = (float *)calloc( Nooflines_Sim, sizeof(float) );

        MPI_Bcast(&Nooflines_Sim,1, MPI_INT, 0,MPI_COMM_WORLD );
        
        /* Read the input file */
        read_data(infile, Nooflines_Sim, xd_sim, yd_sim, zd_sim);
        
        MPI_Bcast(xd_sim,Nooflines_Sim, MPI_FLOAT, 0,MPI_COMM_WORLD );
        MPI_Bcast(yd_sim,Nooflines_Sim, MPI_FLOAT, 0,MPI_COMM_WORLD );
        MPI_Bcast(zd_sim,Nooflines_Sim, MPI_FLOAT, 0,MPI_COMM_WORLD );
        //==========================================================
        

        startTime = MPI_Wtime();      // Start measuring time
        
        //Cyclical decomposition of operations
        //Process i is assigned data items i, i+np, i+2np, i+3np
        
        //===========Histogram DD, RR and DR=====================
        i=0; k=0;
        while (i<Nooflines_Real) {
            
            for ( j = i+1; j < Nooflines_Real; j++ ) // IMPORTANT Nooflines_Real should be equal to Nooflines_Sim
            {
                add_histogram (xd_real[i], yd_real[i], zd_real[i],
                               xd_real[j], yd_real[j], zd_real[j],
                               histogramDD, pi, costotaldegrees);
                
                add_histogram (xd_sim[i], yd_sim[i], zd_sim[i],
                               xd_sim[j], yd_sim[j], zd_sim[j],
                               histogramRR, pi, costotaldegrees);
            }
            
            for ( j = 0; j < Nooflines_Sim; j++ ) // IMPORTANT Nooflines_Real should be equal to Nooflines_Sim
            {
                add_histogram (xd_real[i], yd_real[i], zd_real[i],
                               xd_sim[j], yd_sim[j], zd_sim[j],
                               histogramDR, pi, costotaldegrees);
            }

            
            k = k+1;
            i = k*np;
        }
        
        /* Multiply DD and RR histogram with 2 since we only calculate (i,j) pair, not (j,i) */
        for ( i = 0; i <= nr_of_bins; ++i ){
            histogramDD[i] *= 2L;
            histogramRR[i] *= 2L;
        }
        
        histogramDD[0] += ((long)(k));
        histogramRR[0] += ((long)(k));
        
        /* Count the total nr of values in the DD histograms */
        TotalCountDD = 0L;
        for ( i = 0; i <= nr_of_bins; ++i )
            TotalCountDD += (long)(histogramDD[i]);
        printf("%d:  histogram DD count = %ld\n\n", id, TotalCountDD);
        
        
        /* Count the total nr of values in the RR histograms */
        TotalCountRR = 0L;
        for ( i = 0; i <= nr_of_bins; ++i )
            TotalCountRR += (long)(histogramRR[i]);
        printf("%d:  histogram RR count = %ld\n\n", id, TotalCountRR);
        
        /* Count the total nr of values in the DR histograms */
        TotalCountDR = 0L;
        for ( i = 0; i <= nr_of_bins; ++i )
            TotalCountDR += (long)(histogramDR[i]);
        printf("%d:  histogram DR count = %ld\n\n",id, TotalCountDR);
        
        
        MPI_Allreduce(histogramDD, histogramDD_total, nr_of_bins+1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(histogramDR, histogramDR_total, nr_of_bins+1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(histogramRR, histogramRR_total, nr_of_bins+1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
        
        //=============FOR Debug================
        TotalCountDD = 0L;
        for ( i = 0; i <= nr_of_bins; ++i )
            TotalCountDD += (long)(histogramDD_total[i]);
        printf("TOTAL Histogram DD count = %ld\n\n", TotalCountDD);
        
        TotalCountDR = 0L;
        for ( i = 0; i <= nr_of_bins; ++i )
            TotalCountDR += (long)(histogramDR_total[i]);
        printf("TOTAL Histogram DR count = %ld\n\n",TotalCountDR);
        
        TotalCountRR = 0L;
        for ( i = 0; i <= nr_of_bins; ++i )
            TotalCountRR += (long)(histogramRR_total[i]);
        printf("TOTAL Histogram RR count = %ld\n\n", TotalCountRR);
        //==========================================
        
        /* Open the output file */
        outfile = fopen(argv[3],"w");
        if ( outfile == NULL ) {
            printf("Unable to open %s\n",argv[3]);
            MPI_Finalize();
            exit(0);
        }
        
        W_sum=(double*)calloc(nr_of_bins, sizeof(double));
        
        i=id;
        k=0;
        while (i<nr_of_bins)
        {
            NSimdivNReal = ((double)(Nooflines_Sim))/((double)(Nooflines_Real));
            
            W[i] = 1.0 + NSimdivNReal*NSimdivNReal*histogramDD_total[i]/histogramRR_total[i]
            -2.0*NSimdivNReal*histogramDR_total[i]/((double)(histogramRR_total[i]));
            
            k = k+1;
            i = id+k*np;
        }
        
        MPI_Reduce(W, W_sum, nr_of_bins, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        
        fprintf(outfile,"bin center\tomega\t        hist_DD\t        hist_DR\t        hist_RR\n");
        for ( i = 0; i < nr_of_bins; ++i )
        {
            fprintf(outfile,"%6.3f\t%15lf\t%15ld\t%15ld\t%15ld\n",((float)i+0.5)/binsperdegree, W_sum[i],
                    histogramDD_total[i], histogramDR_total[i], histogramRR_total[i]);
        }
        
        fclose(outfile);
        
        endTime = MPI_Wtime();      // Stop measuring time
        printf("Time: %f s\n", endTime-startTime);
        free(W_sum);
        
    }
    else { /* all other processes do this */
        
        //========Receive real data from process 0=============
        MPI_Bcast(&Nooflines_Real,1, MPI_INT, 0,MPI_COMM_WORLD );
        
        /* Allocate arrays for x, y and z values */
        xd_real = (float *)calloc( Nooflines_Real, sizeof(float) );
        yd_real = (float *)calloc( Nooflines_Real, sizeof(float) );
        zd_real = (float *)calloc( Nooflines_Real, sizeof(float) );
        
        MPI_Bcast(xd_real,Nooflines_Real, MPI_FLOAT, 0,MPI_COMM_WORLD );
        MPI_Bcast(yd_real,Nooflines_Real, MPI_FLOAT, 0,MPI_COMM_WORLD );
        MPI_Bcast(zd_real,Nooflines_Real, MPI_FLOAT, 0,MPI_COMM_WORLD );
        
        //=======Receive simulated data from process 0=======
        MPI_Bcast(&Nooflines_Sim,1, MPI_INT, 0,MPI_COMM_WORLD );
        
        xd_sim = (float *)calloc( Nooflines_Sim, sizeof(float) );
        yd_sim = (float *)calloc( Nooflines_Sim, sizeof(float) );
        zd_sim = (float *)calloc( Nooflines_Sim, sizeof(float) );
        
        MPI_Bcast(xd_sim,Nooflines_Sim, MPI_FLOAT, 0,MPI_COMM_WORLD );
        MPI_Bcast(yd_sim,Nooflines_Sim, MPI_FLOAT, 0,MPI_COMM_WORLD );
        MPI_Bcast(zd_sim,Nooflines_Sim, MPI_FLOAT, 0,MPI_COMM_WORLD );
        
        //Cyclical decomposition of operations
        //Process i is assigned data items i, i+np, i+2np, i+3np
        
        //=============Histogram DD and RR=================
        i=id;
        k=0;
        while (i<Nooflines_Real) {
            
            for ( j = i+1; j < Nooflines_Real; j++ )
            {
                add_histogram (xd_real[i], yd_real[i], zd_real[i],
                               xd_real[j], yd_real[j], zd_real[j],
                               histogramDD, pi, costotaldegrees);
                
                add_histogram (xd_sim[i], yd_sim[i], zd_sim[i],
                               xd_sim[j], yd_sim[j], zd_sim[j],
                               histogramRR, pi, costotaldegrees);
            }
            
            for ( j = 0; j < Nooflines_Sim; j++ ) // IMPORTANT Nooflines_Real should be equal to Nooflines_Sim
            {
                add_histogram (xd_real[i], yd_real[i], zd_real[i],
                               xd_sim[j], yd_sim[j], zd_sim[j],
                               histogramDR, pi, costotaldegrees);
            }

            
            
            k = k+1;
            i = id+k*np;
        }
        
        /* Multiply DD and RR histogram with 2 since we only calculate (i,j) pair, not (j,i) */
        for ( i = 0; i <= nr_of_bins; ++i ){
            histogramDD[i] *= 2L;
            histogramRR[i] *= 2L;
        }
        histogramDD[0] += ((long)(k));
        histogramRR[0] += ((long)(k));
        
        /* Count the total nr of values in the DD histograms */
        TotalCountDD = 0L;
        for ( i = 0; i <= nr_of_bins; ++i )
            TotalCountDD += (long)(histogramDD[i]);
        printf("%d:  histogram DD count = %ld\n\n", id, TotalCountDD);
        
        
        /* Count the total nr of values in the RR histograms */
        TotalCountRR = 0L;
        for ( i = 0; i <= nr_of_bins; ++i )
            TotalCountRR += (long)(histogramRR[i]);
        printf("%d:  histogram RR count = %ld\n\n", id, TotalCountRR);
        
        /* Count the total nr of values in the DR histograms */
        TotalCountDR = 0L;
        for ( i = 0; i <= nr_of_bins; ++i )
            TotalCountDR += (long)(histogramDR[i]);
        printf("%d:  histogram DR count = %ld\n\n",id, TotalCountDR);
        
        MPI_Allreduce(histogramDD, histogramDD_total, nr_of_bins+1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(histogramDR, histogramDR_total, nr_of_bins+1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(histogramRR, histogramRR_total, nr_of_bins+1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
        
        i=id;
        k=0;
        while (i<nr_of_bins)
        {
            NSimdivNReal = ((double)(Nooflines_Sim))/((double)(Nooflines_Real));
            
            W[i] = 1.0 + NSimdivNReal*NSimdivNReal*histogramDD_total[i]/histogramRR_total[i]
            -2.0*NSimdivNReal*histogramDR_total[i]/((double)(histogramRR_total[i]));
            
            k = k+1;
            i = id+k*np;
        }
        
        MPI_Reduce(W, W, nr_of_bins, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        
    }
    /* Free all allocated arrays */
    free(histogramDD);
    free(histogramDR);
    free(histogramRR);
    
    free(histogramDD_total);
    free(histogramDR_total);
    free(histogramRR_total);
    
    free(W);
    
    free(xd_real);free(yd_real);free(zd_real);
    free(xd_sim);free(yd_sim);free(zd_sim);
    
    MPI_Finalize();
    exit(0);
    
}