#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <assert.h>
#include <xmmintrin.h>
 #include <mpi.h>
  


   #define return_data_temp_maxF 2000
   #define send_data_tag_chunk_size 2001
#define send_data_tag_start_chunk 2002

double gettime(void)
{
  struct timeval ttime;
  gettimeofday(&ttime , NULL);
  return ttime.tv_sec + ttime.tv_usec * 0.000001;
}

float randpval ()
{
  int vr = rand();
  int vm = rand()%vr;
  float r = ((float)vm)/(float)vr;
  assert(r>=0.0 && r<=1.00001);
  return r;
}

//////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char ** argv)
{


      

  




      MPI_Status status;
      int my_id, root_process, ierr, i, num_rows, num_procs,
         an_id, chunk_size_to_receive, avg_chunk_per_procs, 
         sender,sender1, chunk_size_received, start_chunk, end_chunk, chunk_size_to_send,temp_maxF;



  int N = atoi(argv[1]);
  int iters = 1000;
  srand(1);
  float * mVec = (float*)malloc(sizeof(float)*N);
  assert(mVec!=NULL);
  float * nVec = (float*)malloc(sizeof(float)*N);
  assert(nVec!=NULL);
  float * LVec = (float*)malloc(sizeof(float)*N);
  assert(LVec!=NULL);
  float * RVec = (float*)malloc(sizeof(float)*N);
  assert(RVec!=NULL);
  float * CVec = (float*)malloc(sizeof(float)*N);
  assert(CVec!=NULL);
  float * FVec = (float*)malloc(sizeof(float)*N);
  assert(FVec!=NULL);

  
  for(int i=0;i<N;i++)
  {
    mVec[i] = (float)(2+rand()%10);
    nVec[i] = (float)(2+rand()%10);
    LVec[i] = 0.0;
    for(int j=0;j<mVec[i];j++)
    {
      LVec[i] += randpval();
    }

    RVec[i] = 0.0;
    for(int j=0;j<nVec[i];j++)
    {
      RVec[i] += randpval();
    }

    CVec[i] = 0.0;
    for(int j=0;j<mVec[i]*nVec[i];j++)
    {
      CVec[i] += randpval();
    }

    FVec[i] = 0.0;
    assert(mVec[i]>=2.0 && mVec[i]<=12.0);
    assert(nVec[i]>=2.0 && nVec[i]<=12.0);
    assert(LVec[i]>0.0 && LVec[i]<=1.0*mVec[i]);
    assert(RVec[i]>0.0 && RVec[i]<=1.0*nVec[i]);
    assert(CVec[i]>0.0 && CVec[i]<=1.0*mVec[i]*nVec[i]);
  }

  float maxF = 0.0f,global_maxF = 0.0f;
  double timeTotal = 0.0f;
  __m128 mF=_mm_set_ps1(0.0f);
  __m128 num_0,num_1,num_2,num,den_0,den_1,den;
  __m128 t_num_0,t_num_1,t_num_2,t_num,t_den_0,t_den_1,t_den;
  __m128 t_1=_mm_set_ps1(1.0f);
  __m128 t_2=_mm_set_ps1(2.0f);
  __m128 t_3=_mm_set_ps1(0.01f);
  __m128 FV;
  __m128 mV,nV,RV,LV,CV;
  float xx[4],x1,x2,local_max[4],global_max[4];

   ierr = MPI_Init(&argc, &argv);
 root_process = 0;

MPI_Bcast(LVec, 1, MPI_FLOAT, root_process, MPI_COMM_WORLD);
MPI_Bcast(RVec, 1, MPI_FLOAT, root_process, MPI_COMM_WORLD);
MPI_Bcast(CVec, 1, MPI_FLOAT, root_process, MPI_COMM_WORLD);
MPI_Bcast(mVec, 1, MPI_FLOAT, root_process, MPI_COMM_WORLD);
MPI_Bcast(nVec, 1, MPI_FLOAT, root_process, MPI_COMM_WORLD);



  for(int j=0;j<iters;j++)
  {

    double time0=gettime();


   
      
      /* find out MY process ID, and how many processes were started. */
      
      ierr = MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
      ierr = MPI_Comm_size(MPI_COMM_WORLD, &num_procs);


      if(my_id == root_process) {
         
         /* I must be the root process. */


         avg_chunk_per_procs = (N / num_procs) ;

         /* distribute a portion of the bector to each child process */
       
	//printf( "Rank[%i] : | start_chunk = %i | end_chunk=%i  | \n",my_id,0,avg_chunk_per_procs);

         for(an_id = 1; an_id < num_procs; an_id++) {

            start_chunk = an_id*avg_chunk_per_procs + 1;
            end_chunk   = (an_id + 1)*avg_chunk_per_procs;
            
            if((N - end_chunk) < avg_chunk_per_procs){
                                      end_chunk = N - 1;
              }

	//printf( "Rank[%i] : | start_chunk = %i | end_chunk=%i  | \n",an_id,start_chunk,end_chunk);
            chunk_size_to_send = end_chunk - start_chunk + 1;

            ierr = MPI_Send( &chunk_size_to_send, 1 , MPI_INT,
                  an_id, send_data_tag_chunk_size, MPI_COMM_WORLD);
                
            ierr = MPI_Send( &start_chunk, 1 , MPI_INT,
                  an_id, send_data_tag_start_chunk, MPI_COMM_WORLD);
  


         }       

         /* and calculate the max of the values in the segment assigned
          * to the root process */
     
         for(i = 0; i < avg_chunk_per_procs + 1; i+=4) {

                  mV=_mm_load_ps(&mVec[i]);
                  nV=_mm_load_ps(&nVec[i]);
                  RV=_mm_load_ps(&RVec[i]);
                  LV=_mm_load_ps(&LVec[i]);
                  CV=_mm_load_ps(&CVec[i]);


                  t_num_0=_mm_add_ps(LV,RV);
            //num_1=_mm_mul_ps(mV,_mm_div_ps(_mm_sub_ps(mV,t_1),t_2));
                  t_num_1=_mm_div_ps(_mm_mul_ps(mV,_mm_sub_ps(mV,t_1)),t_2);
            //num_2=_mm_mul_ps(nV,_mm_div_ps(_mm_sub_ps(nV,t_1),t_2));
                  t_num_2=_mm_div_ps(_mm_mul_ps(nV,_mm_sub_ps(nV,t_1)),t_2);
                  t_num=_mm_div_ps(t_num_0,(_mm_add_ps(t_num_1,t_num_2)));
                  t_den_0=_mm_sub_ps(CV,(_mm_add_ps(LV,RV)));
                  t_den_1=_mm_mul_ps(mV,nV);
                  t_den=_mm_div_ps(t_den_0,t_den_1);
                  FV=_mm_div_ps(t_num,_mm_add_ps(t_den,t_3));
                  mF =_mm_max_ps(FV,mF);



        }


         _mm_store_ps(local_max, mF);
         
      }

      else {

         /* I must be a slave process, so I must receive my array segment,
          * storing it in a "local" array, array1. */
  


           sender1 = status.MPI_SOURCE;
         ierr = MPI_Recv( &chunk_size_to_receive, 1, MPI_INT, 
               root_process, send_data_tag_chunk_size, MPI_COMM_WORLD, &status);
       

         ierr = MPI_Recv( &start_chunk, 1, MPI_FLOAT, 
               root_process, send_data_tag_start_chunk, MPI_COMM_WORLD, &status);
       

         chunk_size_received = chunk_size_to_receive;

     /* for (int w=start_chunk;w<start_chunk +chunk_size_received;w++)  
            printf("Lvec[%i] = %f\n",i,LVec[w]);*/
         for(i = start_chunk; i < start_chunk + chunk_size_received+1; i+=4) {
                
			
               LV=_mm_loadu_ps(&LVec[i]);
               mV=_mm_loadu_ps(&mVec[i]);
               nV=_mm_loadu_ps(&nVec[i]);
               RV=_mm_loadu_ps(&RVec[i]);
                CV=_mm_loadu_ps(&CVec[i]);

            /*  how to print __m128           
              uint16_t *val = (uint16_t*) &LV;
              printf(" || LV[%i] = %f || , || LV[%i] = %f || , || LV[%i] = %f ||,|| LV[%i] = %f ||\n",
                      i,val[0],i+1,val[1],i+2 ,val[2],i+3, val[3],i+4,val[4]);*/


                num_0=_mm_add_ps(LV,RV);
                num_1=_mm_div_ps(_mm_mul_ps(mV,_mm_sub_ps(mV,t_1)),t_2);
                num_2=_mm_div_ps(_mm_mul_ps(nV,_mm_sub_ps(nV,t_1)),t_2);
                num=_mm_div_ps(num_0,(_mm_add_ps(num_1,num_2)));
                den_0=_mm_sub_ps(CV,(_mm_add_ps(LV,RV)));
                den_1=_mm_mul_ps(mV,nV);
                den=_mm_div_ps(den_0,den_1);
                FV=_mm_div_ps(num,_mm_add_ps(den,t_3));
                mF =_mm_max_ps(FV,mF);


         

}	
	
	
        _mm_store_ps(local_max, mF);



     /*  printf("RANK[%i] -> local_max[0]=%f,local_max[1]=%f,local_max[2]=%f,local_max[3]=%f \n",my_id,local_max[0],local_max[1],local_max[2],local_max[3]);
printf("RANK[%i] -> maxF[0]=%f \n",my_id,maxF);*/
      }

      double time1=gettime();
      timeTotal += time1-time0;
  }

MPI_Reduce(local_max,global_max , 4, MPI_FLOAT, MPI_MAX, 0,
           MPI_COMM_WORLD);

         if(my_id==0){

x1 = global_max[0]>global_max[1]?global_max[0]:global_max[1];
x2 = global_max[2]>global_max[3]?global_max[2]:global_max[3];
x2 = x1>x2?x1:x2;
           /* printf("Time %f global_max[0]=%f,global_max[1]=%f,global_max[2]=%f,global_max[3]=%f\n", timeTotal/iters,global_max[0],global_max[1],global_max[2],global_max[3]);*/
 printf("Time %f || Max %f \n", timeTotal/iters,x2);
}
         ierr = MPI_Finalize();

  
  free(mVec);
  free(nVec);
  free(LVec);
  free(RVec);
  free(CVec);
  free(FVec);
  return 0;
}







