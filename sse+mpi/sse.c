#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <assert.h>
#include <xmmintrin.h>


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

	float maxF = 0.0f;
	double timeTotal = 0.0f;
	__m128 mF=_mm_set_ps1(0.0f);
	__m128 num_0,num_1,num_2,num,den_0,den_1,den;
	__m128 t_1=_mm_set_ps1(1.0f);
	__m128 t_2=_mm_set_ps1(2.0f);
	__m128 t_3=_mm_set_ps1(0.01f);
	__m128 FV;
	__m128 mV,nV,RV,LV,CV;
	float xx[4],x1,x2;

	for(int j=0;j<iters;j++)
	{

		double time0=gettime();

		for(int i=0;i<N;i+=4)
		{

	

			mV=_mm_load_ps(&mVec[i]);
			nV=_mm_load_ps(&nVec[i]);
			RV=_mm_load_ps(&RVec[i]);
			LV=_mm_load_ps(&LVec[i]);
			CV=_mm_load_ps(&CVec[i]);


			num_0=_mm_add_ps(LV,RV);
			//num_1=_mm_mul_ps(mV,_mm_div_ps(_mm_sub_ps(mV,t_1),t_2));
			num_1=_mm_div_ps(_mm_mul_ps(mV,_mm_sub_ps(mV,t_1)),t_2);
			//num_2=_mm_mul_ps(nV,_mm_div_ps(_mm_sub_ps(nV,t_1),t_2));
			num_2=_mm_div_ps(_mm_mul_ps(nV,_mm_sub_ps(nV,t_1)),t_2);
			num=_mm_div_ps(num_0,(_mm_add_ps(num_1,num_2)));
			den_0=_mm_sub_ps(CV,(_mm_add_ps(LV,RV)));
			den_1=_mm_mul_ps(mV,nV);
			den=_mm_div_ps(den_0,den_1);
			FV=_mm_div_ps(num,_mm_add_ps(den,t_3));
			mF =_mm_max_ps(FV,mF);


					if( (i+4>N) & (N%4!=0) )
			{

				//printf("Express is %d, i %d, j %d \n", ( (i+4>N) & (N%4!=0) ),i,j);
				
				// printf("CVec is %f, RVec is %f\n", CVec[i],RVec[i]);
				// printf("CVec is %f, RVec is %f\n", CVec[i-1],RVec[i-1]);
				// printf("CVec is %f, RVec is %f\n", CVec[i-2],RVec[i-2]);
				// printf("CVec is %f, RVec is %f\n", CVec[i-3],RVec[i-3]);

				for(int i=(N-N%4);i<N;i++)
				{
					//printf("!!!!!!!!!!!!!\n");
					float num_0 = LVec[i]+RVec[i];
					float num_1 = mVec[i]*(mVec[i]-1.0)/2.0;
					float num_2 = nVec[i]*(nVec[i]-1.0)/2.0;
					float num = num_0/(num_1+num_2);
					float den_0 = CVec[i]-LVec[i]-RVec[i];
					float den_1 = mVec[i]*nVec[i];
					float den = den_0/den_1;
					FVec[i] = num/(den+0.01);
					maxF = FVec[i]>maxF?FVec[i]:maxF;

				}

	}		


		}
		double time1=gettime();
		timeTotal += time1-time0;
	}


	_mm_store_ps(xx, mF);
	x1 = xx[0]>xx[1]?xx[0]:xx[1];
	x2 = xx[2]>xx[3]?xx[2]:xx[3];
	x2 = x1>x2?x1:x2;

	printf("Time %f Max %f\n", timeTotal/iters,/*xx[0],xx[1],xx[2],xx[3],maxF,*/ (x2>maxF?x2:maxF));
	free(mVec);
	free(nVec);
	free(LVec);
	free(RVec);
	free(CVec);
	free(FVec);
}
