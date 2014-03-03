#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cv.h>
#include <highgui.h>
#include <math.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define SIGN(a, b) ( (b) < 0 ? -fabs(a) : fabs(a) )

void main(int argc,char *argv[])
{

 /***********************************************
   (0)资源分配
 ***********************************************/
     FILE *T,*TEST;
     int   n=200*100,m=24,count=0,i,j,k,e,testnum,N,R;                                        //定义矩阵的参数 和 循环变量
     int   height,width,step;                                                    
	 char  tempstr[3];
   	 char *imgjpg=".jpg";
	 char *filename;
	 float pix_value;                                                                        //像素元素变量
	 float temp,temp1,temp2,eps=200000;                                                      //冒泡排序临时存储变量, 特征值门限
	 float **datmat,**cormat,**eignmat,**A,**testdata,*evals,*interm,*Y,*mean,*bi,*di;       //定义内存单元，采用双指针结构
     float **matrix(), *vector();                                                            //开辟资源函数
	 int   free_matrix(), free_vector();                                                     //释放函数资源
     int   corcol(),decormat(),tred2(), tqli(),recognition(),sortegi();                                //生成是对称矩阵,求解协方差矩阵，测试图像识别
     int   imadjust();
	
     uchar *data;
   	 IplImage *trainimg=0;
	
	 datmat = matrix(n, m);                                                       // 为输入图像分配内存空间
	 cormat = matrix(m, m);                                                       // 为协方差矩阵分配存储空间
	
   /************************************************
   (1.1) OPENCV接口，初始化矩阵datmat(n*m)
   *************************************************/
    printf("一：读入样本库货币图像（24张）\n");
    for(i=1;i<=m;i++)
	{
       itoa(i, tempstr, 10);   
	   strcat(tempstr,imgjpg);  
	  	for(j=0;j<=5;j++)
		{
         printf("%c",tempstr[j]);
		} 
		 printf("  ");
      filename=tempstr;
      trainimg=cvLoadImage( filename, 0 );

	  height = trainimg->height;
	  width  = trainimg->width;
	  step   = trainimg->widthStep/sizeof(uchar);
	  data = (uchar *)trainimg->imageData;
     /*******调整图像尺寸为(200x100)************/
	  imadjust(datmat,data,height,width,n,i);
     /*******按原始尺寸获取图像数据*************/
	 /* for(j=1;j<=n;j++)
	  {
	    datmat[j][i]=(float)data[j-1];
	  } */

	/*  cvNamedWindow("trainning",CV_WINDOW_AUTOSIZE);
      cvShowImage("trainning",trainimg);
      cvWaitKey(0);
	  cvReleaseImage(&trainimg);  */
	  tempstr[0]='\0';
      tempstr[1]='\0';
	  filename='\0';
      trainimg=0;
	}
/*    printf("(1)初始化矩阵datmat(n*m)\n");   
	for (i = 1; i <= 20; i++)
	{  
		for (j = 1; j <= m; j++) 
		{
        printf("%7.1f", datmat[i][j]);  
		} 
	    printf("\n"); 
	}  */
 
/*******单张图像测试*************************	  
      itoa(12, tempstr, 10);   
	  strcat(tempstr,imgjpg); 
      filename=tempstr;
      trainimg=cvLoadImage( filename, 0 );
	  
	  height = trainimg->height;
	  width  = trainimg->width;
	  step   = trainimg->widthStep/sizeof(uchar);
	  data = (uchar *)trainimg->imageData;
      imadjust(datmat,data,height,width,n,1) ;
   
	  printf("(1)初始化矩阵datmat(n*m)\n");   
	for (i = 1; i <= 50; i++)
	{  
		for (j = 1; j <= m; j++) 
		{
        printf("%7.1f", datmat[i][j]);  
		} 
	    printf("\n"); 
	}  
     // printf("%d,%d,%d",width,height,data[0]);
	  cvNamedWindow("kevin",CV_WINDOW_AUTOSIZE);
      cvShowImage("kevin",trainimg);
      cvWaitKey(0);
	  cvReleaseImage(&trainimg);

    
/***********************************************
   (1.2) 文件接口,初始化矩阵datmat(n*m)
************************************************/
 /*  T=fopen("testdata.txt","r");                                                   
     for (i = 1; i <= n; i++)
        {
           for (j = 1; j <= m; j++)
            {
            fscanf(T, "%f", &pix_value);
            datmat[i][j] = pix_value;
            }
        }
	                                   
    printf("(1)初始化矩阵datmat(n*m)\n");   
	for (i = 1; i <= n; i++)
	{  
		for (j = 1; j <= m; j++) 
		{
        printf("%7.2f", datmat[i][j]);  
		} 
	    printf("\n"); 
	}  
                                    
 /******************************************************************
	(2)创建实对矩阵cormat(m*m);
 *******************************************************************/
    printf("\n二：训练样本库 \n");
    printf("(1)创建实对矩阵cormat(m*m) \n");
	mean = vector(n);
	corcol(datmat, n, m, cormat,mean);
/*	printf("\nMeans of column vectors:\n");
	 for (i = 1; i <= 20; i++) 
	 {
       printf("%7.2f",mean[i]);
	 }   
       printf("\n"); 
/*************************************************************************************
   (3)通过求解实对称矩阵的特征向量特征值求解协方差矩阵的特征值特征向量以及生成特征空间
**************************************************************************************/
	printf("(2)求解实对称矩阵的特征向量特征值\n");
    /* 1.为特征值分配存储空间. */
    evals = vector(m);    
	/* 2.为过度向量分配存储空间*/
    interm = vector(m);  
	/* 3.求解实对称矩阵的特征向量特征值*/
	decormat(cormat, m, evals, interm);

/*	printf("\nEigenvalues:\n");
     for (i = 1; i<=m; i++) 
	 {
        printf("%7.2f", evals[m-i+1]); 
	 }
    printf("\nEigenvectors:\n");
     for (i= 1; i <= m; i++) 
	 {
       for (j = 1; j<= m; j++)  
	   {
          printf("%7.2f", cormat[i][m-j+1]); 
	   }
          printf("\n"); 
	 }
 /***********************************************************************
    (4)排序对称矩阵的特征值特征向量cormat(n*e);
 ************************************************************************/    
		/*排序后的特征向量矩阵 (冒泡法)*/
	 for(i=1;i<=m;i++)
		{
	       	for(j=i+1;j<=m;j++)
			{
				temp1=0.0;
			    if(evals[i]<=evals[j])
				{   
					temp1=evals[i];
                    evals[i]=evals[j];
					evals[j]=temp1;
                  
					for(k=1;k<=m;k++)
					{   temp2=0.0;
				     	temp2=cormat[k][i];
                        cormat[k][i]=cormat[k][j];
					    cormat[k][j]=temp2;
					}
				}
			}
		}
     printf("(3)排序特征值后求主元:\n");
      for (i = 1; i<=m; i++) 
	 {
		  if(evals[i]>=eps)
		  {
			  count++;
			  printf("%12.1f", evals[i]); 
		  }
       
	 }
	 e=count;
	 printf("\n*************\n number of Eigenvalues:%d\n",count);
/*	 printf("\nsorted nEigenvectors:\n");
     for (i= 1; i <= m; i++) 
	 {
       for (j = 1; j<= e; j++)  
	   {
          printf("%7.2f", cormat[i][j]); 
	   }
          printf("\n"); 
	 }

 /********************************************************************
   (5) 求解协方差矩阵的特征值特征向量以及生成特征空间eignmat(n*e);
 *********************************************************************/
	printf("(4)生成特征空间W=eignmat(n*e)\n");   
	eignmat= matrix(n, e); 
	  for (j= 1; j <= n; j++)
	  {
       for (k = 1; k<= e; k++)
	   {
        eignmat[j][k]=0.0;
        for (i = 1; i <= m; i++)
        {
           eignmat[j][k] += (datmat[j][i] * cormat[i][k]);
         
		}
	/*	if(evals[k]>=eps)
		{ 
		   eignmat[j][k] /=(float) sqrt(evals[m-i+1]);	
		}*/
	   } 
	  }
	  
 /*     for (i = 1; i <= 20; i++)
	  {for (j = 1; j <= e; j++) 
	  {
        printf("%7.2f", eignmat[i][j]);  
	  } 
	  printf("\n"); 
	  }

/***********************************************************************
    (6) 计算样本图像的投影坐标A(e*m)
 ************************************************************************/
	 printf("(5)样本图像的投影坐标A(e*m)\n");  
   	 A= matrix(e, m); 
     for (j= 1; j <= e; j++)
	 {
        for (k = 1; k<= m; k++)
		{
          A[j][k]=0.0;
        for (i = 1; i <= n; i++)
         {
           A[j][k] += (eignmat[i][j] * datmat[i][k]);
		}
	   } 
	  } 
/*	 for (i = 1; i <= e; i++)
	  {for (j = 1; j <= m; j++) 
	  {
        printf("%7.1f", A[i][j]);  
	  } 
	  printf("\n"); 
	  }
/***********************************************************************
    (7.1) 求测试图像的投影以及识别bi(e)(OpenCV接口)
 ***********************************************************************/
   	 Y=vector(n);                                           // 为输入图像分配内存空间
	 bi=vector(e);                                          // 为测试图像坐标投影分配内存空间 
     di=vector(m);                                          // 为测试图像与样本图像间距离分配内存空间
	 printf("\n三：测试图像\n输入测试图像序号:(输入0为识别结束)\n");  
	 scanf("%d",&testnum);
	  while(testnum!=0)
	  {
	  itoa(testnum, tempstr, 10);   
	  strcat(tempstr,"t.jpg"); 
      filename=tempstr;
      trainimg=cvLoadImage( filename, 0 );
	  //  testdata = matrix(n, 1);   
	  height = trainimg->height;
	  width  = trainimg->width;
	  step   = trainimg->widthStep/sizeof(uchar);
	  data = (uchar *)trainimg->imageData;
     

      N=(int)(height*width/n);
	  /* for(j=1;j<=n;j++)
	  {
	    datmat[j][i]=(float)data[j-1];
	  } */

      //printf("%d\n",N);
      for(j=0;j<(n-1);j++)
	  {
        temp=0.0;
		for(k=0;k<(N-1);k++)
		{
         	temp+=(float)data[j*N+k];
			Y[j+1]=(temp/N);
		
		}
	  } 
     for(k=0;k<(height*width-(N-1)*n);k++)
	 {
		 temp+=(float)data[(N-1)*n+k];
	 }
	     Y[n]=(temp/n);

	  cvNamedWindow("testimage",CV_WINDOW_AUTOSIZE);
      cvShowImage("testimage",trainimg);
      cvWaitKey(0);
	  cvReleaseImage(&trainimg); 
/***********************************************************************
    (7.2) 求测试图像的投影以及识别bi(e)(文本接口)
***********************************************************************/
  /*  TEST=fopen("testdata1.txt","r"); 
    Y=vector(n);                                           // 为输入图像分配内存空间
    
	for (i = 1; i <= n; i++)
        {  
            fscanf(TEST, "%f", &pix_value);
            Y[i] = pix_value;
        }
	printf("\n(7)测试图像:\n");  
    for (i = 1; i <= 20; i++)
        {  
			printf("%7.1f\n", Y[i]);  
        }                               */

	R=recognition(A,eignmat,n,e,m,Y,mean,bi,di);  

	itoa(R, tempstr, 10);   
    strcat(tempstr,".jpg"); 
    filename=tempstr;
    trainimg=cvLoadImage( filename, 0 );
    cvNamedWindow("matchimage",CV_WINDOW_AUTOSIZE);
    cvShowImage("matchimage",trainimg);
	cvWaitKey(0);
    cvReleaseImage(&trainimg); 
	printf("\n输入测试图像序号:\n");
	scanf("\n%d",&testnum);
 }
/***********************************************************************
    (8) 释放内存资源
 ************************************************************************/
	free_matrix(datmat, n, m);                             //数据矩阵，中心化后矩阵
    free_matrix(cormat, m, m);                             //协方差矩阵，实对称特征向量巨阵
	free_matrix(eignmat, n, e);                            //缩减后协方差特征相向量矩阵
	free_matrix(A, e, m);                                  //样本图像投影坐标矩阵
	free_vector(mean, n);                                  //样本图像列均值
    free_vector(evals, m);                                 //特征值
    free_vector(interm, m);                                //临时过渡向量
	free_vector(Y, n);                                     //测试图像数据向量
                            
}





 int imadjust(float** img,uchar* d,int h,int w,int n,int i)
 {
    int j,k,N;
    float M=0.0;
	N=(int)(h*w/n);
  //printf("%d\n",N);
	for(j=0;j<(n-1);j++)
	{
	    M=0.0;
		for(k=0;k<(N-1);k++)
		{
	    	M+=(float)d[j*N+k];
			img[j+1][i]=(M/N);
		
		}
  //printf("%7.1f",M/n);	
	 } 
    for(k=0;k<(h*w-(N-1)*n);k++)
	 {
		 M+=(float)d[(N-1)*n+k];
	 }
	     img[n][i]=(M/n);
	 return 0;
 }

/**  创建对称矩阵  ***********************************/
int corcol(float **datmat, int n, int m, float **cormat,float *mean)
{

   int i, j, k;
/*求数据矩阵列均值 */

   for (i = 1; i <= n; i++)
    {
       mean[i] = 0.0;
       for (j = 1; j <= m; j++)
        {
        mean[i] += datmat[i][j];
        }
       mean[i]/= (float)m;
    }
/* 中心化图像矩阵 */

for (i = 1; i <= n; i++)
    {
    for (j = 1; j <= m; j++)
        {
        datmat[i][j] -= mean[i];
        }
    }
  // printf("Centered matrix :\n");

/*	for (i = 1; i <= n; i++) 
	{  
		for (j = 1; j <= m; j++)
		{
        printf("%7.2f", datmat[i][j]); 
		} 
		printf("\n");  
	} 
/* 求协方差矩阵 */
/***方法(1) *********************************/
    for (j = 1; j <= m-1; j++)
    {
       for (k = j+1; k <= m; k++)
        {
           cormat[j][k] = 0.0;
          for (i = 1; i <= n; i++)
            {
            cormat[j][k] += ( datmat[i][j] * datmat[i][k]);
            }
          cormat[k][j] = cormat[j][k];
        }
    }
  
	for (j= 1;j<= m; j++)
	{
		cormat[j][j] = 0.0;
		for(i= 1;i <= n; i++)
		{
         cormat[j][j] += ( datmat[i][j] * datmat[i][j]);
		}
	}
/***  方法2***********************************/
   /*for (j = 1; j <= m; j++)
    {
       for (k = 1; k<= m; k++)
       {
           cormat[j][k]=0.0;
           for (i = 1; i <= n; i++)
          {
           cormat[j][k] += ( datmat[i][j] * datmat[i][k]);
		   }
    	} 
     }*/
	
 /*   printf("Correlation matrix :\n");
   	for (i = 1; i <= m; i++) 
	{  
	  	for (j = 1; j <= m; j++)
		{
        printf("%7.1f", cormat[i][j]); 
		} 
		printf("\n");  
		} */
	   return 0;
}

/**求协方差矩阵的特征值特征向量**/
int decormat(float **a, int n,float *d,float *e)
{
	int i,j;
    tred2(a, n, d, e);  /* 三角分解 */
    tqli(d, e, n, a);   /* 三角协方差矩阵的缩减 */
	return 0;
}
/**  测试识别  ***********************************/
int recognition(float **a,float **d,int n,int e,int m,float *c,float *mean,float *bi,float *di)   //(样本坐标矩阵，特征空间矩阵，向量纬度,图像向量)
 {
	  int i,j,num; 
	  float sum,min;
	/*求测试图像的投影坐标*/
  
	  for (i = 1; i <= n; i++)
    {
        c[i] -= mean[i];   
    }  
	
	  for (j= 1; j <= e; j++)
	   {
          bi[j]=0.0;
        for (i = 1; i <= n; i++)
         {
           bi[j] += (d[i][j] * c[i]);
	   } 
	  } 
	  printf("测试图像的坐标为：\n");  
	  for (i = 1; i <= e; i++)
	  {
         printf("%10.1f", bi[i]);  
	  }
	   /*测试图像和样本库图像的欧式距离*/
	   
       for (i= 1; i <= m; i++)       
	  {    
          sum=0.0;
         for (j = 1; j <=e; j++)
         {
           sum += ((bi[j]- a[j][i])*(bi[j]- a[j][i]));
		 } 
		 di[i]=sqrt(sum);
	  } 
       printf("\n欧式距离为：\n");  
	   for (i = 1; i <= m; i++)
	  {
        printf("%11.1f", di[i]);
	  }
       /*取最小欧式距离为匹配图像*/
	   min=di[1];
	   for(i=1;i<=m;i++)
	   {
	     if(min>=di[i])
		 {
            min=di[i];
			num=i;
		 }
	   }
       printf("\n识别结果为第%d幅图像:", num);
      return num;
 }
/**  Reduce a real, symmetric matrix to a symmetric, tridiag. matrix. */
int tred2(float **a, int n,float *d,float *e)
/* Householder reduction of matrix a to tridiagonal form.
   Algorithm: Martin et al., Num. Math. 11, 181-195, 1968.
   Ref: Smith et al., Matrix Eigensystem Routines -- EISPACK Guide
        Springer-Verlag, 1976, pp. 489-494.
        W H Press et al., Numerical Recipes in C, Cambridge U P,
        1988, pp. 373-374.  */
{
int l, k, j, i;
float scale, hh, h, g, f;

for (i = n; i >= 2; i--)
    {
    l = i - 1;
    h = scale = 0.0;
    if (l > 1)
       {
       for (k = 1; k <= l; k++)
           scale += fabs(a[i][k]);
       if (scale == 0.0)
          e[i] = a[i][l];
       else
          {
          for (k = 1; k <= l; k++)
              {
              a[i][k] /= scale;
              h += a[i][k] * a[i][k];
              }
          f = a[i][l];
          g = f>0 ? -sqrt(h) : sqrt(h);
          e[i] = scale * g;
          h -= f * g;
          a[i][l] = f - g;
          f = 0.0;
          for (j = 1; j <= l; j++)
              {
              a[j][i] = a[i][j]/h;
              g = 0.0;
              for (k = 1; k <= j; k++)
                  g += a[j][k] * a[i][k];
              for (k = j+1; k <= l; k++)
                  g += a[k][j] * a[i][k];
              e[j] = g / h;
              f += e[j] * a[i][j];
              }
          hh = f / (h + h);
          for (j = 1; j <= l; j++)
              {
              f = a[i][j];
              e[j] = g = e[j] - hh * f;
              for (k = 1; k <= j; k++)
                  a[j][k] -= (f * e[k] + g * a[i][k]);
              }
         }
    }
    else
        e[i] = a[i][l];
    d[i] = h;
    }
d[1] = 0.0;
e[1] = 0.0;
for (i = 1; i <= n; i++)
    {
    l = i - 1;
    if (d[i])
       {
       for (j = 1; j <= l; j++)
           {
           g = 0.0;
           for (k = 1; k <= l; k++)
               g += a[i][k] * a[k][j];
           for (k = 1; k <= l; k++)
               a[k][j] -= g * a[k][i];
           }
       }
       d[i] = a[i][i];
       a[i][i] = 1.0;
       for (j = 1; j <= l; j++)
           a[j][i] = a[i][j] = 0.0;
    }
   return 0;
}

/**  Tridiagonal QL algorithm -- Implicit  **********************/

int tqli(float d[], float e[], int n,float **z)
{
int m, l, iter, i, k;
float s, r, p, g, f, dd, c, b;
int erhand();

for (i = 2; i <= n; i++)
    e[i-1] = e[i];
e[n] = 0.0;
for (l = 1; l <= n; l++)
    {
    iter = 0;
    do
      {
      for (m = l; m <= n-1; m++)
          {
          dd = fabs(d[m]) + fabs(d[m+1]);
          if (fabs(e[m]) + dd == dd) break;
          }
          if (m != l)
             {
             if (iter++ == 30) erhand("No convergence in TLQI.");
             g = (d[l+1] - d[l]) / (2.0 * e[l]);
             r = sqrt((g * g) + 1.0);
             g = d[m] - d[l] + e[l] / (g + SIGN(r, g));
             s = c = 1.0;
             p = 0.0;
             for (i = m-1; i >= l; i--)
                 {
                 f = s * e[i];
                 b = c * e[i];
                 if (fabs(f) >= fabs(g))
                    {
                    c = g / f;
                    r = sqrt((c * c) + 1.0);
                    e[i+1] = f * r;
                    c *= (s = 1.0/r);
                    }
                 else
                    {
                    s = f / g;
                    r = sqrt((s * s) + 1.0);
                    e[i+1] = g * r;
                    s *= (c = 1.0/r);
                    }
                 g = d[i+1] - p;
                 r = (d[i] - g) * s + 2.0 * c * b;
                 p = s * r;
                 d[i+1] = g + p;
                 g = c * r - b;
                 for (k = 1; k <= n; k++)
                     {
                     f = z[k][i+1];
                     z[k][i+1] = s * z[k][i] + c * f;
                     z[k][i] = c * z[k][i] - s * f;
                     }
                 }
                 d[l] = d[l] - p;
                 e[l] = g;
                 e[m] = 0.0;
             }
          }  while (m != l);
      }
    return 0;
 }
/***********************************************************************
  资源分配
************************************************************************/
/** Error handler**************************************************/
int erhand(char err_msg)
{
    fprintf(stderr,"Run-time error:\n");
    fprintf(stderr,"%s\n", err_msg);
    fprintf(stderr,"Exiting to system.\n");
    exit(1);
	return 0;
}

/**  Allocation of vector storage  ***********************************/

float *vector(int n)
/* Allocates a float vector with range [1..n]. */
{
    float *v;

    v = (float *) malloc ((unsigned) n*sizeof(float));
    if (!v) erhand("Allocation failure in vector().");
    return v-1;

}

/**  Allocation of float matrix storage  *****************************/

float **matrix(int n,int m)
/* Allocate a float matrix with range [1..n][1..m]. */
{
    int i;
    float **mat;

    /* Allocate pointers to rows. */
    mat = (float **) malloc((unsigned) (n)*sizeof(float*));
    if (!mat) erhand("Allocation failure 1 in matrix().");
    mat -= 1;

    /* Allocate rows and set pointers to them. */
    for (i = 1; i <= n; i++)
        {
        mat[i] = (float **) malloc((unsigned) (m)*sizeof(float));
        if (!mat[i]) erhand("Allocation failure 2 in matrix().");
        mat[i] -= 1;
        }

     /* Return pointer to array of pointers to rows. */
     return mat;
}

/**  Deallocate vector storage  *********************************/

int free_vector(float *v,int n)

/* Free a float vector allocated by vector(). */
{
    n=0;
	free((char*) (v+1));
	return 0;
}

/**  Deallocate float matrix storage  ***************************/

int free_matrix(float **mat,int n,int m)
/* Free a float matrix allocated by matrix(). */
{
   int i;
   for (i = n; i >= 1; i--)
       {
       free ((char*) (mat[i]+1));
       }
   free ((char*) (mat+1));
   return 0;
}
