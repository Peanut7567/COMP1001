/*
------------------DR VASILIOS KELEFOURAS-----------------------------------------------------
------------------COMP1001 ------------------------------------------------------------------
------------------COMPUTER SYSTEMS MODULE-------------------------------------------------
------------------UNIVERSITY OF PLYMOUTH, SCHOOL OF ENGINEERING, COMPUTING AND MATHEMATICS---
*/


#include <stdio.h>
#include <time.h>
#include <pmmintrin.h>
#include <process.h>
#include <chrono>
#include <iostream>
#include <immintrin.h>
#include <omp.h>

#define M 1024*512
#define ARITHMETIC_OPERATIONS1 3*M
#define TIMES1 1

#define N 8192
#define ARITHMETIC_OPERATIONS2 4*N*N
#define TIMES2 1


//function declaration
void initialize();
void routine1(float alpha, float beta);
void routine2(float alpha, float beta);
void routine1_vec(float alpha, float beta);
void routine2_vec(float alpha, float beta);

__declspec(align(64)) float  y[M], z[M];
__declspec(align(64)) float A[N][N], x[N], w[N];

int main() {

    float alpha = 0.023f, beta = 0.045f;
    double run_time, start_time;
    unsigned int t;

    initialize();

    printf("\nRoutine1:");
    start_time = omp_get_wtime(); //start timer

    for (t = 0; t < TIMES1; t++)
        routine1(alpha, beta);

    run_time = omp_get_wtime() - start_time; //end timer
    printf("\n Time elapsed is %f secs \n %e FLOPs achieved\n", run_time, (double)(ARITHMETIC_OPERATIONS1) / ((double)run_time / TIMES1));

    printf("\nRoutine2:");
    start_time = omp_get_wtime(); //start timer

    for (t = 0; t < TIMES2; t++)
        routine2(alpha, beta);

    run_time = omp_get_wtime() - start_time; //end timer
    printf("\n Time elapsed is %f secs \n %e FLOPs achieved\n", run_time, (double)(ARITHMETIC_OPERATIONS2) / ((double)run_time / TIMES2));

    initialize();

    printf("\nRoutine1_Vec:");
    start_time = omp_get_wtime(); //start timer

    for (t = 0; t < TIMES1; t++)
        routine1_vec(alpha, beta);

    run_time = omp_get_wtime() - start_time; //end timer
    printf("\n Time elapsed is %f secs \n %e FLOPs achieved\n", run_time, (double)(ARITHMETIC_OPERATIONS1) / ((double)run_time / TIMES1));

    printf("\nRoutine2_Vec:");
    start_time = omp_get_wtime(); //start timer

    for (t = 0; t < TIMES2; t++)
        routine2_vec(alpha, beta);

    run_time = omp_get_wtime() - start_time; //end timer
    printf("\n Time elapsed is %f secs \n %e FLOPs achieved\n", run_time, (double)(ARITHMETIC_OPERATIONS2) / ((double)run_time / TIMES2));

    return 0;
}

void initialize() {

    unsigned int i, j;

    //initialize routine2 arrays
    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++) {
            A[i][j] = (i % 99) + (j % 14) + 0.013f;
        }


    //initialize routine1 arrays
    for (i = 0; i < N; i++) {
        x[i] = (i % 19) - 0.01f;
        w[i] = (i % 5) - 0.002f;
    }

    //initialize routine1 arrays
    for (i = 0; i < M; i++) {
        z[i] = (i % 9) - 0.08f;
        y[i] = (i % 19) + 0.07f;
    }


}




void routine1(float alpha, float beta) {

    unsigned int i;


    for (i = 0; i < M; i++)
        y[i] = alpha * y[i] + beta * z[i];

}


//VECTORISATION OF ROUTINE 1:

void routine1_vec(float alpha, float beta) {

    __m128 num1, num2, num3, num4, num5, num6, num7;   //Declaring variables
    unsigned int i, j;

    num1 = _mm_set1_ps(alpha);            //Converts alpha into SSE
    num2 = _mm_set1_ps(beta);             //Converts beta into SSE

    for (i = 0; i < M; i += 4) {
        num3 = _mm_load_ps(&y[i]);          //Loads value of y[i] into num1
        num4 = _mm_load_ps(&z[i]);          //Loads value of z[i] into num2

        num5 = _mm_mul_ps(num1, num3);     //Multiplies alpha by num1 and stores result into num3
        num6 = _mm_mul_ps(num2, num4);      //Multiplies beta by num2 and stores result into num4
        num7 = _mm_add_ps(num5, num6);      //Adds num3 to num4 and stores result into num5

        _mm_store_ps(&y[i], num7);          //Stores value of num5 into y[i]

    }

    //Add additional loop for if M is not divisible by 4 to loop through the remaining floats

    for (j = i; i < M; i++) {

        y[i] = alpha * y[i] + beta * z[i];

    }
}






void routine2(float alpha, float beta) {

    unsigned int i, j;


    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++)
            w[i] = w[i] - beta + alpha * A[i][j] * x[j];


}


//VECTORISATION OF ROUTINE 2:

void routine2_vec(float alpha, float beta) {

    __m128 num1, num2, num3, num4, num5, num6, num7, num8, num9;    //Declaring variables
    unsigned int i, j;


    for (i = 0; i < N; i += 4) {
        num1 = _mm_set1_ps(alpha);                  //Converts alpha into SSE
        num2 = _mm_set1_ps(beta);                   //Comverts beta into SSE

        for (j = 0; j < N; j++) {

            num3 = _mm_loadu_ps(&A[i][j]);          //Loads value of A[i][j] into num3
            num4 = _mm_set1_ps(x[j]);               //Converts x[j] into SSE
            num5 = _mm_loadu_ps(&w[i]);             //Loads value of w[i] into num5

            num6 = _mm_sub_ps(num5, num2);          //Subtracts num2 from num5 and stores result into num6
            num7 = _mm_mul_ps(num1, num3);          //Multiplies num1 by num3 and stores result into num7
            num8 = _mm_mul_ps(num7, num4);          //Multiplies num7 by num4 and stores result into num8
            num9 = _mm_add_ps(num6, num8);          //Adds num6 to num8 and stores result in num9

            _mm_storeu_ps(&w[i], num9);             //Stores value of num9 into w[i]
        }
    }

    //Add additional loop for if N is not divisible by 4 to loop through the remaining floats

    for (j = i; i < N; i++) {
        for (j = 0; j < N; j++) {

            w[i] = w[i] - beta + alpha * A[i][j] * x[j];

        }
    }
}
