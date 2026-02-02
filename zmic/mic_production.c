#include <stdio.h>
#include <pthread.h>
#include <stdlib.h>
#include <time.h>

#define NUM_THREADS 60  // 60 núcleos por MIC
#define CACHE_LINE_SIZE 64  // Para evitar false sharing

typedef struct {
    long start;
    long end;
    double partial_sum;
    char padding[CACHE_LINE_SIZE - sizeof(long) * 2 - sizeof(double)];
} thread_data __attribute__((aligned(CACHE_LINE_SIZE)));

// Función de cálculo intensivo (personalizable)
double intensive_computation(double x) {
    // Ejemplo: Serie matemática intensiva
    double result = 0.0;
    double term = x;
    
    // 20 iteraciones para hacerlo más intensivo
    for (int i = 1; i <= 20; i++) {
        result += term;
        term = -term * x * x / ((2*i) * (2*i + 1));
    }
    
    return result;
}

void* compute_chunk(void* arg) {
    thread_data* data = (thread_data*)arg;
    data->partial_sum = 0.0;
    
    for (long i = data->start; i < data->end; i++) {
        double x = (double)i * 1e-7;  // Escala más pequeña para precisión
        data->partial_sum += intensive_computation(x);
    }
    
    pthread_exit(NULL);
}

int main(int argc, char** argv) {
    // Configuración
    long total_size = 100000000;  // 10 millones por defecto
    if (argc > 1) {
        total_size = atol(argv[1]);
    }
    
    printf("=== MIC PARALLEL COMPUTATION ===\n");
    printf("Configuration:\n");
    printf("  Threads per MIC: %d\n", NUM_THREADS);
    printf("  Total elements: %ld\n", total_size);
    printf("  Elements per thread: %ld\n", total_size / NUM_THREADS);
    
    // Inicializar datos de hilos
    pthread_t threads[NUM_THREADS];
    thread_data thread_data_array[NUM_THREADS];
    
    long chunk_size = total_size / NUM_THREADS;
    
    // Timer start
    clock_t start_time = clock();
    
    // Crear hilos
    for (int i = 0; i < NUM_THREADS; i++) {
        thread_data_array[i].start = i * chunk_size;
        thread_data_array[i].end = (i == NUM_THREADS - 1) ? total_size : thread_data_array[i].start + chunk_size;
        
        pthread_create(&threads[i], NULL, compute_chunk, &thread_data_array[i]);
    }
    
    // Esperar hilos y recolectar
    double global_sum = 0.0;
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
        global_sum += thread_data_array[i].partial_sum;
    }
    
    // Timer end
    clock_t end_time = clock();
    double elapsed_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    
    // Resultados
    printf("\nResults:\n");
    printf("  Final sum: %.15f\n", global_sum);
    printf("  Execution time: %.3f seconds\n", elapsed_time);
    printf("  Performance: %.0f elements/second\n", total_size / elapsed_time);
    printf("  Threads used: %d\n", NUM_THREADS);
    
    return 0;
}

