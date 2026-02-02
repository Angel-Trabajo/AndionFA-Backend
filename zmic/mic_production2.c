#include <stdio.h>
#include <pthread.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <stdint.h>

#define NUM_THREADS 60  // 60 núcleos por MIC
#define CACHE_LINE_SIZE 64
#define NUM_ALGORITHMS 5  // 5 algoritmos diferentes por elemento

typedef struct {
    long start;
    long end;
    double partial_sum;
    uint64_t partial_iterations;
    char padding[CACHE_LINE_SIZE - sizeof(long)*2 - sizeof(double) - sizeof(uint64_t)];
} thread_data __attribute__((aligned(CACHE_LINE_SIZE)));

// ==================== ALGORITMOS INTENSIVOS ====================

// 1. Serie de Taylor muy larga para sin(x)*cos(x)*exp(x)
double algorithm_1(double x) {
    double result = 0.0;
    double term = 1.0;
    double x_sq = x * x;
    
    // 50 iteraciones para hacerlo muy pesado
    for (int i = 0; i < 50; i++) {
        result += term;
        term *= -x_sq / ((2*i + 1) * (2*i + 2));
        
        // Cálculo extra dentro del bucle
        double temp = term;
        for (int j = 0; j < 10; j++) {
            temp = temp * (1.0 - temp);
        }
        term += temp * 0.01;
    }
    
    // Post-procesamiento pesado
    for (int i = 0; i < 20; i++) {
        result = sqrt(fabs(result)) + log(fabs(result) + 1.0);
    }
    
    return result;
}

// 2. Cálculo de raíces polinómicas complejas
double algorithm_2(double x) {
    double a = x;
    double b = x * 0.5;
    double c = x * 0.25;
    
    // Resolver ecuación cúbica: ax³ + bx² + cx + 1 = 0
    // Usando método iterativo de Newton-Raphson con muchas iteraciones
    double root = 1.0;
    for (int iter = 0; iter < 100; iter++) {
        double fx = a*root*root*root + b*root*root + c*root + 1.0;
        double fpx = 3.0*a*root*root + 2.0*b*root + c;
        
        if (fabs(fpx) < 1e-12) break;
        root = root - fx / fpx;
        
        // Cálculos adicionales por iteración
        double mod = sqrt(root*root + fx*fx);
        for (int j = 0; j < 5; j++) {
            mod = mod * (2.0 - mod * root);
        }
        root = root + mod * 0.001;
    }
    
    return root;
}

// 3. Transformada discreta tipo Fourier (simplificada pero pesada)
double algorithm_3(double x) {
    const int N = 64;  // Tamaño de la transformada
    double real[N], imag[N];
    
    // Inicializar
    for (int i = 0; i < N; i++) {
        real[i] = cos(x * i);
        imag[i] = sin(x * i);
    }
    
    // DFT "manual" - O(N²) intencionalmente
    double magnitude = 0.0;
    for (int k = 0; k < N; k++) {
        double sum_real = 0.0, sum_imag = 0.0;
        
        for (int n = 0; n < N; n++) {
            double angle = 2.0 * M_PI * k * n / N;
            sum_real += real[n] * cos(angle) + imag[n] * sin(angle);
            sum_imag += -real[n] * sin(angle) + imag[n] * cos(angle);
        }
        
        magnitude += sqrt(sum_real*sum_real + sum_imag*sum_imag);
    }
    
    // Normalizar y aplicar función no lineal
    magnitude /= N;
    for (int i = 0; i < 10; i++) {
        magnitude = sin(magnitude) * cos(magnitude) * exp(magnitude * 0.1);
    }
    
    return magnitude;
}

// 4. Generación de números pseudo-aleatorios y estadísticas
double algorithm_4(double x) {
    // Generador LCG con múltiples iteraciones
    uint64_t state = (uint64_t)(x * 1e12);
    double sum = 0.0;
    double sum_sq = 0.0;
    double min_val = 1e100, max_val = -1e100;
    
    // 1000 números aleatorios
    for (int i = 0; i < 1000; i++) {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        double rnd = (double)(state >> 32) / 4294967296.0;
        
        // Transformación Box-Muller para distribución normal
        double u1 = rnd;
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        double u2 = (double)(state >> 32) / 4294967296.0;
        
        double z0 = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
        double z1 = sqrt(-2.0 * log(u1)) * sin(2.0 * M_PI * u2);
        
        sum += z0 + z1;
        sum_sq += z0*z0 + z1*z1;
        min_val = (z0 < min_val) ? z0 : min_val;
        min_val = (z1 < min_val) ? z1 : min_val;
        max_val = (z0 > max_val) ? z0 : max_val;
        max_val = (z1 > max_val) ? z1 : max_val;
    }
    
    double mean = sum / 2000.0;
    double variance = sum_sq / 2000.0 - mean*mean;
    double range = max_val - min_val;
    
    return mean * variance * range;
}

// 5. Sistema de ecuaciones diferenciales simples (Euler)
double algorithm_5(double x) {
    // Sistema: dy1/dt = -y2, dy2/dt = y1
    double y1 = 1.0, y2 = 0.0;
    double dt = 0.001;
    int steps = 1000;
    
    for (int t = 0; t < steps; t++) {
        double dy1 = -y2 * dt;
        double dy2 = y1 * dt;
        
        // Método de Euler mejorado (Heun)
        double y1_temp = y1 + dy1;
        double y2_temp = y2 + dy2;
        
        double dy1_corr = -y2_temp * dt;
        double dy2_corr = y1_temp * dt;
        
        y1 += 0.5 * (dy1 + dy1_corr);
        y2 += 0.5 * (dy2 + dy2_corr);
        
        // Acoplamiento no lineal
        double coupling = sin(x * t * dt) * 0.01;
        y1 += coupling * y2;
        y2 -= coupling * y1;
    }
    
    return sqrt(y1*y1 + y2*y2);
}

// Función principal que ejecuta TODOS los algoritmos
double execute_all_algorithms(double x) {
    double result = 0.0;
    
    // Ejecutar todos los algoritmos y combinar resultados
    result += algorithm_1(x);
    result += algorithm_2(x * 1.1);
    result += algorithm_3(x * 0.9);
    result += algorithm_4(x * 1.2);
    result += algorithm_5(x * 0.8);
    
    // Post-procesamiento final intensivo
    for (int i = 0; i < 50; i++) {
        result = sin(result) * cos(result) + 
                 sqrt(fabs(result)) * log(fabs(result) + 1.0);
        
        // Cálculo de punto fijo adicional
        double temp = result;
        for (int j = 0; j < 10; j++) {
            temp = 0.5 * (temp + 2.0 / (temp + 1e-10));
        }
        result += temp * 0.1;
    }
    
    return result;
}

// ==================== FUNCIÓN DEL HILO ====================

void* compute_chunk(void* arg) {
    thread_data* data = (thread_data*)arg;
    data->partial_sum = 0.0;
    data->partial_iterations = 0;
    
    for (long i = data->start; i < data->end; i++) {
        double x = (double)i * 1e-7;
        
        // Ejecutar TODOS los algoritmos para cada elemento
        data->partial_sum += execute_all_algorithms(x);
        
        // Contador de operaciones
        data->partial_iterations += 
            50 + 100 + 64*64 + 1000*2 + 1000*2 + 50*10;
    }
    
    pthread_exit(NULL);
}

// ==================== PROGRAMA PRINCIPAL ====================

int main(int argc, char** argv) {
    // Configuración - ¡MUY GRANDE por defecto!
    long total_size = 5000000;  // 5 millones de elementos
    if (argc > 1) {
        total_size = atol(argv[1]);
    }
    
    printf("╔══════════════════════════════════════════════════════════╗\n");
    printf("║      BENCHMARK EXTREMO - INTEL XEON PHI STRESS TEST      ║\n");
    printf("╚══════════════════════════════════════════════════════════╝\n");
    
    printf("\nCONFIGURACIÓN:\n");
    printf("  Hilos por MIC: %d\n", NUM_THREADS);
    printf("  Elementos totales: %'ld\n", total_size);
    printf("  Elementos por hilo: %'ld\n", total_size / NUM_THREADS);
    printf("  Algoritmos por elemento: %d\n", NUM_ALGORITHMS);
    
    // Calcular estimación de operaciones
    uint64_t total_ops_est = (uint64_t)total_size * 
        (50 + 100 + 64*64 + 1000*2 + 1000*2 + 50*10);
    printf("  Operaciones estimadas: %'llu\n", total_ops_est);
    
    // Inicializar
    pthread_t threads[NUM_THREADS];
    thread_data thread_data_array[NUM_THREADS];
    
    long chunk_size = total_size / NUM_THREADS;
    
    printf("\nINICIANDO BENCHMARK...\n");
    printf("╔══════════════════════════════════════════════════════════╗\n");
    
    // Timer de alta precisión
    struct timespec start_time, end_time;
    clock_gettime(CLOCK_MONOTONIC, &start_time);
    
    // Crear hilos
    for (int i = 0; i < NUM_THREADS; i++) {
        thread_data_array[i].start = i * chunk_size;
        thread_data_array[i].end = (i == NUM_THREADS - 1) ? 
            total_size : thread_data_array[i].start + chunk_size;
        
        int rc = pthread_create(&threads[i], NULL, compute_chunk, 
                               &thread_data_array[i]);
        if (rc) {
            printf("Error creando hilo %d\n", i);
            return 1;
        }
    }
    
    // Esperar y recolectar
    double global_sum = 0.0;
    uint64_t total_iterations = 0;
    
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
        global_sum += thread_data_array[i].partial_sum;
        total_iterations += thread_data_array[i].partial_iterations;
    }
    
    // Timer end
    clock_gettime(CLOCK_MONOTONIC, &end_time);
    
    double elapsed_time = (end_time.tv_sec - start_time.tv_sec) +
                         (end_time.tv_nsec - start_time.tv_nsec) / 1e9;
    
    printf("╚══════════════════════════════════════════════════════════╝\n");
    
    // ==================== RESULTADOS DETALLADOS ====================
    
    printf("\n╔══════════════════════════════════════════════════════════╗\n");
    printf("║                     RESULTADOS FINALES                    ║\n");
    printf("╚══════════════════════════════════════════════════════════╝\n");
    
    printf("\nMÉTRICAS DE RENDIMIENTO:\n");
    printf("  ┌────────────────────────────────────────────┐\n");
    printf("  │ Tiempo total:        %12.3f segundos │\n", elapsed_time);
    printf("  │ Elementos procesados: %'12ld         │\n", total_size);
    printf("  │ Suma final:          %12.6e      │\n", global_sum);
    printf("  │ Operaciones totales:  %'12llu         │\n", total_iterations);
    printf("  └────────────────────────────────────────────┘\n");
    
    // Cálculos de rendimiento
    double elements_per_sec = total_size / elapsed_time;
    double ops_per_sec = total_iterations / elapsed_time;
    double flops = ops_per_sec / 1e9;  // GigaFLOPS
    double efficiency = (elements_per_sec / (NUM_THREADS * 1e6)) * 100;
    
    printf("\nRENDIMIENTO:\n");
    printf("  ┌────────────────────────────────────────────┐\n");
    printf("  │ Throughput:        %8.2f Millones elem/s │\n", 
           elements_per_sec / 1e6);
    printf("  │ Operaciones/seg:   %8.2f Millones ops/s  │\n", 
           ops_per_sec / 1e6);
    printf("  │ Potencia cálculo:  %8.2f GigaFLOPS       │\n", flops);
    printf("  │ Eficiencia:        %8.2f %%              │\n", efficiency);
    printf("  │ Hilos activos:     %8d / %-2d           │\n", 
           NUM_THREADS, NUM_THREADS);
    printf("  └────────────────────────────────────────────┘\n");
    
    // Estadísticas por núcleo (si hay tiempo)
    if (elapsed_time > 2.0) {
        printf("\nESTADÍSTICAS POR NÚCLEO (aprox):\n");
        printf("  ┌────────────────────────────────────────────┐\n");
        for (int i = 0; i < 4; i++) {  // Mostrar primeros 4 como muestra
            double chunk_work = thread_data_array[i].partial_sum;
            printf("  │ Núcleo %2d: %12.6e unidades trabajo │\n", 
                   i, chunk_work);
        }
        printf("  │ ... (otros %d núcleos activos)            │\n", 
               NUM_THREADS - 4);
        printf("  └────────────────────────────────────────────┘\n");
    }
    
    // Recomendaciones basadas en resultados
    printf("\nANÁLISIS:\n");
    if (elapsed_time < 5.0) {
        printf("  ⚡  El benchmark fue muy rápido. Incrementa el tamaño.\n");
        printf("  💡  Sugerencia: Ejecutar con 20-50 millones de elementos.\n");
    } else if (elapsed_time < 30.0) {
        printf("  ✅  Benchmark en rango ideal para mediciones.\n");
        printf("  📊  Buen balance entre tiempo y carga de trabajo.\n");
    } else {
        printf("  🐢  Benchmark muy largo. Considera reducir el tamaño.\n");
        printf("  ⏱️  Tiempo adecuado para pruebas de estabilidad.\n");
    }
    
    printf("\n╔══════════════════════════════════════════════════════════╗\n");
    printf("║     BENCHMARK COMPLETADO - 240 NÚCLEOS EN ACCIÓN!        ║\n");
    printf("╚══════════════════════════════════════════════════════════╝\n");
    
    return 0;
}