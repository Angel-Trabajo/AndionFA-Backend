#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#define OP_GT 0
#define OP_LT 1
#define OP_GE 2
#define OP_LE 3
#define OP_EQ 4
#define OP_NE 5

static inline int eval(double a, int op, double b) {
    switch(op) {
        case OP_GT: return a > b;
        case OP_LT: return a < b;
        case OP_GE: return a >= b;
        case OP_LE: return a <= b;
        case OP_EQ: return a == b;
        case OP_NE: return a != b;
        default: return 0;
    }
}

int main(int argc, char** argv) {

    if (argc != 4) {
        fprintf(stderr, "usage: app matrix.bin conditions.bin mask.bin\n");
        return 1;
    }

    FILE *fm = fopen(argv[1], "rb");
    FILE *fc = fopen(argv[2], "rb");
    FILE *fo = fopen(argv[3], "wb");

    if (!fm || !fc || !fo) {
        fprintf(stderr, "file open error\n");
        return 2;
    }

    int rows, cols;
    fread(&rows, sizeof(int), 1, fm);
    fread(&cols, sizeof(int), 1, fm);

    double *matrix = (double*)malloc(sizeof(double) * rows * cols);
    fread(matrix, sizeof(double), rows * cols, fm);
    fclose(fm);

    int num_nodes;
    fread(&num_nodes, sizeof(int), 1, fc);

    fwrite(&num_nodes, sizeof(int), 1, fo);

    for (int n = 0; n < num_nodes; n++) {

        int num_cond;
        fread(&num_cond, sizeof(int), 1, fc);

        int *col = (int*)malloc(sizeof(int) * num_cond);
        int *op  = (int*)malloc(sizeof(int) * num_cond);
        double *val = (double*)malloc(sizeof(double) * num_cond);

        for (int i = 0; i < num_cond; i++) {
            fread(&col[i], sizeof(int), 1, fc);
            fread(&op[i],  sizeof(int), 1, fc);
            fread(&val[i], sizeof(double), 1, fc);
        }

        uint8_t *mask = (uint8_t*)malloc(rows);

        for (int r = 0; r < rows; r++) {
            int ok = 1;
            for (int c = 0; c < num_cond; c++) {
                double x = matrix[r * cols + col[c]];
                if (!eval(x, op[c], val[c])) {
                    ok = 0;
                    break;
                }
            }
            mask[r] = ok;
        }

        fwrite(&rows, sizeof(int), 1, fo);
        fwrite(mask, sizeof(uint8_t), rows, fo);

        free(col);
        free(op);
        free(val);
        free(mask);
    }

    fclose(fc);
    fclose(fo);
    free(matrix);

    return 0;
}
