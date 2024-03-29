//
// Created by alireza on 3/3/22.
//

#include <cstdint>
#include "iostream"
#include "smm_gem.h"
#include <cmath>

#define D_MODEL 512

#define W_DATA 4
#define MAX_COL 2
#define KERNEL_DIM 8
#define mem2d(data, data_len, row, col)   data[((row)*(data_len))+(col)]

//#define DEVELOP

#ifdef DEVELOP

void add8in32(uint32_t &memory, uint32_t &systolicResult);

/* CM Core Process (MVM)
* Instruction format: |____Opcode___|__rm__|_X|__ra__|__rn__|__rd__|
* Bits:               |31_________21|20__16|15|14__10|9____5|4____0|
* Binary layout:      |0000_0001_000|0_1000|_0|001_11|01_001|0_1010|
* Hex layout:         |__0____1____0|____8_|__|_1____|D____2|____A_|
* gem5 variables:     |_____________|_Op264|__|_Op364|_Op164|Dest64|
*
* Queueing arguments:
* -- rd = Parameter value.
* -- rm = Parameter x index.
* -- ra = Unused.
* -- rn = Parameter y index.
*/

uint64_t smmStream(uint64_t rn) {
    uint64_t res;

    __asm__ volatile(
    "MOV X7, %[input_k];"
    ".long 0x01081D2A;"
    "MOV %[output], X10;"
    : [output] "=r"(res)
    : [input_k] "r"(rn)
    : "x7", "x10"
    );

    return res;
}

/* CM Core Queue (MVM)
* Instruction format: |____Opcode___|__rm__|_X|__ra__|__rn__|__rd__|
* Bits:               |31_________21|20__16|15|14__10|9____5|4____0|
* Binary layout:      |0010_0001_000|0_1000|_0|001_11|01_001|0_1010|
* Hex layout:         |__2____1____0|____8_|__|_1____|D____2|____A_|
* gem5 variables:     |_____________|_Op264|__|_Op364|_Op164|Dest64|
*
* Queueing arguments:
* -- rd = Parameter value.
* -- rm = Parameter x index.
* -- ra = Unused.
* -- rn = Parameter y index.
*/
uint64_t smmQueue(uint64_t rm, uint64_t rn) {
    uint64_t res;

    __asm__ volatile(
    "MOV X9, %[input_j];"
    "MOV X7, %[input_k];"
    ".long 0x21089D2A;"
    "MOV %[output], X10;"
    : [output] "=r"(res)
    : [input_j] "r"(rm), [input_k] "r"(rn)
    : "x7", "x9", "x10"
    );

    return res;
}

/* CM Core Parameter Write
 * Instruction format: |____Opcode___|__rm__|_?|__ra__|__rn__|__rd__|
 * Bits:               |31_________21|20__16|15|14__10|9____5|4____0|
 * Binary layout:      |0100_0001_000|0_1000|_0|001_11|01_001|0_1010|
 * Hex layout:         |__4____1____0|____8_|__|_1____|D____2|____A_|
 * gem5 variables:     |_____________|_Op264|__|_Op364|_Op164|Dest64|
 *
 * Queueing arguments:
 * -- rd = Success code.
 * -- rm = Parameter x index.
 * -- ra = Parameter value.
 * -- rn = Parameter y index.
 */
uint64_t smmParamWrite(uint64_t rm, uint64_t rn, uint64_t ra) {
    uint64_t res;

    __asm__ volatile(
    "MOV X8, %[input_i];"
    "MOV X9, %[input_j];"
    "MOV X7, %[input_k];"
    ".long 0x41081D2A;"
    "MOV %[output], X10;"
    : [output] "=r"(res)
    : [input_i] "r"(ra), [input_j] "r"(rm), [input_k] "r"(rn)
    : "x7", "x8", "x9", "x10"
    );

    return res;

}


void smmCompute(std::size_t seq_len, const uint32_t *input, uint32_t *output, uint32_t *weights,
                std::size_t input_size_, std::size_t output_size_) {

    int ROWS_IN_BLOCK = std::min(128, (int) (seq_len));
    int rowMaxL1 = std::min(64, (int) (input_size_)) / KERNEL_DIM;
    int ratio = 64 / std::min(64, (int) (input_size_));
    int colMaxL1 = std::min(32 * ratio, (int) (output_size_)) / KERNEL_DIM;

    int ROWS_IN_L2 = std::min(512 / ROWS_IN_BLOCK, (int) ceil((float) (seq_len) / (float) ROWS_IN_BLOCK));
    int rowMaxL2 = std::min(256, (int) (input_size_)) / KERNEL_DIM / rowMaxL1;
    int colMaxL2 = std::min(256, (int) (output_size_)) / KERNEL_DIM / colMaxL1;
    std::cout << rowMaxL2 << "\t\t" << colMaxL2 << std::endl;
    for (int l2In = 0; l2In < (int) ceil((float) seq_len / (float) ROWS_IN_BLOCK / (float) ROWS_IN_L2); l2In++) {
        for (int l2Row = 0; l2Row < (input_size_ / KERNEL_DIM) / rowMaxL2 / rowMaxL1; l2Row++) {
            for (int l2Col = 0; l2Col < (output_size_ / KERNEL_DIM) / colMaxL2 / colMaxL1; l2Col++) {
                for (int tileInL2 = 0; tileInL2 < (ROWS_IN_L2); tileInL2++) {
                    for (int tileRowL2 = 0; tileRowL2 < rowMaxL2; tileRowL2++) {
                        for (int tileColL2 = 0; tileColL2 < colMaxL2; tileColL2++) {
                            for (int tileRowL1 = 0; tileRowL1 < rowMaxL1; tileRowL1++) {
                                for (int tileColL1 = 0; tileColL1 < colMaxL1; tileColL1++) {
                                    int tileRow = tileRowL2 * rowMaxL1 + tileRowL1;
                                    int tileCol = tileColL2 * colMaxL1 + tileColL1;
                                    int seqBlockIdx = l2In * ROWS_IN_L2 + tileInL2;
                                    // Load the kernel with the corresponding weight
                                    int rowStart = (l2Row * rowMaxL2 * rowMaxL1 + tileRow) * KERNEL_DIM;
                                    int colStart = (l2Col * colMaxL2 * colMaxL1 + tileCol) * KERNEL_DIM / W_DATA;
                                    int rowBlockSize = KERNEL_DIM;
                                    int colBlockSize = KERNEL_DIM / W_DATA;

                                    //Possible Control Kernel
                                    {
                                        uint32_t *wPtr = weights + rowStart * (output_size_ / W_DATA);
                                        for (int i = rowStart; i < rowStart + rowBlockSize; i++) {
                                            for (int j = colStart; j < colStart + colBlockSize; j++) {
                                                uint32_t weight = *(wPtr + j);
                                                smmParamWrite(i - rowStart, j - colStart, weight);
                                            }
                                            wPtr += output_size_ / W_DATA;
                                        }
                                    }

                                    // Process the multiplication
                                    int base_col_idx = (l2Row * rowMaxL2 * rowMaxL1 + tileRow) * MAX_COL;
                                    int seqBlockLen = std::min(ROWS_IN_BLOCK,
                                                               (int) (seq_len - seqBlockIdx * ROWS_IN_BLOCK));
                                    int outputIndex = 0;
                                    uint32_t *outPtr = output + seqBlockIdx * ROWS_IN_BLOCK * (output_size_ / W_DATA);
                                    uint32_t mult;
                                    const uint32_t *inPtr =
                                            input + base_col_idx + seqBlockIdx * ROWS_IN_BLOCK * (input_size_ / W_DATA);

                                    //Possible Control Kernel
                                    {
                                        for (int i = 0; i < seqBlockLen; i++) {
                                            for (int j = 0; j < MAX_COL; j++) {
                                                if (j == MAX_COL - 1) {
                                                    mult = smmStream(*(inPtr + j));
                                                } else {
                                                    mult = smmQueue(j % MAX_COL, *(inPtr + j));
                                                }

                                                if ((i * MAX_COL + j) >=
                                                    (MAX_COL * (2 * KERNEL_DIM - 1) -
                                                    1)) {    // check if the output is valid
                                                    add8in32(
                                                            mem2d(outPtr, output_size_ / W_DATA, outputIndex / colBlockSize,
                                                                colStart + outputIndex % colBlockSize), mult);
                                                    outputIndex++;
                                                }
                                            }
                                            inPtr += (input_size_ / W_DATA);
                                        }
                                        for (int i = seqBlockLen * MAX_COL;
                                            i < MAX_COL * (seqBlockLen + 2 * KERNEL_DIM - 1) - 1; i++) {
                                            if ((i % MAX_COL) == MAX_COL - 1) {
                                                mult = smmStream(0);
                                            } else {
                                                mult = smmQueue(i % MAX_COL, 0);
                                            }
                                            if (i >= (MAX_COL * (2 * KERNEL_DIM - 1) - 1)) { // check if the output is valid
                                                add8in32(mem2d(outPtr, output_size_ / W_DATA, outputIndex / colBlockSize,
                                                            colStart + outputIndex % colBlockSize), mult);
                                                outputIndex++;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

void add8in32(uint32_t &memory, uint32_t &systolicResult) {
    /*
     * This function separates every 32-bit input to four 8-bit integers and add them. Then, packing again and
     * make a 32-bit  unsigned int.
     */
    auto *mem_ptr = (int8_t *) (&memory);
    auto *sys_ptr = (int8_t *) (&systolicResult);
    for (int i = 0; i < W_DATA; i++) {
        *mem_ptr = (int8_t) (*mem_ptr + *sys_ptr);
        mem_ptr++;
        sys_ptr++;
    }
}

void print_arr(uint32_t *array, int n, int p) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p / W_DATA; j++)
            std::cout << std::hex << (uint32_t) mem2d(array, p / W_DATA, i, j) << "\t";
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

#endif

void MatrixCompute(std::size_t seq_len, const uint32_t *input, uint32_t *output, uint32_t *weight,
                         std::size_t input_size_, std::size_t output_size_) {
    // Loop over each row of the input matrix
    for (std::size_t i = 0; i < seq_len; ++i) {
        // Loop over each column of the weight matrix (and each row of the output matrix)
        for (std::size_t j = 0; j < output_size_; ++j) {
            output[i * output_size_ + j] = 0; // Initialize the output element to 0

            // Perform the dot product
            for (std::size_t k = 0; k < input_size_; ++k) {
                output[i * output_size_ + j] += input[i * input_size_ + k] * weight[k * output_size_ + j];
            }
        }
    }
}

static void SystolicArray(int length, int width_array, int *input, int *weight, int *output, std::size_t output_size_) {
    
    for (int j = 0; j < width_array; j ++) {
        for (int k = 0; k < width_array; k++) {
            int sum = 0;
            for (int i = 0; i < length; i++) {
                sum += *(input + j * length + i) * (*(weight + i * output_size_ + k));
            }
            *(output + j * output_size_ + k) = sum;
        }
    }
}

void SystolicCompute(std::size_t seq_len, const uint32_t *input, uint32_t *output, uint32_t *weight,
                         std::size_t input_size_, std::size_t output_size_) {
    int width_array = 16;
    for (int length = 0; length < seq_len; length+=width_array) { // the length of input token is divided by W_DATA
        for (int out_idx = 0; out_idx < output_size_; out_idx+=width_array) {// the length of output demenson is divided by W_DATA
            auto *weight_ptr = (int32_t *) (weight + out_idx);
            auto *input_ptr  = (int32_t *) (input + (length * input_size_ ));
            auto *output_ptr = (int32_t *) (output + (length * output_size_) + out_idx);
            // The following loop is for the systolic array accelerator
            SystolicArray(input_size_, width_array, (int *) input_ptr, (int *) weight_ptr, (int *) output_ptr, output_size_);
        }
    }
}