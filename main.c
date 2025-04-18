/**
 * @file main.c
 * @brief Simple XOR neural network using fixed-point arithmetic
 * @author Badger Code
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// Define constants
#define INPUTS 2
#define HIDDEN 2
#define OUTPUTS 1
#define FIXED_SCALE 4.0f // 2^2 for 4-bit (Q2.2)
#define LR 0.5f
#define EPOCHS 1000
#define UPDATE_INTERVAL 100

// XOR dataset
float inputs[4][2] = {
	{0, 0},
	{0, 1},
	{1, 0},
	{1, 1}
};
float targets[4] = {0, 1, 1, 0};

/**
 * @brief Convert float to fixed-point representation
 * @param x Float value to convert
 * @return Fixed-point representation of the float value
 */
int float_to_fixed4(float x) {
	int q = (int)roundf(x * FIXED_SCALE);
	if (q < -8) q = -8;
	if (q > 7) q = 7;
	return q;
}
/**
 * @brief Convert fixed-point representation to float
 * @param x Fixed-point value to convert
 * @return Float representation of the fixed-point value
 */
float fixed4_to_float(int x) {
	return x / FIXED_SCALE;
}

/**
 * @brief Sigmoid activation function
 * @param x Input value
 * @return Sigmoid of the input value
 */
float sigmoid(float x) {
	if (x < -6.0f) x = -6.0f;
	if (x >  6.0f) x =  6.0f;
	return 1.0f / (1.0f + expf(-x));
}
/**
 * @brief Derivative of sigmoid function
 * @param y Output of sigmoid function
 * @return Derivative of sigmoid function
 */
float dsigmoid(float y) {
	return y * (1.0f - y); // y = sigmoid(x)
}

// Network weights
int w_input_hidden[HIDDEN][INPUTS], b_hidden[HIDDEN];
int w_hidden_output[OUTPUTS][HIDDEN], b_output[OUTPUTS];

float w_input_hidden_f[HIDDEN][INPUTS], b_hidden_f[HIDDEN];
float w_hidden_output_f[OUTPUTS][HIDDEN], b_output_f[OUTPUTS];

/**
 * @brief Generate a random weight in the range [-1.0, 1.0]
 * @return Random weight
 */
float rand_weight() {
	return ((rand() % 9 - 4) / FIXED_SCALE); // -1.0 to 1.0 in 0.25 steps
}

/**
 * @brief Initialize weights and biases
 * @details Weights are initialized randomly in the range [-1.0, 1.0]
 * 	Biases are initialized to 0.0
 * @note The weights and biases are stored in both fixed-point and float representations
 * @warning The weights are stored in fixed-point format to save memory
 */
void init_weights() {
	srand((unsigned int)time(NULL));
	for (int i = 0; i < HIDDEN; ++i) {
		for (int j = 0; j < INPUTS; ++j) {
			float w = rand_weight();
			w_input_hidden_f[i][j] = w;
			w_input_hidden[i][j] = float_to_fixed4(w);
		}
		float b = rand_weight();
		b_hidden_f[i] = b;
		b_hidden[i] = float_to_fixed4(b);
	}
	for (int i = 0; i < OUTPUTS; ++i) {
		for (int j = 0; j < HIDDEN; ++j) {
			float w = rand_weight();
			w_hidden_output_f[i][j] = w;
			w_hidden_output[i][j] = float_to_fixed4(w);
		}
		float b = rand_weight();
		b_output_f[i] = b;
		b_output[i] = float_to_fixed4(b);
	}
}

/**
 * @brief Forward pass through the network
 * @param in Input array
 * @param hidden_out Output array for hidden layer
 * @param out Output value
 */
void forward(float in[INPUTS], float hidden_out[HIDDEN], float *out) {
	for (int i = 0; i < HIDDEN; ++i) {
		float sum = b_hidden_f[i];
		for (int j = 0; j < INPUTS; ++j) {
			sum += w_input_hidden_f[i][j] * in[j];
		}
		hidden_out[i] = sigmoid(sum);
	}
	float sum = b_output_f[0];
	for (int i = 0; i < HIDDEN; ++i) {
		sum += w_hidden_output_f[0][i] * hidden_out[i];
	}
	*out = sigmoid(sum);
}

/**
 * @brief Train the network on a single sample
 * @param in Input array
 * @param target Target output value
 * @details The function performs a forward pass, calculates the error, and updates the weights and biases
 */
void train_sample(float in[INPUTS], float target) {
	float hidden_out[HIDDEN], output;
	forward(in, hidden_out, &output);

	float error = target - output;
	float d_output = error * dsigmoid(output);

	float d_hidden[HIDDEN];
	for (int i = 0; i < HIDDEN; ++i)
		d_hidden[i] = d_output * w_hidden_output_f[0][i] * dsigmoid(hidden_out[i]);

	for (int i = 0; i < HIDDEN; ++i) {
		w_hidden_output_f[0][i] += LR * d_output * hidden_out[i];
		w_hidden_output[0][i] = float_to_fixed4(w_hidden_output_f[0][i]);
	}
	b_output_f[0] += LR * d_output;
	b_output[0] = float_to_fixed4(b_output_f[0]);

	for (int i = 0; i < HIDDEN; ++i) {
		for (int j = 0; j < INPUTS; ++j) {
			w_input_hidden_f[i][j] += LR * d_hidden[i] * in[j];
			w_input_hidden[i][j] = float_to_fixed4(w_input_hidden_f[i][j]);
		}
		b_hidden_f[i] += LR * d_hidden[i];
		b_hidden[i] = float_to_fixed4(b_hidden_f[i]);
	}
}

/**
 * @brief Train the network for a number of epochs
 * @details The function iterates over the training samples and updates the weights and biases
 */
void train() {
	printf("--- TRAINING ---\n");
	for (int epoch = 0; epoch <= EPOCHS; ++epoch) {
		float loss = 0.0f;
		for (int i = 0; i < 4; ++i) {
			train_sample(inputs[i], targets[i]);
			float hidden[HIDDEN], out;
			forward(inputs[i], hidden, &out);
			loss += 0.5f * (targets[i] - out) * (targets[i] - out);
		}
		if (epoch % UPDATE_INTERVAL == 0)
			printf("Epoch %5d | Loss: %.6f\n", epoch, loss);
	}
}

/**
 * @brief Save the model to a file
 * @param filename Name of the file to save the model
 */
void save_model(const char* filename) {
	FILE* file = fopen(filename, "w");
	if (file == NULL) {
		perror("Failed to open file");
		return;
	}
	for (int i = 0; i < HIDDEN; ++i) {
		for (int j = 0; j < INPUTS; ++j) {
			fprintf(file, "%d ", w_input_hidden[i][j]);
		}
		fprintf(file, "%d\n", b_hidden[i]);
	}
	fprintf(file, "\n");
	for (int i = 0; i < OUTPUTS; ++i) {
		for (int j = 0; j < HIDDEN; ++j) {
			fprintf(file, "%d ", w_hidden_output[i][j]);
		}

		if (i == OUTPUTS - 1) {
			fprintf(file, "%d", b_output[i]);
		} else {
			fprintf(file, "%d\n", b_output[i]);
		}
	}
	fclose(file);
}

/**
 * @brief Test the network on the XOR problem
 * @details The function performs a forward pass for each input and prints the output
 */
void test_xor() {
	printf("\n--- XOR TEST ---\n");
	for (int i = 0; i < 4; ++i) {
		float hidden[HIDDEN], out;
		forward(inputs[i], hidden, &out);
		int out_round = (out > 0.5f) ? 1.0f : 0.0f;
		printf("Input: %.0f %.0f => Output: %.4f (%d)\n", inputs[i][0], inputs[i][1], out, out_round);
	}
}

/**
 * @brief Main function
 * @param argc Argument count
 * @param argv Argument vector
 * @return Exit status: 0 on success, non-zero on failure
 * @details Initializes weights, trains the network, tests it on the XOR problem, and saves the model
 */
int main(int argc, char** argv) {
	init_weights();
	train();
	test_xor();
	save_model("./xor_model.bin");
	return 0;
}