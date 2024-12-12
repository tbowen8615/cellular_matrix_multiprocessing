"""
=======================================================================================================================
Title           : Modified Cellular Life Simulator with Multiprocessing
Description     : This program is designed to simulate the evolution of a cellular matrix over 100 iterations based on
                    a set of predefined rules. The program supports parallel processing using Python's multiprocessing
                    module, enabling it to handle both small and large matrices efficiently.
Author          : Tyler Bowen
Date            : 11/19/2024
Version         : 1
Usage           : In the Command Prompt, navigate to the directory containing this python script, then enter the
                    following command:

                    python3 Cellular_Life_Matrix_Multiprocessing.py -i <input_file> -o <output_file> -p <process_count>

                    The program will take an input matrix, split it into a number of chunks to be processed
                    concurrently, and then write a final matrix to an output file.
Python Version  : 3.12.1
=======================================================================================================================
"""

import argparse
import os
import sys
from multiprocessing import Process, Queue

# Static Prime Set (Primes up to 16)
PRIME_SET = {2, 3, 5, 7, 11, 13}

def is_prime(num):
    return num in PRIME_SET

# Argument Parsing
def parse_arguments():
    parser = argparse.ArgumentParser(description="Modified Cellular Life Simulator")
    parser.add_argument("-i", required=True, help="Path to input file")
    parser.add_argument("-o", required=True, help="Path to output file")
    parser.add_argument("-p", type=int, default=1, help="Number of processes to spawn (default: 1)")
    args = parser.parse_args()

    # Validate input file path
    if not os.path.isfile(args.i):
        print("Error: Input file does not exist.")
        sys.exit(1)

    # Validate output file directory
    output_dir = os.path.dirname(args.o)
    if output_dir and not os.path.exists(output_dir):
        print("Error: Output directory does not exist.")
        sys.exit(1)

    # Validate process count
    if args.p < 1:
        print("Error: Process count must be greater than 0.")
        sys.exit(1)

    return args

# Reading and Writing the Matrix
def read_matrix(input_file):
    with open(input_file, "r") as f:
        lines = f.readlines()

    matrix = []
    valid_symbols = {'O', 'o', 'X', 'x', '.'}

    for line in lines:
        row = line.strip()
        if not all(cell in valid_symbols for cell in row):
            print(f"Error: Invalid symbol found in input file.")
            sys.exit(1)
        matrix.append(list(row))

    return matrix

def write_matrix(output_file, matrix):
    with open(output_file, "w") as f:
        for row in matrix:
            f.write("".join(row) + "\n")

# Neighbor Sum Calculation
def sum_neighbors(matrix, row, col):
    rows, cols = len(matrix), len(matrix[0])
    neighbor_sum = 0

    for i in range(row - 1, row + 2):
        for j in range(col - 1, col + 2):
            if (i == row and j == col) or i < 0 or j < 0 or i >= rows or j >= cols:
                continue
            cell = matrix[i][j]
            if cell == 'O':
                neighbor_sum += 2
            elif cell == 'o':
                neighbor_sum += 1
            elif cell == 'X':
                neighbor_sum -= 2
            elif cell == 'x':
                neighbor_sum -= 1
            # Dead cells contribute 0, so no need to add explicitly
    return neighbor_sum

# Splitting the Matrix with Boundaries
def split_matrix_with_boundaries(matrix, num_chunks):
    chunk_size = len(matrix) // num_chunks
    chunks = []

    for i in range(num_chunks):
        start = i * chunk_size
        end = start + chunk_size if i != num_chunks - 1 else len(matrix)  # Last chunk takes the remaining rows
        chunk = matrix[start:end]

        # Include boundary rows (if they exist)
        boundary_above = matrix[start - 1] if start > 0 else None
        boundary_below = matrix[end] if end < len(matrix) else None

        chunks.append((i, start, end, chunk, boundary_above, boundary_below))

    return chunks

# Processing Chunks with Boundaries
def process_chunk_with_boundaries(chunk_id, start_row, end_row, chunk, boundary_above, boundary_below, matrix_cols, result_queue):
    rows = len(chunk)
    new_chunk = [row.copy() for row in chunk]

    # Include boundary rows in neighbor calculations
    extended_chunk = []
    if boundary_above:
        extended_chunk.append(boundary_above)
    extended_chunk.extend(chunk)
    if boundary_below:
        extended_chunk.append(boundary_below)

    # Process each cell in the chunk
    for r in range(rows):
        for c in range(matrix_cols):
            neighbor_sum = sum_neighbors(extended_chunk, r + (1 if boundary_above else 0), c)

            cell = chunk[r][c]
            if cell == 'O':
                # Rule 2a: If the sum is a power of 2, cell dies
                if neighbor_sum > 0 and (neighbor_sum & (neighbor_sum - 1)) == 0:
                    new_chunk[r][c] = '.'
                # Rule 2b: If the sum is less than 10, becomes weakened 'o'
                elif neighbor_sum < 10:
                    new_chunk[r][c] = 'o'
                # Rule 2c: Else, remains unchanged
            elif cell == 'o':
                # Rule 3a: If the sum is <= 0, cell dies
                if neighbor_sum <= 0:
                    new_chunk[r][c] = '.'
                # Rule 3b: If the sum is >= 8, becomes healthy 'O'
                elif neighbor_sum >= 8:
                    new_chunk[r][c] = 'O'
                # Rule 3c: Else, remains unchanged
            elif cell == '.':
                # Rule 4a: If the sum is a prime number, becomes weakened 'o'
                if is_prime(neighbor_sum):
                    new_chunk[r][c] = 'o'
                # Rule 4b: If the absolute value of the sum is a prime number, becomes weakened 'x'
                elif is_prime(abs(neighbor_sum)):
                    new_chunk[r][c] = 'x'
                # Rule 4c: Else, remains unchanged
            elif cell == 'x':
                # Rule 5a: If the sum >= 1, cell dies
                if neighbor_sum >= 1:
                    new_chunk[r][c] = '.'
                # Rule 5b: If the sum <= -8, becomes healthy 'X'
                elif neighbor_sum <= -8:
                    new_chunk[r][c] = 'X'
                # Rule 5c: Else, remains unchanged
            elif cell == 'X':
                # Rule 6a: If the absolute value of the sum is a power of 2, cell dies
                abs_sum = abs(neighbor_sum)
                if abs_sum > 0 and (abs_sum & (abs_sum - 1)) == 0:
                    new_chunk[r][c] = '.'
                # Rule 6b: If the sum is greater than -10, becomes weakened 'x'
                elif neighbor_sum > -10:
                    new_chunk[r][c] = 'x'
                # Rule 6c: Else, remains unchanged

    # Send the updated chunk back via the result queue
    result_queue.put((chunk_id, new_chunk))

# Simulation with Multiprocessing
def simulate_parallel(matrix, iterations=100, num_processes=1):
    for _ in range(iterations):
        chunks = split_matrix_with_boundaries(matrix, num_processes)
        result_queue = Queue()
        processes = []

        # Spawn processes to handle each chunk
        for chunk_id, start_row, end_row, chunk, boundary_above, boundary_below in chunks:
            p = Process(
                target=process_chunk_with_boundaries,
                args=(chunk_id, start_row, end_row, chunk, boundary_above, boundary_below, len(matrix[0]), result_queue)
            )
            processes.append(p)
            p.start()

        # Collect results from the processes
        results = []
        for _ in range(len(chunks)):
            results.append(result_queue.get())

        # Ensure all processes are finished
        for p in processes:
            p.join()

        # Sort results by chunk ID and merge the matrix
        results.sort(key=lambda x: x[0])  # Sort by chunk_id
        matrix = [row for _, chunk in results for row in chunk]

    return matrix

# Main Function
def main():
    args = parse_arguments()
    print("Project :: R11474743")

    matrix = read_matrix(args.i)
    num_processes = args.p
    matrix = simulate_parallel(matrix, iterations=100, num_processes=num_processes)
    write_matrix(args.o, matrix)

if __name__ == "__main__":
    main()
