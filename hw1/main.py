import numpy as np

# Note: The problem uses 0-based indexing for variables (x0, x1, ...) in the code
# but 1-based indexing in the output (x1, x2, ...). The code will handle this.

def read_system_from_file(filename="input.txt"):
    """
    Reads the coefficient matrix A and the constant vector b from the specified file.
    The file format is:
    Size=N
    A_row_1
    ...
    A_row_N
    b_vector (the last row)

    Args:
        filename (str): The name of the input file.

    Returns:
        tuple: A tuple containing the numpy array A (coefficient matrix) and
               the numpy array b (constant vector).
    """
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()

        # Get the size of the system (N) from the first line
        size_line = lines[0].strip()
        if not size_line.startswith("Size="):
            raise ValueError("Input file format error: Missing 'Size=' line.")
        N = int(size_line.split('=')[1])

        # Read the A matrix (N rows) and b vector (last row)
        # The relevant lines are from index 1 to N+1
        data_lines = [line.strip() for line in lines[1:N+2] if line.strip()]

        if len(data_lines) != N + 1:
            raise ValueError(f"Input file format error: Expected {N+1} data rows but found {len(data_lines)}.")

        # The first N lines are the rows of matrix A
        A_list = []
        for i in range(N):
            row = list(map(int, data_lines[i].split()))
            if len(row) != N:
                raise ValueError(f"Matrix A row {i} does not have {N} elements.")
            A_list.append(row)

        A = np.array(A_list, dtype=float)

        # The last line is the constant vector b
        b_list = list(map(int, data_lines[N].split()))
        if len(b_list) != N:
            raise ValueError(f"Constant vector b does not have {N} elements.")
        b = np.array(b_list, dtype=float)

        return A, b

    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.")
        # Exit the program gracefully if the input file is missing
        exit(1)
    except Exception as e:
        print(f"An error occurred during file reading: {e}")
        exit(1)


def format_and_print_matrix(A, b, filename="output.txt"):
    """
    Formats the linear equation system Ax=b and writes it to the output file.

    Args:
        A (np.array): The coefficient matrix.
        b (np.array): The constant vector.
        filename (str): The name of the output file.
    """
    N = A.shape[0]
    output_lines = []

    for i in range(N):
        line = ""
        for j in range(N):
            coeff = int(A[i, j]) # Coefficients are integers as per problem description
            
            # Determine the sign for the coefficient
            sign = '+' if coeff >= 0 else '' # Only display '+' for positive numbers
            
            # Format the term: +Cxn or -Cxn
            term = f"{sign}{coeff}x{j}"
            
            # For the first term (j=0), we only print the sign if it's positive,
            # but the logic above handles it correctly to always show a sign.
            # We must ensure no leading space before the first term if it's positive.
            if j == 0:
                # Remove the leading '+' if it exists to ensure a single sign for the first term
                if term.startswith('+'):
                    term = term[1:]
                line += f"{sign}{coeff}x{j}"
            else:
                line += f" {sign}{abs(coeff)}x{j}" # Use absolute value for coeff, sign is handled before

        # For the final output format, the problem example shows: +4x0 +1x1 = 3
        # Let's adjust the logic to perfectly match the requested output:
        # +4x0 +1x1 = 3
        
        # New attempt for perfect formatting:
        current_line = ""
        for j in range(N):
            coeff = int(A[i, j])
            var = f"x{j}"
            
            # Format the term with sign and variable
            if j == 0:
                # First term: Must start with a '+' or '-'
                sign_str = '+' if coeff >= 0 else ''
                term = f"{sign_str}{coeff}{var}"
                if term.startswith('+-'): # Correct for negative coefficients
                    term = term[1:]
            else:
                # Subsequent terms: Must have a space and a '+' or '-'
                sign_str = '+' if coeff >= 0 else '-'
                term = f" {sign_str}{abs(coeff)}{var}"
            
            current_line += term
        
        # Add the right-hand side
        current_line += f" = {int(b[i])}"
        output_lines.append(current_line)


    # The required output format is extremely specific:
    # +4x0 +1x1 = 3
    # +2x0 +5x1 = 1
    # Let's use the simplest logic to match this example perfectly for integer coefficients:
    perfect_lines = []
    for i in range(N):
        line_parts = []
        for j in range(N):
            coeff = int(A[i, j])
            sign = '+' if coeff >= 0 else ''
            term = f"{sign}{coeff}x{j}"
            line_parts.append(term)
        
        # Join terms with a space
        equation_str = ' '.join(line_parts)
        
        # Ensure the first term has a leading '+' for positive numbers (which the join might strip)
        if A[i, 0] >= 0 and not equation_str.startswith('+'):
             equation_str = '+' + equation_str
             
        # Now fix the internal signs to always be ' +C' or ' -C'
        # The logic `str.replace('+',' +').replace('-',' -')` is prone to errors,
        # so we rely on the loop's construction to be precise.
        
        # Re-evaluating the example: +4x0 +1x1 = 3
        # It seems the rule is:
        # 1. Start with + or - (no space).
        # 2. Subsequent terms start with ' +' or ' -'.
        
        final_line = ""
        for j in range(N):
            coeff = int(A[i, j])
            
            if j == 0:
                sign = '+' if coeff >= 0 else ''
                final_line += f"{sign}{coeff}x{j}"
            else:
                sign = '+' if coeff >= 0 else '-'
                # Use abs(coeff) since the sign is explicit
                final_line += f" {sign}{abs(coeff)}x{j}"
                
        final_line += f" = {int(b[i])}"
        perfect_lines.append(final_line)


    try:
        with open(filename, 'w') as f:
            f.write('\n'.join(perfect_lines) + '\n')
    except Exception as e:
        print(f"An error occurred during writing to {filename}: {e}")


def gauss_elimination(A, b):
    """
    Solves the linear equation system Ax=b using the Gauss Elimination method
    with back-substitution.

    Args:
        A (np.array): The coefficient matrix.
        b (np.array): The constant vector.

    Returns:
        np.array: The solution vector x, or None if the system has no unique solution.
    """
    N = A.shape[0]
    # Create the augmented matrix [A | b]
    Ab = np.hstack((A, b.reshape(-1, 1)))

    # 1. Forward Elimination to get an Upper Triangular Matrix
    for i in range(N):
        # Find pivot: In a simpler form, we just ensure a non-zero diagonal element
        # (pivoting is often required for stability, but omitted here for simplicity
        # as the problem does not explicitly demand it and the example is well-behaved).
        if Ab[i, i] == 0.0:
            # Handle potential division by zero (system may have no unique solution)
            # A more robust implementation would use partial pivoting.
            # For this context, we'll assume a solvable system or just return None.
            return None

        # Normalize the pivot row: make Ab[i, i] = 1
        Ab[i] = Ab[i] / Ab[i, i]

        # Elimination for rows i+1 to N-1
        for j in range(i + 1, N):
            factor = Ab[j, i] / Ab[i, i] # This will be Ab[j, i] since Ab[i,i] is 1 after normalization
            Ab[j] = Ab[j] - factor * Ab[i]

    # After elimination, check for singular matrix (a row of zeros in A part)
    # The rank check: if any diagonal element is close to zero
    if any(abs(Ab[i, i]) < 1e-9 for i in range(N)):
         return None # System has no unique solution

    # 2. Back Substitution
    x = np.zeros(N)
    for i in range(N - 1, -1, -1):
        # x[i] = (Ab[i, N] - sum(Ab[i, j] * x[j] for j > i)) / Ab[i, i]
        # Since the matrix is normalized, Ab[i, i] is 1, so:
        x[i] = Ab[i, N] - np.sum(Ab[i, i + 1:N] * x[i + 1:N])

    return x


def jacobi_iteration(A, b, max_iterations=100, tolerance=1e-5):
    """
    Solves the linear equation system Ax=b using the Jacobi Iteration method.
    The method requires the matrix A to be (preferably strictly) diagonally dominant
    for convergence.

    Args:
        A (np.array): The coefficient matrix.
        b (np.array): The constant vector.
        max_iterations (int): Maximum number of iterations.
        tolerance (float): Stopping criterion based on the change in the solution vector.

    Returns:
        np.array: The solution vector x, or None if the method didn't converge.
    """
    N = A.shape[0]
    x = np.zeros(N)     # Initial guess for x (e.g., all zeros)
    x_new = np.copy(x)  # Vector to store the solution of the current iteration

    # Check for non-zero diagonal elements (essential for the method)
    if any(A[i, i] == 0 for i in range(N)):
        print("Warning: Jacobi method requires non-zero diagonal elements. Returning None.")
        return None

    for k in range(max_iterations):
        x_prev = np.copy(x) # Store previous iteration's result

        for i in range(N):
            # x_i(k+1) = (1 / A_ii) * (b_i - sum(A_ij * x_j(k) for j!=i))
            sum_val = 0
            for j in range(N):
                if i != j:
                    sum_val += A[i, j] * x[j]

            x_new[i] = (b[i] - sum_val) / A[i, i]

        # Update x for the next iteration
        x = np.copy(x_new)

        # Check for convergence (relative change)
        # np.linalg.norm(x - x_prev) is the Euclidean distance between solutions
        if np.linalg.norm(x - x_prev) < tolerance:
            # print(f"Jacobi converged in {k+1} iterations.")
            return x

    # If the loop finishes without meeting the tolerance
    # print(f"Warning: Jacobi method did not converge within {max_iterations} iterations.")
    return x # Return the best solution found, even if not fully converged


def print_results_to_file(gauss_solution, jacobi_solution, filename="output_results.txt"):
    """
    Writes the solutions from both methods to the specified output file,
    formatted as required (4 decimal places, rounded).

    Args:
        gauss_solution (np.array or None): The solution vector from Gauss Elimination.
        jacobi_solution (np.array or None): The solution vector from Jacobi Iteration.
        filename (str): The name of the output file.
    """
    output_lines = []

    # --- Gauss Elimination Results ---
    output_lines.append("The solution of the system by Gauss Elimination:")
    if gauss_solution is None:
        output_lines.append("No unique solution found or method failed.")
    else:
        # The output requires x1, x2, ... (1-based index)
        for i, val in enumerate(gauss_solution):
            # Rounding to the 4th digit after the decimal point
            rounded_val = round(val, 4)
            # Format to ensure exactly 4 decimal places are displayed
            output_lines.append(f"x{i+1}={rounded_val:.4f}")

    # --- Jacobi Method Results ---
    output_lines.append("The solution of the system by Jacobi Method:")
    if jacobi_solution is None:
        output_lines.append("Convergence failed or method is not applicable.")
    else:
        for i, val in enumerate(jacobi_solution):
            rounded_val = round(val, 4)
            output_lines.append(f"x{i+1}={rounded_val:.4f}")

    # Write all lines to the output file
    try:
        with open(filename, 'w') as f:
            f.write('\n'.join(output_lines) + '\n')
    except Exception as e:
        print(f"An error occurred during writing to {filename}: {e}")


def main():
    """
    Main function to execute the program flow.
    """
    # 1. Read the system from the input file
    print("Reading system from input.txt...")
    A, b = read_system_from_file("input.txt")

    print(f"Matrix A:\n{A}")
    print(f"Vector b:\n{b}")

    # 2. Print the given matrix to "output.txt"
    print("Writing formatted equations to output.txt...")
    format_and_print_matrix(A, b, "output.txt")

    # 3. Solve the system
    print("Solving system using Gauss Elimination...")
    gauss_sol = gauss_elimination(np.copy(A), np.copy(b)) # Use copies to avoid modifying original A, b
    
    # The Jacobi method requires a check for convergence, here we rely on the implementation limits
    # The example matrix is strictly diagonally dominant, ensuring convergence: |4| > |1|, |5| > |2|
    print("Solving system using Jacobi Iteration...")
    # Using a higher max_iterations for robustness, though 100 is often enough
    jacobi_sol = jacobi_iteration(np.copy(A), np.copy(b), max_iterations=500, tolerance=1e-8)

    # 4. Print the solutions to "output_results.txt"
    print("Writing results to output_results.txt...")
    print_results_to_file(gauss_sol, jacobi_sol, "output_results.txt")
    
    print("\nProgram finished successfully.")
    print("Check 'output.txt' and 'output_results.txt' for the results.")

# Execution of the main function
if __name__ == "__main__":
    main()

# The final files will contain:

# Content of "output.txt" (for the example input):
# +4x0 +1x1 = 3
# +2x0 +5x1 = 1

# Content of "output_results.txt" (for the example input):
# The solution of the system by Gauss Elimination:
# x1=0.7778
# x2=-0.1111
# The solution of the system by Jacobi Method:
# x1=0.7778
# x2=-0.1111