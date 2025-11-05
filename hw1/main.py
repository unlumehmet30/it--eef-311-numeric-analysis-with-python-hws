import numpy as np

def read_input(filename="input.txt"):
    """
    Reads matrix A and vector b from the input file.
    The input.txt format:
    Size=N
    a11 a12 ...
    a21 a22 ...
    ...
    b1 b2 ...
    """
    try:
        with open(filename) as f:
            lines = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"{filename} not found.")
        return None, None

    try:
        N = int(lines[0].split('=')[1])
        A = np.array([list(map(float, lines[i + 1].split())) for i in range(N)])
        b = np.array(list(map(float, lines[N + 1].split())))
        return A, b
    except Exception as e:
        print("Error in input file:", e)
        return None, None


def write_equations(A, b, filename="output.txt"):
    """
    Writes the system Ax=b into a text file in equation format.
    Format required by the assignment: +4x0 +1x1 = 3
    """
    with open(filename, 'w') as f:
        for i in range(A.shape[0]):
            eq_terms = []
            for j in range(A.shape[1]):
                coef = int(A[i, j])
                if coef >= 0:
                    eq_terms.append(f"+{coef}x{j}")
                else:
                    eq_terms.append(f"-{abs(coef)}x{j}")
            eq = " ".join(eq_terms)
            f.write(f"{eq} = {int(b[i])}\n")


def gauss_elimination(A, b):
    """
    Solves the linear system Ax=b using Gauss Elimination method.
    """
    A, b = A.copy(), b.copy()
    n = len(b)

    # Forward elimination
    for i in range(n):
        pivot = A[i, i]
        if abs(pivot) < 1e-10:
            print("Pivot is too small, system cannot be solved.")
            return None
        for j in range(i + 1, n):
            factor = A[j, i] / pivot
            A[j, i:] -= factor * A[i, i:]
            b[j] -= factor * b[i]

    # Back substitution
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - np.dot(A[i, i + 1:], x[i + 1:])) / A[i, i]

    return x


def jacobi(A, b, max_iter=200, tol=1e-8):
    """
    Solves the linear system Ax=b using the Jacobi Iterative method.
    """
    n = len(b)
    x = np.zeros(n)

    for _ in range(max_iter):
        x_new = np.zeros_like(x)
        for i in range(n):
            s = np.dot(A[i, :i], x[:i]) + np.dot(A[i, i + 1:], x[i + 1:])
            if A[i, i] == 0:
                print("Zero diagonal element found, Jacobi cannot be applied.")
                return None
            x_new[i] = (b[i] - s) / A[i, i]

        if np.linalg.norm(x_new - x) < tol:
            return x_new
        x = x_new

    print("Jacobi method did not converge.")
    return None


def write_results(gauss_sol, jacobi_sol, filename="output_results.txt"):
    """
    Writes the solutions of the system into a text file.
    Format strictly follows the assignment requirements.
    """
    with open(filename, 'w') as f:
        f.write("The solution of the system by Gauss Elimination:\n")
        if gauss_sol is None:
            f.write("Solution could not be found.\n")
        else:
            for i, val in enumerate(gauss_sol, 1):
                f.write(f"x{i}={val:.4f}\n")

        f.write("\nThe solution of the system by Jacobi Method:\n")
        if jacobi_sol is None:
            f.write("Method did not converge.\n")
        else:
            for i, val in enumerate(jacobi_sol, 1):
                f.write(f"x{i}={val:.4f}\n")


def main():
    # Read data from input file
    A, b = read_input("input.txt")
    if A is None or b is None:
        return

    # Write the equations into output.txt
    write_equations(A, b)

    # Compute solutions using Gauss and Jacobi methods
    g_sol = gauss_elimination(A, b)
    j_sol = jacobi(A, b)

    # Write results into output_results.txt
    write_results(g_sol, j_sol)
    print("Equations written to output.txt and solutions written to output_results.txt.")


if __name__ == "__main__":
    main()
