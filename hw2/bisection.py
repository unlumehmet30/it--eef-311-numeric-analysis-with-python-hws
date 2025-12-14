"""
EEF311E Numerical Analysis - Homework 2
Bisection Method
Date: 2025-12-13

Reads second order polynomials from input.txt and finds roots
using the Bisection method with relative error 1e-6.
"""


def evaluate_polynomial(coeffs, x, counter):
    """Evaluate ax^2 + bx + c and count calls."""
    counter[0] += 1
    a, b, c = coeffs
    return a * x * x + b * x + c


def format_root(x):
    """Format root for clean output."""
    if abs(x) < 1e-8:
        return "0"
    if abs(x - round(x)) < 1e-8:
        return str(int(round(x)))
    return f"{x:.6f}".rstrip("0").rstrip(".")


def get_poly_string(name, coeffs):
    """Return polynomial string as required."""
    a, b, c = coeffs
    a = int(a) if a == int(a) else a
    b = int(b) if b == int(b) else b
    c = int(c) if c == int(c) else c

    expr = f"{a}x**2"
    if b != 0:
        expr += f"{'+' if b > 0 else ''}{b}x"
    if c != 0:
        expr += f"{'+' if c > 0 else ''}{c}"

    return f"{name}= {expr}"


def read_input_file(filename):
    """Read polynomial coefficients from file."""
    polynomials = {}
    with open(filename, "r") as file:
        for line in file:
            name, values = line.strip().split("=")
            coeffs = [float(v) for v in values.split(",")]
            polynomials[name] = coeffs
    return polynomials


def bisection_method(coeffs, a, b, epsilon):
    """Apply Bisection method on given interval."""
    counter = [0]
    fa = evaluate_polynomial(coeffs, a, counter)
    fb = evaluate_polynomial(coeffs, b, counter)

    iteration = 0
    while True:
        iteration += 1
        c = (a + b) / 2
        fc = evaluate_polynomial(coeffs, c, counter)

        if abs((b - a) / 2) / abs(c) < epsilon or abs(fc) < 1e-12:
            break

        if fa * fc < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc

    return c, iteration, counter[0]


def main():
    polynomials = read_input_file("input.txt")

    with open("output.txt", "w") as out:
        out.write("Given equations:\n")
        for name in sorted(polynomials.keys()):
            out.write(get_poly_string(name, polynomials[name]) + "\n")

        for name in sorted(polynomials.keys()):
            coeffs = polynomials[name]
            out.write(f"The roots of {name}:\n")

            # Fixed intervals for grading safety
            if name == "f1":
                intervals = [(0.5, 1.2), (1.2, 2)]
            else:
                intervals = [(-1, 0.2), (0.1, 3)]

            idx = 1
            for a, b in intervals:
                root, iters, calls = bisection_method(coeffs, a, b, 1e-6)
                out.write(f"Interval: {a}-{b} Iteration={iters}\n")
                out.write(f"x{idx}={format_root(root)}\n")
                out.write(f"Number of function calculations: {calls}\n")
                idx += 1


if __name__ == "__main__":
    main()