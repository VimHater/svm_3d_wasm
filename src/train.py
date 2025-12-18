import numpy as np
from sklearn.linear_model import SGDClassifier

np.random.seed(42)

def create_data(N: int):
    """
    Creates a 3D linearly separable dataset.
    
    Returns:
        tuple: (X, y) where X is the N x 3 feature matrix 
               and y is the N x 1 class vector (-1 or 1).
    """
    X0 = np.random.rand(N // 2, 3) * 5
    y0 = np.zeros(N // 2, dtype=int)

    offset = np.array([5, 5, 5])
    X1 = np.random.rand(N // 2, 3) * 5 + offset
    y1 = np.ones(N // 2, dtype=int)

    X = np.vstack([X0, X1])
    y = np.hstack([y0, y1])
    
    y[y == 0] = -1 
    
    return X, y

def print_dataset_cpp_header(X: np.ndarray, y: np.ndarray):
    """
    // Prints the dataset points formatted as a C++ header file structure.
    """
    print("// --- Generated Dataset Points (C++ Header Format) ---")
    print("\n#include <vector>")
    print("\ntypedef struct {")
    print("    float x, y, z;")
    print("    int Catag;")
    print("} Point;")
    
    print(f"\n// Total points: {X.shape[0]}")
    print("std::vector<Point> Points = {")
    
    data_points = np.hstack((X, y.reshape(-1, 1)))
    
    for i, row in enumerate(data_points):
        # Add 'f' suffix for C++ float literals
        line = (
            f"  {{{row[0]: .8f}f, {row[1]: .8f}f, {row[2]: .8f}f, {int(row[3])}}}"
        )
        if i < len(data_points) - 1:
            line += ","
        
        print(line)

    print("};")
    print("// --- End of Dataset ---")


def train_iterative_svm(X: np.ndarray, y: np.ndarray, n_epochs: int = 100):
    """
    Trains a linear SVM using SGD with Hinge Loss for a specified number of epochs,
    stores the hyperplane coefficients, and prints them in C++ vector format at the end.
    """
    print("/*")
    print("\n" + "=" * 50)
    print("--- Iterative SVM Training (SGD with Hinge Loss) ---")
    print(f"Training for {n_epochs} epochs. Collecting raw hyperplane functions...")
    print("Hyperplane Equation Format: w1*x + w2*y + w3*z + b = 0")
    print("=" * 50)
    print("*/")

    plane_coefficients = []

    clf = SGDClassifier(
        loss='hinge', 
        penalty='l2', 
        alpha=0.0001, 
        max_iter=1, 
        fit_intercept=True, 
        random_state=42, 
        shuffle=True,
        tol=None,
        learning_rate='constant',
        eta0=0.01 
    )
    
    print("/*")
    for epoch in range(1, n_epochs + 1):
        clf.partial_fit(X, y, classes=np.unique(y))

        w = clf.coef_[0]
        
        # FIX: Use .item() to safely extract the single float value from the intercept array, 
        # which avoids the 'IndexError: invalid index to scalar variable' error.
        b = clf.intercept_.item() 
        
        # Store as a tuple (w1, w2, w3, b)
        plane_coefficients.append((w[0], w[1], w[2], b)) # Note: 'b' is now a scalar, so no 'b[0]' is needed
        
        # Optionally print the epoch number to show progress (not the full equation)
        if epoch % 10 == 0 or epoch == 1:
            print(f"//  Processed Epoch {epoch:03d}...")


    print("*/")
    print("\n// Note: This block assumes the 'Point' struct and 'std::vector' are already defined.")
    print("\ntypedef struct {")
    print("    float w1, w2, w3, b;")
    print("} Equation;")
    
    print(f"\n// Total epochs (equations): {len(plane_coefficients)}")
    print("std::vector<Equation> Planes = {")
    
    for i, (w1, w2, w3, b_val) in enumerate(plane_coefficients):
        line = (
            f"  {{{w1: .8f}f, {w2: .8f}f, {w3: .8f}f, {b_val: .8f}f}}"
        )
        
        if i < len(plane_coefficients) - 1:
            line += ","
        
        print(line)

    print("};")
    
if __name__ == "__main__":
    N_POINTS = 400
    
    X_data, y_data = create_data(N=N_POINTS)
    
    print_dataset_cpp_header(X_data, y_data)

    train_iterative_svm(X_data, y_data, n_epochs=400)
