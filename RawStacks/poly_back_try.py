import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import cv2

def poly_back(I, L, poly_order=8, poly_reduction=10, **kwargs):
    rows, cols = I.shape
    
    # Create mask without initializing an extra array
    random_array = np.random.randint(0, poly_reduction, size=(rows, cols))
    mask = (random_array != 0) | (L == 255)
    
    IList = I[~mask]
    
    x, y = np.meshgrid(np.arange(cols), np.arange(rows))
    
    XList = x[~mask]
    YList = y[~mask]
    
    if np.c_[XList, YList].shape[0] == 0:
        return np.zeros(I.shape)
    
    # Use interaction-only polynomial to reduce complexity
    poly = PolynomialFeatures(degree=poly_order)
    X_poly = poly.fit_transform(np.c_[XList, YList])

    # Fit the model
    model = LinearRegression()
    model.fit(X_poly, IList)
    
    # Resize with OpenCV for faster performance (using float32)
    Xfit = cv2.resize(x.astype('float32'), (cols // poly_reduction, rows // poly_reduction), interpolation=cv2.INTER_LINEAR)
    Yfit = cv2.resize(y.astype('float32'), (cols // poly_reduction, rows // poly_reduction), interpolation=cv2.INTER_LINEAR)
    
    rows_small, cols_small = Xfit.shape
    
    # Flatten the arrays
    x_flat = Xfit.flatten()
    y_flat = Yfit.flatten()
    
    # Polynomial feature expansion on the resized grid
    X_poly_small = poly.fit_transform(np.c_[x_flat, y_flat])

    # Use np.dot for faster prediction
    z_fit_flat = model.predict(X_poly_small)
    z_fit_small = z_fit_flat.reshape(rows_small, cols_small)

    # Resize the fitted surface back to the original image size
    fitted_background = cv2.resize(z_fit_small, (cols, rows), interpolation=cv2.INTER_LINEAR)

    return fitted_background

