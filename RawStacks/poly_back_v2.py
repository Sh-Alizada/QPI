import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from skimage.transform import resize

def poly_back(I, L, poly_order=8, poly_reduction=5, **kwargs):
    """
    Fit a polynomial surface to an image and resize it back to the original size.
    
    Parameters:
    - I (numpy array): The input phase image.
    - L (numpy array): The input binary image where cell pixels are 1.
    - poly_order (int): The order of the polynomial to fit.
    - reduction_factor (int): The factor by which to reduce the fitting points.
    
    Returns:
    - fitted_background (numpy array): The fitted polynomial background resized back to the original image size.
    """
    
    rows, cols = I.shape
    
    random_array = np.random.randint(0, poly_reduction, size=(rows, cols))
    mask = np.zeros(I.shape)
    mask[(random_array != 0) | (L == 255)] = 1
    
    IList=I[mask==0]
    
    
    x, y = np.meshgrid(np.arange(cols), np.arange(rows))
    
    XList=x[mask==0]
    YList=y[mask==0]
    
    if np.c_[XList, YList].shape[0] == 0:
        return np.zeros(I.shape)
    
    poly = PolynomialFeatures(degree=poly_order)
    X_poly = poly.fit_transform(np.c_[XList, YList])

    # Fit the model
    model = LinearRegression()
    model.fit(X_poly, IList)
    
    
    Xfit = resize(x.astype('float64'), (rows // poly_reduction, cols // poly_reduction))
    Yfit = resize(y.astype('float64'), (rows // poly_reduction, cols // poly_reduction))
    
    rows_small, cols_small = Xfit.shape
    
    # Flatten the arrays
    x_flat = Xfit.flatten()
    y_flat = Yfit.flatten()
    
    X_poly_small = poly.fit_transform(np.c_[x_flat, y_flat])

    
    # Predict the surface on the reduced image
    z_fit_flat = model.predict(X_poly_small)
    z_fit_small = z_fit_flat.reshape(rows_small, cols_small)

    # Resize the fitted surface back to the original image size
    fitted_background = resize(z_fit_small, (rows, cols))


    return fitted_background

