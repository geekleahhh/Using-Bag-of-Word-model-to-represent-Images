Put image into data(only one)
Then run "rmac3.py"
and you can get the representation for this image.


1. Use keras.applications to build VGG net.(import first)
2. Use the VGG(removed last 4 layers) to get image representation from VGG.
3. Use RoiPooling to the image representation to get a RMAC vector to represent this image.
4. Normalize the RMAC vector
5. From sklearn to use pca
6. Use pca to RMAC and thus to reduce the dimensions
