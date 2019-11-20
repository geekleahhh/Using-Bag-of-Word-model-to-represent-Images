# Using-Bag-of-Word-model-to-represent-Images
The problem is that such systems scale linearly, meaning that as more images are added to our system, the longer it will take to perform a search.

For example, if you included a few thousand images in your dataset, search time would slow dramatically because we need to compare our query image to every image in our dataset.

So, what’s the solution?

In practice, to build image search engines that scale to millions of images, we apply the bag of visual words (BOVW) method.

# Using Deep Networks to represent Images
We can now compare the above approaches to an approach where we represent the images with Deep Representation.

We are going to implement a simpler version of the R-MAC method proposed in the paper: Tolias, G., Sicre, R., & Jégou, H. (2015). Particular object retrieval with integral max-pooling of CNN activations. arXiv preprint arXiv:1511.05879.
