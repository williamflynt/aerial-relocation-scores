# NOTES

## 21 April 2024

Got an overhead imagery dataset: https://project.inria.fr/aerialimagelabeling/files/

It's about 20GB compressed in 7z multipart format

Idea is to predict location or something, based on single overhead snapshot. Maybe join it with a public dataset on some other metric?

Made some scripts to download and extract data.

## 22 April 2024

I have image data, and I want to know if "similar" images have a similar metric.

We could probably increase our dataset size by splitting up images to many smaller images convolution-style.

Then we can extract some features somehow... Color seems to be the big differentiator in the images. What about angles, or straight line length?

We can use our [similarity measures](https://www.coursera.org/learn/unsupervised-algorithms-in-machine-learning/lecture/8n99y/similarity-measures) to see which algorithm gives similar scores for images from the same parent/group, then see how well we can learn from images about our metric.

We could use clustering - and that lends itself well to PCA pre-analysis visualization.

### The Point

Really, if we can get a good set of features to group within parent + location, we've already shown we can also predict some metric, if it's differentiated across overhead imagery dimensions. So we should focus on that...

Since this is unsupervised learning, we may want to NOT feature engineer... but that defeats my ulterior motive of knowing certain human features I care about for choosing a place to live (ex: red clay roofs, public squares, narrow roads compared to home size, ...)

What I ACTUALLY want, as a person, is to upload a picture of a place and to find other places like it in terms of architecture, lifestyle, traffic...
    - How can I make this happen? Perhaps an application can do some blocking on ex: weather, get tiles for 1_000 places, and then run the model?

### Images

So I split these up to 512x512 - it's all too big to hold in memory, so I'll down sample the source and hope for the best.

I'm also going to get greens, reds, and road-color because that's what I care about.

### More complicated

This isn't working as well as I thought.
I am going to go after my own personal goals of finding Mediterranean towns all over the world
Let's use... magical computer vision. To get features. To train a less sophisticated model.

## 23 April 2024

KNN training is very slow on 1024 dimensions. Let's extract the dimensions we might care about with OpenCV.

I started listing specific colors I care about, but what if we cut the color space into variable segments and just used that.

I'm treating HSL like a cylinder, and slicing it radially and by luminance or saturation.
Saturation appears to have a much better accuracy...

So now I can pretty well cluster to the group - that means I have some kind of meaningful representation.
But what about predicting a "would I live there" score?

### Predicting Would-I-Live-There

1. Rank each city
2. Write into metadata
3. Train a new KNN regression on the vectorized images
4. Test with places I want and don't want to live
5. Make presentation
