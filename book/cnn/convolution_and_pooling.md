# Convolution and Pooling

A **filter** can extract a feature from a signal by combining nearby values. In a time series, for example, comparing each measurement with its neighbours can highlight local peaks. Averaging nearby measurements smooths short-term fluctuations and makes slower changes easier to see.

Filters also work on images. In a satellite image, for example, a filter that responds to a sharp change in brightness can help locate the boundary between water and land. A different filter might respond to the repeating pattern of crop rows.

A **convolutional neural network** (CNN) learns its filters from examples during training. Each filter computes a weighted sum of nearby values. The network can apply a nonlinear activation to the results and pass them to another layer of filters. In this way, an early layer might pick out short edges in a satellite image, while later layers combine those edges into corners and eventually the outline of a building. With many filters and enough training data, CNNs can handle complex applications such as detecting signs of drought in satellite images or identifying cracks in bridges from camera images. The same approach can be adapted to simulate how floods spread and to predict the movement of storms from radar observations. To see how a single filter works, we’ll start with one whose weights we choose ourselves.

**Resources**

- [Slides](https://surfdrive.surf.nl/files/index.php/s/i6pLtXtmlc7rV6R/)
- Simon J. D. Prince, *Understanding Deep Learning* (MIT Press, 2023), chapter 10, especially sections 10.2.1–10.2.6, 10.3 and 10.4.1–10.4.2. [Book website](https://udlbook.github.io/udlbook/)

## Convolutions

A common way to apply a filter, also called a *kernel*, to a signal is through **cross-correlation**. The kernel slides along the signal, one window at a time. At each position, the signal values are multiplied by the corresponding kernel weights and summed. The resulting value is placed in the output signal at the position corresponding to the centre of the window.

Mathematical **convolution** is cross-correlation with the kernel weights reversed. For example, the weights $(1,2,3)$ are applied in the order $(3,2,1)$. Deep learning frameworks usually compute cross-correlations and learn the weights in that order, but call the operation *convolution*. We use that convention here. [PyTorch convolution convention](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)

### 1D Convolutions

#### Detecting an isolated peak

Imagine we record the water level in a creek every six hours. Our six readings are $(40,40,60,40,40,40)$ cm. We can see a peak at the third reading because the water rose by 20 cm and then fell by 20 cm. We would like a filter that picks out a similar pattern in a longer record.

To measure how sharply a reading stands out, we compare it with each of its neighbours. If $x_i$ is the centre reading, the rise from the previous reading is $x_i-x_{i-1}$ and the drop to the next is $x_i-x_{i+1}$. At a peak both differences are positive, while on a flat stretch both are zero. Their sum gives us a simple measure of how pronounced a peak is. We’ll call it a *peak score* for this example.

In this sum, the centre reading counts twice and we subtract each neighbour once. Reading from left to right, the three weights are therefore $(\omega_1,\omega_2,\omega_3)=(-1,2,-1)$. In kernel notation, the score $z_i$ at reading $i$ is

```{math}
:label: conv_3
z_i = \omega_1 x_{i-1} + \omega_2 x_i + \omega_3 x_{i+1}.
```

In {numref}`conv_1`, the window is centred on reading 2. The creek is still at 40 cm and the next reading is higher, so the filter gives a negative score.

```{figure} ../figures/cnn_first_conv.svg
:name: conv_1
:alt: The window centred on reading 2 covers levels 40, 40 and 60 cm. Its score of minus 20 aligns with reading 2 on the output plot.

At reading 2, $z_2=-40+2(40)-60=-20$. Readings 1 and 6 have no score yet because each is missing a neighbour.
```

Move the window forward by one reading ({numref}`conv_2`). Now the centre is 20 cm above each neighbour, so the score is $20+20=40$. Across readings 2 through 5, the scores are $(-20,40,-20,0)$. The largest score locates the isolated peak at reading 3.

```{figure} ../figures/cnn_second_conv.svg
:name: conv_2
:alt: The window shifts to levels 40, 60 and 40 cm. The peak in the output aligns with reading 3 in the input.

At reading 3, the same weights give $z_3=-40+2(60)-40=40$.
```

#### A moving average filter

To smooth a signal, we can use the weights $(1/3,1/3,1/3)$ so that each reading in a window contributes equally. Sliding this filter along the record gives a sequence of local means called a **moving average**. In the window centred on reading 3, for example, the water levels are $(40,60,40)$ cm, so the mean is $(40+60+40)/3=46\tfrac{2}{3}$ cm. The high reading raises the average in each of the three windows that contain it.

#### Zero Padding and Valid Convolutions

So far we have applied both filters using **valid convolution**, which means we calculate an output only where all three readings in the window are available. For the peak detector, the first complete window is centred on reading 2 and the last on reading 5 ({numref}`valid_convolution`). We therefore obtain four scores from six readings. At either end of the record, one neighbour is missing.

```{figure} ../figures/cnn_creek_peak_valid.svg
:name: valid_convolution
:alt: Four windows centred on readings 2 through 5 give peak scores of minus 20, 40, minus 20 and 0.

The four scores $(-20,40,-20,0)$ form the output **feature map**. Each score comes from three measured water levels.
```

To obtain a score at every input position, we can extend the record with extra values at its edges. This is called **padding**. A common choice is **zero padding**. Here, adding one zero at each end allows the three-reading window to reach all six positions. In {numref}`zero_padding`, the first window is $(0,40,40)$, giving $2(40)-0-40=40$. The last window is $(40,40,0)$, giving $2(40)-40-0=40$.

```{figure} ../figures/cnn_creek_peak_zero_padding.svg
:name: zero_padding
:alt: Six windows give peak scores of 40, minus 20, 40, minus 20, 0 and 40. Only the first and last windows include a padding zero.

The padded feature map is $(40,-20,40,-20,0,40)$. Its four interior scores are the same as in valid convolution.
```

Zero padding preserves the length of the signal here, but both boundary scores are as large as the score at the observed peak. The filter sees a rise from 0 to 40 cm at the start and a drop from 40 to 0 cm at the end. Both changes come from the padding. In many applications this has little effect, particularly when the inputs are normalized around zero and the boundary readings are close to that value. When artificial boundary responses matter, valid convolution avoids them by using only complete windows.

### Kernel size, stride and dilation

Kernel size, stride and dilation are the three settings that determine which samples contribute to each output and how often the filter produces an output. All the examples so far have used a kernel size of 3, a stride of 1 and a dilation of 1. {numref}`dilation` illustrates all three on a signal with nine samples.

- The **kernel size** is the number of samples the filter combines in one window. With adjacent samples, increasing the size from three to five gives the filter more context to recognize a pattern that extends over several positions. It also requires two more weights and two more multiplications per output.
- The **stride** is the number of positions the window moves between outputs. With stride 2, a window centred on position 2 moves next to position 4. This *downsamples* the signal. A long input produces roughly half as many outputs as it would with stride 1, so later layers have fewer values to process. Strided convolution is one alternative to pooling, which we introduce later in the chapter.
- The **dilation** is the spacing between successive samples within a window. A three-weight filter centred on position 5 uses samples 4, 5 and 6 at dilation 1, or samples 3, 5 and 7 at dilation 2. Increasing dilation lets the filter detect relationships between more distant samples with the same number of weights. We can still compute an output at every position where the window fits.

```{figure} ../figures/cnn_dilation.svg
:name: dilation
:alt: A window centred on position 5 uses three or five inputs as its size changes. Increasing stride moves successive centres farther apart. Increasing dilation spreads three inputs farther apart within one window.

In the top and bottom panels, the window stays centred on position 5. The filled circles show which readings contribute to that score. In the middle panel, the diamonds show where successive scores are calculated as the window moves.
```

#### Try a filter on a weather record

The graph below shows hourly precipitation at KNMI’s De Bilt station in the Netherlands during a rainy week, 20–26 May 2024 (UTC). Start with the moving average. Each point on the filtered curve shows the mean of three consecutive readings, plotted at the time of the middle reading. Choose **Peak detector** and select **Apply filter**. Which showers now stand out? Change the stride to 4, apply the filter again and check whether the highest-scoring hours still appear on the filtered curve.

You can also choose another station measurement. Both curves update immediately using the current filter. If you edit the weights or settings, select **Apply filter** to see the result. The curves share a time axis, and you can adjust their vertical scales separately.

<iframe src="../_static/cnn-filter-explorer.html" style="width: 100%; height: 850px; border: 0;" loading="lazy" title="Apply a 1D filter to hourly weather station measurements"></iframe>

(creating-an-1d-convolutional-layer)=
### Creating a 1D Convolutional Layer

We chose the peak detector’s weights because we knew what pattern we wanted to find. In a convolutional layer, training adjusts the weights so that the filter responds to patterns that help with the prediction task. The filter uses the same weights at every position, so it can respond to a pattern wherever it occurs.

At each position, the layer adds a learned **bias** $\beta$ to the weighted sum and applies a nonlinear activation function $a$. For a window of three consecutive samples, the resulting output $h_i$ is

```{math}
:label: conv-1d
\begin{aligned}
h_i &= a\!\left(\beta + \omega_1 x_{i-1} + \omega_2 x_i + \omega_3 x_{i+1}\right)\\
    &= a\!\left(\beta + \sum_{j=1}^{3} \omega_j x_{i+j-2}\right).
\end{aligned}
```

```{note}
A **fully connected layer** also forms a weighted sum, but each output can use all the inputs and has its own weights. For $D$ inputs, the output at unit $i$ is

$$
h_i = a\!\left(\beta_i + \sum_{j=1}^{D} \omega_{ij} x_j\right).
$$

With $D$ output units as well, the fully connected layer needs $D^2$ weights and $D$ biases. A convolutional filter of width $K$ applied to one input sequence needs only $K$ weights and, if used, one bias. To see how the two layers are related, imagine giving each fully connected unit access only to its local window by setting its other weights to zero. If every unit now uses the same weights in the same order within its window and shares the same bias, it performs the convolutional layer’s calculation.
```

### Receptive Field

The first layer of a CNN might use just three readings to detect a peak. A second layer can use several of those peak scores to recognize a pattern that lasts longer. To see how much of the original record a particular output can depend on, we can trace its inputs back through the layers. The set of original input positions that can influence that output is called its **receptive field**.

Follow the output at position 5 in {numref}`receptive_field_1`. In the first layer, it depends on readings 4, 5 and 6. The second layer uses the same window size, but its three inputs are now the first layer’s outputs at positions 4, 5 and 6. The leftmost of those outputs already uses reading 3, and the rightmost uses reading 7. Together, their windows cover readings 3 through 7. A third layer extends the range again, to readings 2 through 8.

```{figure} ../figures/receptive_field_1.svg
:name: receptive_field_1
:alt: Intervals centred on position 5 cover 3, 5 and 7 original input elements at successive depths.

All three layers have kernel size 3, stride 1 and dilation 1. Each layer extends the receptive field by one original position on either side.
```

Stride also affects how quickly the receptive field grows. Suppose the first layer uses stride 2, so that three successive outputs are centred on readings 2, 4 and 6. When the second layer combines those outputs, its receptive field extends from reading 1 through reading 7. It now covers seven readings after just two layers.

Dilation also widens the receptive field. If we keep stride 1 and use dilation 2 in the second layer, its filter at position 5 combines first-layer outputs at positions 3, 5 and 7. Together, their windows cover readings 2 through 8, using the same three filter weights.

### Channels

#### Multiple filters produce multiple features

For a flood warning model, we might want to know both how sharply a water level peaks and how high the water has been on average. We could supply those two features by applying the peak detector and the moving average to the same water-level record.

Each filter produces its own **feature map**, also called an **output channel**. In {numref}`cnn_channels`, both filters use the same windows, so their outputs line up. At reading 3, for example, the next layer receives both a peak score and the average water level around that peak.

```{figure} ../figures/cnn_creek_output_channels.svg
:name: cnn_channels
:alt: For creek readings 2 through 5, the peak scores are minus 20, 40, minus 20 and 0. The moving averages are 46⅔, 46⅔, 46⅔ and 40 cm.

At reading 3 both filters see $(40,60,40)$ cm. The peak score is $-40+2(60)-40=40$ and the moving average is $(40+60+40)/3=46\tfrac{2}{3}$ cm.
```

#### Filters operating on multiple input channels

When one layer applies several filters, its feature maps become the **input channels** of the next layer. How can a filter in that next layer combine information from two channels? The same question arises in the first layer when the input contains several measured variables, such as the hourly air temperature and relative humidity recorded at the KNMI weather station.

The filter uses a separate row of weights for each channel. At each position, it takes the same window in both channels, computes a weighted sum for each and adds the two sums to produce one output value.

Consider the two short sequences $A=(2,1,3,2)$ and $B=(2,5,3,7)$ in {numref}`2inchannels`. Suppose the filter has weights $(1,1,0)$ for channel A and $(0,0,1)$ for channel B. In each window, it adds the first two values from A to the last value from B. We use valid convolution, so both channels must provide all three values in the window. Only the windows centred on positions 2 and 3 fit, giving two output values.

```{figure} ../figures/cnn_generic_input_channels.svg
:name: 2inchannels
:alt: The two-channel filter gives 3 from A and 3 from B at centre 2, then 4 from A and 7 from B at centre 3. The output map is 6, 11.

In the first window, A contributes $2+1$ and B contributes $3$, giving a total of $6$. One step later the same weights give $(1+3)+7=11$. These two values form one output map, $(6,11)$.
```

Additional filters can learn different relationships between these channels.

### Convolutions in 2D

Convolutions can also be applied in two or more dimensions. A major turning point for 2D CNNs came in 2012, when **AlexNet** won the ImageNet image classification competition by a wide margin {cite:p}`krizhevsky2012imagenet`. This breakthrough helped reignite interest in deep learning across AI, paving the way for rapid advances that, a decade later, led to systems such as ChatGPT and the widespread adoption of AI in everyday applications.

A 2D filter moves across rows and columns, so it can also process a physical quantity recorded on a grid, such as rainfall. Consider a $4\times4$ rainfall map. To calculate a spatial average around each cell, we centre a $3\times3$ window on it and give each of the nine values a weight of $1/9$. A one-cell border of zeros lets us use the filter at the edges. In the corner window shown in {numref}`zero_padding_2d`, four values are rainfall readings and the remaining five come from the border.

```{figure} ../figures/zero_padding_2d.svg
:name: zero_padding_2d
:alt: A three-by-three window at the bottom-right corner contains four rainfall readings and five padding zeros.

With all nine weights equal to $1/9$, the corner output is $(4+6+5+9)/9=8/3$ mm. The five zeros count in the average.
```

The corner calculation fills one cell of the output feature map. We fill the rest by sliding the filter across the grid and repeating the calculation at each position. With stride 1 and a one-cell border of zeros, every input cell can serve as the centre of a window, so the output has the same four rows and four columns as the input.

A **2D convolutional layer** uses the same sliding window, but training determines the weights. For a $3\times3$ filter on a single input channel, the layer learns nine weights and a bias $\beta$. At row $i$ and column $j$, it adds the bias to the weighted sum and applies a nonlinear activation $a$ to produce the output $h_{ij}$, just as in {eq}`conv-1d`. The indices $m$ and $n$ run over the rows and columns of the window.

$$
h_{ij} = a\!\left(\beta + \sum_{m=1}^{3}\sum_{n=1}^{3}
\omega_{mn}\,x_{i+m-2,\,j+n-2}\right).
$$

Square filters are the most common choice. To use the same equation for a square of a different size, let $K$ be its side length. If $K$ is odd, the middle cell is at row and column $(K+1)/2$ within the window. Subtracting this index from $m$ and $n$ gives the row and column offsets from the centre. At the middle cell both offsets are zero, so its weight multiplies the input at $(i,j)$.

$$
h_{ij} = a\!\left(\beta + \sum_{m=1}^{K}\sum_{n=1}^{K}
\omega_{mn}\,x_{i+m-(K+1)/2,\,j+n-(K+1)/2}\right).
$$

For $K=3$, the middle index is 2, which gives the first equation. Rectangular filters can have different numbers of rows and columns. We focus on square filters here.

(multiple-input-channels)=
#### Multiple input and output channels

As in 1D, each filter produces one output channel. On a grayscale image, for example, one filter might respond to vertical edges and another to horizontal edges. Sliding both filters across the image gives two feature maps, so the next layer receives two values at each location.

{numref}`cnn_output_tensor` shows the same idea for an image with three input channels. Follow the two filters, labelled banks A and B, to their output maps on the right. On an unpadded $4\times4$ grid, a $3\times3$ window fits in two positions across and two down at stride 1. Each filter produces a $2\times2$ map. Stacking the two maps gives an output of shape $2\times2\times2$. The first two numbers give the height and width, and the third gives the number of output channels. This is the **channels-last** convention.

```{figure} ../figures/cnn_output_tensor.svg
:name: cnn_output_tensor
:alt: Three input channels feed banks A and B. Each bank has three kernels and produces its own two-by-two map. Maps A and B stack into the next layer’s two-channel input.

Map A and map B each record one feature at four locations. Stacking them keeps the two features at each location together for the next layer.
```

Many images already have several input channels. An RGB image has red, green and blue channels, and a satellite image may record additional spectral bands. To combine information from these channels, a filter has a separate set of weights for each one. At a given location, we take the same $3\times3$ patch from each channel, calculate a weighted sum for each patch, then add the sums and the filter’s bias.

{numref}`rgb_conv` shows this calculation for three input channels at **one spatial position**. To obtain the full output feature map, we slide the filter across all three channels together and repeat the calculation at every window position. Each additional filter produces its own map in the same way.

```{figure} ../figures/cnn_multichannel_pointwise_products.svg
:name: rgb_conv
:alt: Each of three channels shows an input patch, its weights and the nine products of corresponding entries. The products sum to 5, 1 and 3 for the three channels.

The symbol $\odot$ means element-wise multiplication. Each entry in the “Pointwise products” column is a patch value multiplied by the weight in the same position. For example, the middle-left product in channel 1 is $1\times(-1)=-1$. Summing the nine products in each channel gives 5, 1 and 3. Adding the bias gives $5+1+3-2=7$ at this location.
```

#### Computing the number of learnable parameters

A filter uses the same weights wherever it goes, so we only need to count its weights once. A square filter with side length $K$ has $K^2$ weights for each input channel. If there are $C_{\mathrm{in}}$ input channels and the filter has one bias, that gives $C_{\mathrm{in}}K^2+1$ parameters for one filter. Each output channel has its own filter, so we multiply by the number of output channels, $C_{\mathrm{out}}$. In 1D, the same reasoning applies with $K$ weights per input channel.

```{math}
:label: number_of_params
\begin{aligned}
\text{1D:}\quad & C_{\mathrm{out}}(C_{\mathrm{in}} K+1),\\
\text{2D:}\quad & C_{\mathrm{out}}(C_{\mathrm{in}} K^2+1).
\end{aligned}
```

```{admonition} Example
:class: dropdown tip

Suppose a layer has 64 input channels, 128 output channels and $3\times3$ kernels. First count the parameters of one filter. It uses $3\times3=9$ weights for each input channel, giving $64\times9=576$ weights. Adding one bias brings the count to $576+1=577$ parameters per filter.

Each of the 128 output channels needs one such filter. Across the layer, that gives $128\times577=73{,}856$ learnable parameters, that is $73{,}728$ weights and $128$ biases.
```

## Pooling

A convolutional layer can produce a large feature map that records how strongly a filter responds at each location. To recognize a building in a satellite image, for example, a network may only need to know that a roof corner appears within a small region. Its exact pixel position can vary from image to image. Summarizing nearby responses can make recognition less sensitive to small shifts in the building’s position. It also reduces computation because later layers work on a smaller grid.

**Pooling** produces such a summary by replacing the values in a local window with a single representative value. A pooling layer applies the same rule independently to each feature map. Averaging captures the overall level of activity in a window, while taking the maximum preserves the strongest response, even if it occurs at only one location.

To understand how pooling works, consider a $4\times4$ rainfall map in which each cell records the amount of rain at one location. A $2\times2$ window with stride 2 divides the map into four districts without overlap. With either pooling rule, each district supplies one value to a smaller $2\times2$ map.

### Average Pooling

Consider the rainfall example described above. **Average pooling** takes the mean of all the values in a window. In the southeast district of {numref}`avg_pooling`, the four readings are 4, 6, 5 and 9 mm. Their mean is $(4+6+5+9)/4=6$ mm, so we place 6 in the southeast cell of the pooled map.

```{figure} ../figures/cnn_avg_pooling.svg
:name: avg_pooling
:alt: A four-by-four rainfall grid reduces to district means of 0.25, 2, 3 and 6 mm.

In the northwest district, only one of the four locations recorded rain. The district average is therefore a quarter of that location’s 1 mm reading.
```

### Max Pooling

On the other hand, to find the strongest response in each window, we can use **max pooling**, which keeps the largest value. In the highlighted window of the feature map, the largest value is 9 ({numref}`max_pooling`). Moving that value to another cell within the same window would leave the pooled value unchanged. We can also save the position of the cell that supplied the maximum (the *argmax*).

```{figure} ../figures/cnn_max_pooling.svg
:name: max_pooling
:alt: A 4 × 4 feature map is pooled into a 2 × 2 map with rows [1, 3] and [5, 9].

Slide a 2 × 2 window across the original feature map with stride 2. The highlighted window contains 4, 6, 5 and 9, so its maximum, 9, goes in the bottom-right cell of the pooled map.
```

### Upsampling

In later lectures, we will encounter networks that first reduce the spatial resolution of a feature map and then increase it again. **Upsampling** takes a map on a coarse grid and assigns values to a finer grid. It reverses the change in resolution caused by pooling, although the pooled values alone cannot recover all the original detail. To compare three ways of assigning values to the new cells, we can expand the $2\times2$ map of maxima to a $4\times4$ grid ({numref}`upsampling`).

1. **Nearest neighbour** fills each new cell with the value from the closest cell on the coarse grid. Here each district maximum fills a whole $2\times2$ block. For example, all four cells in the southeast receive 9.
2. **Max unpooling** uses the positions we saved during max pooling. It places 9 back in the cell where we recorded it and does the same for the other three maxima. The remaining cells receive zeros because pooling discarded their original readings.
3. **Bilinear interpolation** estimates values between coarse cells by taking a weighted average. The weights depend on how far the new position lies between its neighbours along each axis. Halfway across and halfway down, the four neighbours contribute equally, giving $(1+3+5+9)/4=4.5$. At the outer edges, this illustration uses zero for any neighbour outside the coarse grid.

```{figure} ../figures/cnn_upsampling.svg
:name: upsampling
:alt: Four pooled values are repeated in blocks, placed at saved positions, or interpolated between samples.

Each method enlarges the grid within each feature map, so the number of channels stays the same. In panel (c), the underlined values mark the four original samples.
```

A network can also learn how to distribute values across a finer grid. One way to do this is through a **transposed convolution**. In the one-dimensional example in {numref}`transposed_conv`, we multiply each input value by the same three learned weights. The products occupy three neighbouring positions in the output. Where the two groups overlap, we add their contributions. The same idea applies to a two-dimensional map, where each input contributes to a patch of output cells.

```{figure} ../figures/cnn_transposed_convolution.svg
:name: transposed_conv
:alt: Two coarse input positions contribute to five fine output positions. The middle output receives contributions from both.

Input A contributes to the first three output positions and input B to the last three. The middle position receives a contribution from each.
```

For the connection to ordinary convolution and a worked derivation, see section 10.4.2 of Simon J. D. Prince’s [*Understanding Deep Learning*](https://udlbook.github.io/udlbook/).

## Conclusions

A convolutional layer learns filters that respond to local patterns in its input. Each filter uses the same weights at every position and produces one output feature map. By combining these maps in successive layers, a CNN can learn patterns that extend over a larger part of a signal or image.

Kernel size and dilation determine which input values contribute to each response. Padding determines how the filter handles boundaries, and stride sets the spacing between outputs. Pooling can reduce the resolution further by summarizing nearby responses. When a later layer needs a finer grid, upsampling increases the resolution using the information the network has retained.

## References

```{bibliography}
:filter: docname in docnames
```
