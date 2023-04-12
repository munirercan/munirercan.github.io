Add file
Add another file

Vision Transformer: A step by step simple explanation

Let's make a simple and lucid explanation for ViT that is described in Dosovitskiy et al paper:

    Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S. and Uszkoreit, J., 2020. An image is worth 16x16 words: Transformers for image recognition at scale. ICLR 2021. https://arxiv.org/abs/2010.11929

Transformers firstly used in NLP tasks and achieved a great success. Now, we will see how we can extend it to use for image recognition. Let's start to make it understandable for everyone.
Steps

1. We have images, for example size of 28x28 (one channel for simplicity). Patches are created from each image. What is a patch?

28x28 => 16x49

if batch size is 64 then our input is now 64x16x49

Batch Vs Patch, Do not get confused.

2. Embedding. Patches which have length of 49 are projected into new dimensions lets say 128 by a linear layer. So 49 => 128

nn.Linear(49,128)

64x16x49 => 64x16x128

3. Add learnable class embedding. This is an extra patch appended to the beginning of the input. Size is the same as other patches, 1x28. Create 64 of them since batch size is 64, one for each image

64x16x128 append as first row 64x1x128 = 64x17x128

4. Add position embedding. Sum position embedding with current input (this is not append, this is sum).

64x17x128 sum 64x17x128 => 64x17x128

Input is ready for encoder.
Transformer Encoder starts here

5. Apply Layer Normalization, dimensions do not change: 64x17x128

6. Create query, key and value matrices. These are created by linear projection of current input.

Create Linear layers

    qlinear = nn.Linear(128, 128)

    klinear = nn.Linear(128, 128)

    vlinear = nn.Linear(128, 128)

Apply Linear layers:

    query = qlinear(x)

    key = klinear(x)

    value = vlinear(x)

Note that query, key and value are not equal, they pass through different linear layers. What happened to our current input: each query, key and value size is 64x17x128 => 64x17x128

From now on we will use query, key and value in our coding. Input x has done its job, we no more need it.

7. Multihead attention is used in the paper. Lets create our heads, think we have 4 heads. Query, key and value will be divided into 4 heads, then what we get:

For query, head-1 is 64x17x32, head-2 is 64x17x32, head-3 is 64x17x32 and head-4 is 64x17x32. Lets put all 4 heads into one matrix then 64x17x4x25. This is for query, do the same calculation for key and value.

Then size of query is 64x17x4x32

size of key is 64x17x4x32

size of value is 64x17x4x32

64: number of batches (images)

17: number of patches in an image + one learnable class embedding

4: 4 heads

32: embedding size

Paper says heads are created by projection of input value. This is the same as seperating input into head counts.

8. Each head applies self attention among patches. So lets bring patches together.

64x17x4x32 => 64x4x17x32

17x32 is embedding of each image. Since we have 4 heads, we have 4 different embeddings for each image.

9. It is time to calculate weights.

A = softmax(query * transpose(key) / sqrt(32))

64x4x17x32 * 64x4x32x17 => 64x4x17x17

10. Now the weighted sum.

A * value

64x4x17x17 * 64x4x17x32 => 64x4x17x32

11. All heads have done the calculations, lets concatanate the heads.

We have four heads 64x4x17x32, concatanate their outputs: 64x17x128

12. Add residuals : Add value at step 4

64x17x128 sum 64x17x128 => 64x17x128

13. Apply layer normalization: 64x17x128

14. Create a Multilayer Perceptron and apply it.

    nn.Linear(128, 128 * 4)

    nn.GELU()

    nn.Dropout(0.1)

    nn.Linear(128 * 4, 128)

Dimensions do not change: 64x17x128

15. Add residuals from result of step 12: 64x17x128
Tranformer Encoder ends here

16. Learnable class embedding that we added to our input at step 3 is used for image representation. It is the first row in our input.

cls = out[:,0]

cls is 64x1x128

Apply MLP Head. If there are 10 classes

    nn.LayerNorm(128)

    nn.Linear(128, 10)

Output is 64x10. We have 10 class values for each image.

 

