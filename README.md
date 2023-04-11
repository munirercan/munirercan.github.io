{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "afdb990f-f1fd-479c-930d-50e7e6b872e4",
   "metadata": {},
   "source": [
    "## Vision Transformer: A step by step simple explanation"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "c3d27df7-c6a5-4c0d-9bb1-fd557a8e7f60",
   "metadata": {},
   "source": [
    "Let's make a simple and lucid explanation for ViT that is described in Dosovitskiy et al paper:\n",
    "\n",
    "> Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S. and Uszkoreit, J., 2020. An image is worth 16x16 words: Transformers for image recognition at scale. ICLR 2021. https://arxiv.org/abs/2010.11929 \n",
    "\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "3906c80a-07c8-4ce9-b564-73eb09c68e8a",
   "metadata": {},
   "source": [
    "Transformers firstly used in NLP tasks and achieved a great success. Now, we will see how we can extend it to use for image recognition. Let's start to make it understandable for everyone."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "155c4546-4f09-4812-9b50-66a56d0e9ab9",
   "metadata": {},
   "source": [
    "\n",
    "###  Steps\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a9d3a846-7811-4ab9-b256-2ea30ef3081c",
   "metadata": {},
   "source": [
    "**1.** We have images, for example size of 28x28 (one channel for simplicity). Patches are created from each image. What is a patch?\n",
    "\n",
    "28x28  => 16x49 \n",
    "\n",
    "if batch size is 64 then our input is now 64x16x49\n",
    "\n",
    "Batch Vs Patch, Do not get confused.\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "7f15eff2-9073-459c-a408-f1aa4ba18ff6",
   "metadata": {},
   "source": [
    "**2.** Embedding. Patches which have length of 49 are projected into new dimensions lets say 128 by a linear layer. So 49 => 128\n",
    "\n",
    "nn.Linear(49,128)\n",
    "\n",
    "64x16x49 => 64x16x128\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "c87ebfc7-c57a-4fbf-809f-9a41ce3861bd",
   "metadata": {},
   "source": [
    "**3.** Add learnable class embedding. This is an extra patch appended to the beginning of the input. Size is the same as other patches, 1x28. Create 64 of them since batch size is 64, one for each image\n",
    "\n",
    "64x16x128 append as first row 64x1x128 = 64x17x128"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "c49fc825-01e7-4a11-915f-71c0bfe6e411",
   "metadata": {},
   "source": [
    "**4.** Add position embedding. Sum position embedding with current input (this is not append, this is sum).\n",
    "\n",
    "64x17x128 sum 64x17x128 => 64x17x128\n",
    "\n",
    "Input is ready for encoder."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "3a57d3e9-f34d-41c8-af57-9055496e45ee",
   "metadata": {},
   "source": [
    "### Transformer Encoder starts here"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a373d211-1911-4bca-ab15-67cc4aeeb93b",
   "metadata": {},
   "source": [
    "**5.** Apply Layer Normalization, dimensions do not change: 64x17x128"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "68cee7dd-f8e3-4cb2-bae6-0eca099446af",
   "metadata": {},
   "source": [
    "**6.** Create query, key and value matrices. These are created by linear projection of current input.\n",
    "\n",
    "Create Linear layers\n",
    ">qlinear = nn.Linear(128, 128)\n",
    "\n",
    ">klinear = nn.Linear(128, 128)\n",
    "\n",
    ">vlinear = nn.Linear(128, 128)\n",
    "\n",
    "Apply Linear layers:\n",
    ">query = qlinear(x)\n",
    "\n",
    ">key = klinear(x)\n",
    "\n",
    ">value = vlinear(x)\n",
    "\n",
    "Note that query, key and value are not equal, they pass through different linear layers.\n",
    "What happened to our current input: each query, key and value size is 64x17x128 => 64x17x128\n",
    "\n",
    "From now on we will use query, key and value in our coding. Input x has done its job, we no more need it."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "cc8bdc23-65a8-456f-81db-f20905d473fa",
   "metadata": {},
   "source": [
    "**7.** Multihead attention is used in the paper. Lets create our heads, think we have 4 heads. Query, key and value will be divided into 4 heads, then what we get:\n",
    "\n",
    "For query, head-1 is 64x17x32, head-2 is 64x17x32, head-3 is 64x17x32 and head-4 is 64x17x32. Lets put all 4 heads into one matrix then 64x17x4x25. This is for query, do the same calculation for key and value.\n",
    "\n",
    "Then size of query is 64x17x4x32\n",
    "\n",
    "size of key is 64x17x4x32\n",
    "\n",
    "size of value is 64x17x4x32\n",
    "\n",
    "64: number of batches (images)\n",
    "\n",
    "17: number of patches in an image + one learnable class embedding\n",
    "\n",
    "4: 4 heads\n",
    "\n",
    "32: embedding size\n",
    "\n",
    "Paper says heads are created by projection of input value. This is the same as seperating input into head counts."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "2c5fa4ed-2f55-4a2e-85fa-2a278e8102b3",
   "metadata": {},
   "source": [
    "**8.** Each head applies self attention among patches. So lets bring patches together.\n",
    "\n",
    "64x17x4x32 => 64x4x17x32\n",
    "\n",
    "17x32 is embedding of each image. Since we have 4 heads, we have 4 different embeddings for each image. "
   ]
  },
  {
   "cell_type": "markdown",
   "id": "447a8265-21e1-4494-8f4c-a1a99f9d9cab",
   "metadata": {},
   "source": [
    "**9.** It is time to calculate weights.\n",
    "\n",
    "A = softmax(query * transpose(key) / sqrt(32))\n",
    "\n",
    "64x4x17x32 * 64x4x32x17 => 64x4x17x17"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "0acf2d5a-beba-40fb-a081-a2709b4f2bf5",
   "metadata": {},
   "source": [
    "**10.** Now the weighted sum.\n",
    "\n",
    "A * value\n",
    "\n",
    "64x4x17x17 * 64x4x17x32 => 64x4x17x32"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "68e4396a-a65f-4d09-ab81-ad25a9a6d2d3",
   "metadata": {},
   "source": [
    "**11.** All heads have done the calculations, lets concatanate the heads.\n",
    "\n",
    "We have four heads 64x4x17x32, concatanate their outputs: 64x17x128"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "db3e19b3-1bb3-415b-8eb4-5e5e71d75f97",
   "metadata": {},
   "source": [
    "**12.** Add residuals : Add value at step 4\n",
    "\n",
    "64x17x128 sum 64x17x128 => 64x17x128"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "7cd14c99-14f7-4116-a0e9-49e10f7b3c48",
   "metadata": {},
   "source": [
    "**13.** Apply layer normalization: 64x17x128"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b98760e9-41fb-46ae-91a9-ac65e2ba77af",
   "metadata": {},
   "source": [
    "**14.** Create a Multilayer Perceptron and apply it.\n",
    "\n",
    ">nn.Linear(128, 128 * 4)\n",
    "\n",
    ">nn.GELU()\n",
    "\n",
    ">nn.Dropout(0.1)\n",
    "\n",
    ">nn.Linear(128 * 4, 128)\n",
    "\n",
    "Dimensions do not change: 64x17x128"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "bf3457ef-3cd0-455e-9865-5cdb243fd2e6",
   "metadata": {},
   "source": [
    "**15.** Add residuals from result of step 12: 64x17x128"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "9675a970-6d60-40c3-b560-d7141963574b",
   "metadata": {},
   "source": [
    "### Tranformer Encoder ends here"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "7284aaa9-33d8-4221-8e6b-d84ab9163845",
   "metadata": {},
   "source": [
    "**16.** Learnable class embedding that we added to our input at step 3 is used for image representation. It is the first row in our input. \n",
    "\n",
    "cls = out[:,0]\n",
    "\n",
    "cls is 64x1x128\n",
    "\n",
    "Apply MLP Head. If there are 10 classes\n",
    "\n",
    ">nn.LayerNorm(128)\n",
    "\n",
    ">nn.Linear(128, 10)\n",
    "\n",
    "Output is 64x10. We have 10 class values for each image."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "aa04d5b3-befc-48f5-af57-d5a42c18934e",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.2"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
