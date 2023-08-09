# Learn to Transformer

## Sequence to Sequence Learning

1. The number of items to the input and the number of items to the output need not be the same.
Ex) Je suis etudiant -> I am a student

2. Main idea: encoder와 decoder
- Encoder: How to process and store the entered information
-> By made context vector
- Decoder: How to unpack and return compressed information from the encoder
-> By taken context vector and made output sequence item by item

## Attention
Attention allows the model to focus on the relevant part of the input sequence as needed.

### Attention model differs from a classic Seq2Seq model in 2 main ways:
1. The encoder passes a lot more data to the decoder
: Instead of passing the last hidden state of the encoding stage, the encoder passes all the hidden states to the decoder.
2. An attention decoder does an extra step before producing its output.
: Look at the set of encoder hidden states it received - each encoder hidden states is most associated with a certain word 

## Transformer

### Self-Attention
 1. Query, Key, Value
First, to calculate the connection between the two, it internalizes Query and Key. This internal value is called the "Attention score". However, if the dimension of Query and Key increases, the internal value called Attention score also increases, which makes it difficult to learn the model. Therefore, to solve this problem, we perform scaling to divide the root of dimension d_k. This stage is called "Scaled dot-product Attention". Then, we go through the softmax active function to normalize the values, and finally internalize the score and value matrices calculated so far for correction, and we get the final attention matrix.
![enter image description here](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https://blog.kakaocdn.net/dn/yVVfe/btrTzCrzFGc/Zh23AOAdSZiNgMzmU7KsF0/img.png)
![enter image description here](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https://blog.kakaocdn.net/dn/esNIJL/btrTu3cUZiT/pl72Q5rMS4jjHmTvXTo0n1/img.png)

### Multi-head Attention: To calculate and concatenate in installments
Previously, learning was done through a single attachment, but using multiple attachments in parallel improved performance.
Transformer calculates by dividing each attribute in parallel by the number of heads. The derived Attention Values are merged into one through concatenate at the end. This results in the same size as using Attention once.
So Multi-head Attention is refers to multiple self-attention.![enter image description here](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https://blog.kakaocdn.net/dn/PrYWZ/btrTGZz0eAH/KhrIxmbrDlU8CueeztcEJK/img.png)
There are four heads, it means that each computation process is only a quarter of the number of heads is four. Therefore, Query, Key, and Value, which were sized [4x8] above, are divided into four equal parts to make [4x2]. This naturally results in each Attention Value being [4x2]. If you concatenate these Attention Values at the end, the size becomes [4x8] as shown in the figure above, which is the same as the result of a typical Attention mechanism.

### Position encoding: the way to add information about sequence in transformer
Positional encoding = Positinal embedding vector + Word embedding vector

* Conditions for positional embedding
1. For efficient learning of the model, the scale must be within a certain range. If the scale range of positional embedding is too large, the influence of its own value becomes too large, and other values (such as word embedding) are ignored and learning is not performed.  
2. Output should be derived regardless of the size of the input data. If output can be derived only when the size of input data is 10, output cannot be derived for input data of other sizes.
-> The function that satisfies all of these conditions is the "triangle function" as shown in the graph below. First, the trigonometric function's output value ranges from -1 to 1. Second, since it is a periodic function, output values can be derived regardless of the x value. 
If you use the linear function, the larger the x value (the more the word is after the sentence), the greater the positional embedding value, which can make learning difficult.

However, the periodic function has the disadvantage of overlapping information because the y value is repeated periodically. To solve this problem, we use both cos and sin functions, and alternately sin and cos functions for each dimension of embedding.
If the embedded dimension is even, we use the sin function and the odd cos function. Add the positional embedding vector obtained in this way to the word embedding vector, the "positional encoding" task is completed.

> Written with [StackEdit](https://stackedit.io/).
