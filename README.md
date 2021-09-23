# Shakespeare Language Modeling

## Character based RNN, LSTM Language model
#### 2021_1 Deep learning - Assignment#3

- main.py
<pre>
<code>
python main.py --model charrnn --batch_size 256 --hidden 256 --num_layer 4
</code>
</pre>

#### weight initialization

1. in RNN
<pre>
<code>
def init_hidden(self, batch_size):
  weight = next(self.parameters()).data
  hidden = weight.new(self.n_layers, batch_size, self.n_hidden).zero_()
  
  return hidden
</code>
</pre>

2. in LSTM
<pre>
<code>
def init_hidden(self, batch_size):
  weight = next(self.parameters()).data
  hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_(),
            weight.new(self.n_layers, batch_size, self.n_hidden).zero_())
  
  return hidden
</code>
</pre>

-> in LSTM, we need to initialze hidden state and cell state


- loss graph in training

![image](https://user-images.githubusercontent.com/46701548/134507277-6d9082b2-b527-4e6f-a1af-e9d767591d05.png)

### Softmax Temperature
In this experiment, a comparative experiment was conducted on three softmax values that change through temperature.

![image](https://user-images.githubusercontent.com/46701548/134512344-f5ccf1b3-7498-4bb9-89b5-1890a7d1c54e.png)

1. generate length : 30
2. model : char_LSTM
3. seed character : N

- T = 0.1
-> NIUS:
   What is the gods be so.
   
   ![image](https://user-images.githubusercontent.com/46701548/134512647-8c60f412-2165-46f9-9c27-96075dc483fc.png)

- T = 1
->NNE:
  Why patieres; that's sort
  
  ![image](https://user-images.githubusercontent.com/46701548/134512780-01c2d481-f402-40b8-959f-0242d96445a6.png)

- T = 10
-> NFGHdmC! NPIzkvMKpImk
   DfeobpOq
   
   ![image](https://user-images.githubusercontent.com/46701548/134512896-2fe4b644-aef6-4db5-9597-842b9c5b2bbf.png)

### Generating Example

- Apply the seed character to be used for the test to one-hot-encoding and output the result by calling the next character as much as specified.

![image](https://user-images.githubusercontent.com/46701548/134513106-454509ba-6e29-47d8-90ae-ab8c396e3c18.png)


![image](https://user-images.githubusercontent.com/46701548/134513145-797ef62f-fe75-4510-9281-79274545602c.png)

