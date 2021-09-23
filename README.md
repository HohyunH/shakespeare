# Shakespeare Language Modeling

## Character based RNN, LSTM Language model
##### 2021_1 Deep learning - Assignment#3

- weight initialization

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
