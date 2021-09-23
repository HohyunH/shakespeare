# Shakespeare Language Modeling

### Character based RNN, LSTM Language model

- weight initialization

  def init_hidden(self, batch_size):
    weight = next(self.parameters()).data
    hidden = weight.new(self.n_layers, batch_size, self.n_hidden).zero_()
    return hidden


- loss graph in training

![image](https://user-images.githubusercontent.com/46701548/134507277-6d9082b2-b527-4e6f-a1af-e9d767591d05.png)

### Softmax Temperature
In this experiment, a comparative experiment was conducted on three softmax values that change through temperature.
