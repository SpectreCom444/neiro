import numpy as np

def sigmoid(x):
  # Сигмоидная функция активации: f(x) = 1 / (1 + e^(-x))
  return 1 / (1 + np.exp(-x))

def deriv_sigmoid(x):
  # Производная сигмоиды: f'(x) = f(x) * (1 - f(x))
  fx = sigmoid(x)
  return fx * (1 - fx)

def mse_loss(y_true, y_pred):
  # y_true и y_pred - массивы numpy одинаковой длины.
  return ((y_true - y_pred) ** 2).mean()

class OurNeuralNetwork:
  def __init__(self):
    # Веса
    self.w=[np.random.normal() for i in range(36)]
    
    self.ws1 = np.random.normal()
    self.ws2 = np.random.normal()
    self.ws3 = np.random.normal()
    self.ws4 = np.random.normal()
    # Пороги
    self.b=[np.random.normal() for i in range(4)]
        
    self.bs1 = np.random.normal()


  def feedforward(self, x):
    # x is a numpy array with 2 elements.
    h=[0 for i in range(4)]
    counter=0
    for i in range(4):
        kof=0
        for j in range(9):
            kof+=self.w[counter] * x[j] 
            counter+=1
        h[i] = sigmoid(kof+ self.b[i])
    
    o1 = sigmoid(self.ws1 * h[0] + self.ws2 * h[1]+ self.ws3 * h[2]+ self.ws4 * h[3] + self.bs1)
    return o1

  def train(self, data, all_y_trues):
    learn_rate = 0.1
    epochs = 3000 # сколько раз пройти по всему набору данных 

    for epoch in range(epochs):
      for x, y_true in zip(data, all_y_trues):
        # --- Прямой проход (эти значения нам понадобятся позже)
        sum_h=[0 for i in range(4)]
        h=[0 for i in range(4)]
        counter=0
        for i in range(4):
            sum_h[i]=self.b[i]
            for j in range(9):
                sum_h[i]+=self.w[counter] * x[j] 
                counter+=1
            h[i] = sigmoid(sum_h[i])
        

        sum_o1 = self.ws1 * h[0] + self.ws2 * h[1]+ self.ws3 * h[2]+ self.ws4 * h[3] + self.bs1
        o1 = sigmoid(sum_o1)
        y_pred = o1

        # --- Считаем частные производные.
        # --- Имена: d_L_d_w1 = "частная производная L по w1"
        d_L_d_ypred = -2 * (y_true - y_pred)

        # Нейрон o1
        d_ypred_d_ws1 = h[0] * deriv_sigmoid(sum_o1)
        d_ypred_d_ws2 = h[1] * deriv_sigmoid(sum_o1)
        d_ypred_d_ws3 = h[2] * deriv_sigmoid(sum_o1)
        d_ypred_d_ws4 = h[3] * deriv_sigmoid(sum_o1)
        
        d_ypred_d_bs1 = deriv_sigmoid(sum_o1)

        d_ypred_d_h1 = self.ws1 * deriv_sigmoid(sum_o1)
        d_ypred_d_h2 = self.ws2 * deriv_sigmoid(sum_o1)
        d_ypred_d_h3 = self.ws3 * deriv_sigmoid(sum_o1)
        d_ypred_d_h4 = self.ws4 * deriv_sigmoid(sum_o1)

        # Нейрон h1
        d_h1_d_w=[x[i] * deriv_sigmoid(sum_h[0]) for i in range(9)]
        d_h1_d_b1 = deriv_sigmoid(sum_h[0])

        # Нейрон h2

        d_h2_d_w=[x[i] * deriv_sigmoid(sum_h[1]) for i in range(9)]
        d_h2_d_b2 = deriv_sigmoid(sum_h[1])
        
        # Нейрон h3
        d_h3_d_w=[x[i] * deriv_sigmoid(sum_h[2]) for i in range(9)]
        d_h3_d_b3 = deriv_sigmoid(sum_h[2])
        
        # Нейрон h4
        d_h4_d_w=[x[i] * deriv_sigmoid(sum_h[3]) for i in range(9)]
        d_h4_d_b4 = deriv_sigmoid(sum_h[3])

        # --- Обновляем веса и пороги
        # Нейрон h1
        counter=0
        for i in range(9):
            self.w[counter] -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w[i]
            counter+=1
            
        self.b[0] -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_b1

        # Нейрон h2
        for i in range(9):
            self.w[counter] -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w[i]
            counter+=1
            
        self.b[1] -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_b2

        # Нейрон h3
        for i in range(9):
            self.w[counter] -= learn_rate * d_L_d_ypred * d_ypred_d_h3 * d_h3_d_w[i]
            counter+=1
            
        self.b[2] -= learn_rate * d_L_d_ypred * d_ypred_d_h3 * d_h3_d_b3
        
        # Нейрон h4
        for i in range(9):
            self.w[counter] -= learn_rate * d_L_d_ypred * d_ypred_d_h4 * d_h4_d_w[i]
            counter+=1
            
        self.b[3] -= learn_rate * d_L_d_ypred * d_ypred_d_h4 * d_h4_d_b4
        
        # Нейрон o1
        self.ws1 -= learn_rate * d_L_d_ypred * d_ypred_d_ws1
        self.ws2 -= learn_rate * d_L_d_ypred * d_ypred_d_ws2
        self.ws3 -= learn_rate * d_L_d_ypred * d_ypred_d_ws3
        self.ws4 -= learn_rate * d_L_d_ypred * d_ypred_d_ws4
        self.bs1 -= learn_rate * d_L_d_ypred * d_ypred_d_bs1

      # --- Считаем полные потери в конце каждой эпохи
      if epoch % 500 == 0:
        y_preds = np.apply_along_axis(self.feedforward, 1, data)
        loss = mse_loss(all_y_trues, y_preds)
        print("Epoch %d loss: %.3f" % (epoch, loss))

# Определим набор данных
data = np.array([
  [0, 1, 0, 
   0, 1, 0,
   0, 1, 0,],  
   
   [1, 0, 0, 
    1, 0, 0,
    1, 0, 0,],
   
   [0, 0, 1, 
    0, 0, 1,
    0, 0, 1,],
   
    [1, 0, 0, 
    0, 0, 0,
    1, 0, 0,],  
   
   [0, 0, 0, 
    0, 0, 1,
    0, 0, 1,],
   
   [0, 1, 0, 
    0, 1, 0,
    0, 0, 0,],
    
   [1, 1, 0, 
   0, 0, 1,
   0, 1, 0,],  
   
   [0, 1, 0, 
    0, 1, 0,
    0, 1, 0,],
   
   [0, 1, 0, 
    1, 1, 1,
    0, 1, 0,],
   
    [1, 0, 0, 
    1, 0, 0,
    1, 0, 0,],  
   
   [1, 0, 1, 
    0, 0, 0,
    1, 0, 1,],
   
   [0, 1, 1, 
    1, 0, 0,
    0, 1, 1,],
  
])
all_y_trues = np.array([
  1, 
  1, 
  1, 
  0,
  0, 
  0, 
  0, 
  1, 
  1, 
  1,
  0, 
  0,
])

# Обучаем нашу нейронную сеть!
network = OurNeuralNetwork()
network.train(data, all_y_trues)

# Делаем пару предсказаний
one = np.array(
    [1, 1, 1, 
     0, 0, 0,
     0, 1, 0,]) 
two = np.array(
    [0, 1, 0, 
     0, 1, 1,
     0, 1, 0,])  
print("1: %.3f" % network.feedforward(one))
print("2: %.3f" % network.feedforward(two))
