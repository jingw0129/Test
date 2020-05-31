import os
import torch
from mnist_net import Net
from torchvision import datasets, transforms
import cv2

model_path = 'mnist_model/18model.pth'

def test():
  correct_count, all_count = 0, 0
  for images, labels in valloader:
      for i in range(len(labels)):
          img = images[i].view(1, 784)
          with torch.no_grad():
              logps = net(img)

          ps = torch.exp(logps)
          probab = list(ps.numpy()[0])
          pred_label = probab.index(max(probab))
          true_label = labels.numpy()[i]
          if (true_label == pred_label):
              correct_count += 1
          all_count += 1

  print("Number Of Images Tested =", all_count)
  print("\nModel Accuracy =", (correct_count / all_count))

# test()


# -----------------------------------------------------------------------------------------
class SimpleInfer(object):

    def __init__(self):
        self.load_model()
        self.model = None

    def load_model(model_path: str):

        if os.path.exists(model_path) is False:
            raise NotImplementedError

    def infer(inp: list):
        net = Net()
        net.load_state_dict(torch.load(model_path)['net'], strict = False)
        net.eval()
        torch.manual_seed(1)
        inp = cv2.cvtColor(inp, cv2.COLOR_BGR2GRAY)
        transform = transforms.Normalize((0.5,), (0.5,))


        inp = cv2.resize(inp, (28,28), interpolation=cv2.INTER_CUBIC)
        inp = torch.tensor(inp, dtype = torch.float)

        inp = inp.unsqueeze(0)
        inp = transform(inp)
        # print(inp.shape)
        # inp = inp.transpose(2, 0, 1)
        # print(type(inp))

        img = inp.view(1, 784)
        with torch.no_grad():
            logps = net(img)

        ps = torch.exp(logps)
        probab = list(ps.numpy()[0])
        pred_label = probab.index(max(probab))

        return [pred_label]
        assert 'Model is not None'
        raise NotImplementedError



# model = SimpleInfer.load_model(model_path)
image = cv2.imread('./test_num.png')
out_put = SimpleInfer.infer(image)
# cv2.imshow('num', image)
# cv2.waitKey()
print('pred', out_put)
